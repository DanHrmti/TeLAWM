"""TD-MPC-style planning agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule

from agents.modules.actor_critic import QNetwork, VNetwork, Actor


def gumbel_softmax_sample(p, temperature=1.0, dim=0):
	logits = p.log()
	# Generate Gumbel noise
	gumbels = (
		-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
	)  # ~Gumbel(0,1)
	gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
	y_soft = gumbels.softmax(dim)
	return y_soft.argmax(-1)


class TDMPC(nn.Module):
    """
    Model-predictive control over primitive actions with a learned dynamics model and goal-conditioned value function.

      - A one-step dynamics model s_{t+1} = f(s_t, a_t) is trained alongside goal-conditioned IQL.
      - act(): MPPI over horizon_plan-step action sequences (num_pi_trajs of them seeded by the IQL policy
        pi(a | s, g)), scored by V(s, g) along the rollout; the first plan_execute_steps actions are executed before
        replanning.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda:0')

        #### Hyperparameters ####
        # general
        self.grad_clip_norm = self.cfg.grad_clip_norm
        self.pretrained_vf = self.cfg.tdmpc["pretrained_vf"]
        self.pretrained_policy = self.cfg.tdmpc["pretrained_policy"]
        # planning
        self.horizon_plan = self.cfg.tdmpc["horizon_plan"]
        self.plan_execute_steps = min(self.cfg.tdmpc["plan_execute_steps"], self.horizon_plan)
        self.iterations = self.cfg.tdmpc["iterations"]
        self.num_pi_trajs = self.cfg.tdmpc["num_pi_trajs"]
        self.pi_noise = self.cfg.tdmpc['pi_noise']
        self.pi_noise_clip = self.cfg.tdmpc['pi_noise_clip']
        self.num_samples = self.cfg.tdmpc["num_samples"]
        self.num_elites = self.cfg.tdmpc["num_elites"]
        self.temperature = self.cfg.tdmpc["temperature"]
        self.min_std = self.cfg.tdmpc["min_std"]
        self.max_std = self.cfg.tdmpc["max_std"]
        self.planning_score = self.cfg.tdmpc["planning_score"]
        # iql
        self.tau = self.cfg.tdmpc['tau']
        self.expectile = self.cfg.tdmpc['expectile']
        self.alpha = self.cfg.tdmpc['alpha']
        self.pi_extract_type = self.cfg.tdmpc["pi_extract_type"]
        self.pi_subgoal = self.cfg.tdmpc["pi_subgoal"]

        #### Dimensions ####
        self.obs_dim = cfg.obs_shape[-1]
        self.action_dim = cfg.action_dim
        h_dim = self.cfg.arch_mlp["h_dim"]

        #### Value Functions ####
        # iql value
        self.vf = VNetwork(self.cfg, self.device)
        self.vf_detach = VNetwork(self.cfg, self.device)
        from_module(self.vf).data.to_module(self.vf_detach)  # Copy params to vf_detach without grad

        # iql q-value
        qf1 = QNetwork(self.cfg, self.device)
        qf2 = QNetwork(self.cfg, self.device)
        self.qnet_params = from_modules(qf1, qf2, as_module=True)
        self.qnet_target_params = self.qnet_params.data.clone()
        # discard params of net
        self.qnet = QNetwork(self.cfg, device="meta")
        self.qnet_params.to_module(self.qnet)

        #### Prior Policy ####
        self.pi = Actor(self.cfg, self.device)
        self.pi_detach = Actor(self.cfg, self.device)
        from_module(self.pi).data.to_module(self.pi_detach)  # Copy params to pi_detach without grad

        #### Dynamics Model ####
        self.dynamics = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, h_dim, device=self.device),
            nn.GELU(),
            nn.LayerNorm(h_dim, device=self.device),
            nn.Linear(h_dim, h_dim, device=self.device),
            nn.GELU(),
            nn.LayerNorm(h_dim, device=self.device),
            nn.Linear(h_dim, h_dim, device=self.device),
            nn.GELU(),
            nn.LayerNorm(h_dim, device=self.device),
            nn.Linear(h_dim, h_dim, device=self.device),
            nn.GELU(),
            nn.LayerNorm(h_dim, device=self.device),
            nn.Linear(h_dim, self.obs_dim, device=self.device),
        )

        #### Pretrained Weight Initialization ####
        if self.cfg.train_from_checkpoint:
            self.load(self.cfg.checkpoint)
        if self.pretrained_vf:
            print(f"Loading pretrained value function from {self.cfg.vf_chkpt}")
            vf_state_dict = torch.load(self.cfg.vf_chkpt)
            self.vf.load_state_dict(vf_state_dict["vf"])
            self.qnet_params.load_state_dict(vf_state_dict["qfs"])
        if self.pretrained_policy:
            print(f"Loading pretrained policy from {self.cfg.policy_chkpt}")
            policy_state_dict = torch.load(self.cfg.policy_chkpt)
            self.pi.load_state_dict(policy_state_dict["pi"])
        
        #### Optimizers ####
        self.dynamics_optimizer = optim.Adam(
            list(self.dynamics.parameters()), lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile
        )

        if not self.pretrained_vf:
            self.v_optimizer = optim.Adam(
                list(self.vf.parameters()), lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile
            )

            self.q_optimizer = optim.Adam(
                self.qnet_params.values(include_nested=True, leaves_only=True),
                lr=self.cfg.lr,
                capturable=self.cfg.cudagraphs and not self.cfg.compile,
            )

        if not self.pretrained_policy:
            self.actor_optimizer = optim.Adam(
                list(self.pi.parameters()), lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile
            )

        #### torch.compile ####
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            self._update_dynamics = torch.compile(self._update_dynamics, mode=mode)
            if not self.pretrained_vf:
                self._update_q = torch.compile(self._update_q, mode=mode)
                self._update_v = torch.compile(self._update_v, mode=mode)
            if not self.pretrained_policy:
                self._update_pi = torch.compile(self._update_pi, mode=mode)
            self._plan = torch.compile(self._plan, mode="reduce-overhead")

        if self.cfg.cudagraphs:
            self._update_dynamics = CudaGraphModule(self._update_dynamics, in_keys=[], out_keys=[], warmup=5)
            if not self.pretrained_vf:
                self._update_q = CudaGraphModule(self._update_q, in_keys=[], out_keys=[], warmup=5)
                self._update_v = CudaGraphModule(self._update_v, in_keys=[], out_keys=[], warmup=5)
            if not self.pretrained_policy:
                self._update_pi = CudaGraphModule(self._update_pi, in_keys=[], out_keys=[], warmup=5)

        self._plan_step = 0
        self._planned_actions = torch.nn.Buffer(torch.zeros(self.horizon_plan, self.action_dim, device=self.device))
        self._prev_mean = torch.nn.Buffer(torch.zeros(self.horizon_plan, self.action_dim, device=self.device))
        self._traj_tensor = torch.nn.Buffer(torch.empty(self.horizon_plan, self.num_samples, self.obs_dim, device=self.device))
        self._flat_goals = torch.nn.Buffer(torch.empty(self.horizon_plan * self.num_samples, self.obs_dim, device=self.device))

    def _forward_dynamics(self, obs, action):
        return self.dynamics(torch.cat([obs, action], dim=-1))

    @torch.no_grad()
    def act(self, td):
        td = td.to(self.device, non_blocking=True)
        step = td["step"]

        if step == 0 or self._plan_step >= self.plan_execute_steps:
            plan = self._plan(td)
            self._planned_actions.copy_(plan["actions"])
            self._plan_step = 0

        action = self._planned_actions[self._plan_step]
        self._plan_step = self._plan_step + 1

        info = {"planned_actions": self._planned_actions.detach().clone()}

        return action, info

    @torch.no_grad()
    def _estimate_value(self, obs, goal, actions):
        """Estimate value of trajectories by rolling out dynamics and scoring with V(s, g)."""
        horizon = self.horizon_plan
        num_traj = self.num_samples
        traj_tensor = self._traj_tensor
        _obs = obs
        for t in range(horizon):
            _obs = self._forward_dynamics(_obs, actions[t])
            traj_tensor[t] = _obs

        flat_states = traj_tensor.reshape(horizon * num_traj, self.obs_dim)
        self._flat_goals[:] = goal.repeat(horizon, 1)
        flat_values = self.vf_detach(flat_states, self._flat_goals).view(horizon, num_traj)

        if self.planning_score == "best":
            traj_scores, _ = flat_values.max(dim=0)
        elif self.planning_score == "last":
            traj_scores = flat_values[-1]
        elif self.planning_score == "all":
            traj_scores = flat_values.sum(dim=0)
        else:
            raise ValueError(
                f"Unknown planning_score '{self.planning_score}' (expected 'best', 'last', or 'all')."
            )

        return traj_scores.unsqueeze(-1)

    @torch.no_grad()
    def _plan(self, td):
        """
        Plan a sequence of actions using the learned world model and MPPI.
        """
        step = td["step"]
        obs = td["obs"]
        goal = td["goal"]

        # Sample policy trajectories
        if self.num_pi_trajs > 0:
            pi_actions = torch.empty(self.horizon_plan, self.num_pi_trajs, self.cfg.action_dim, device=self.device)
            _obs = obs.repeat(self.num_pi_trajs, 1)
            _goal = goal.repeat(self.num_pi_trajs, 1)
            for t in range(self.horizon_plan-1):
                pi_out = self.pi_detach(_obs, _goal)
                noise = torch.randn_like(pi_out)
                clipped_noise = noise.mul(self.pi_noise).clamp(-self.pi_noise_clip, self.pi_noise_clip)  # NOTE: action scale is assumed to be 1
                action = (pi_out + clipped_noise).clamp(self.cfg.action_low, self.cfg.action_high)
                pi_actions[t] = action
                _obs = self._forward_dynamics(_obs, pi_actions[t])
            pi_actions[-1] = self.pi_detach(_obs, _goal)

        # Initialize state and parameters
        obs = obs.repeat(self.num_samples, 1)
        goal = goal.repeat(self.num_samples, 1)
        mean = torch.zeros(self.horizon_plan, self.cfg.action_dim, device=self.device)
        std = torch.full((self.horizon_plan, self.cfg.action_dim), self.max_std, dtype=torch.float, device=self.device)
        if step > 0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(self.horizon_plan, self.num_samples, self.cfg.action_dim, device=self.device)
        if self.num_pi_trajs > 0:
            actions[:, :self.num_pi_trajs] = pi_actions

        # Iterate MPPI
        for _ in range(self.iterations):

            # Sample actions
            r = torch.randn(self.horizon_plan, self.num_samples-self.num_pi_trajs, self.cfg.action_dim, device=std.device)
            actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, self.num_pi_trajs:] = actions_sample

            # Compute elite actions
            value = self._estimate_value(obs, goal, actions).nan_to_num(0)
            elite_idxs = torch.topk(value.squeeze(1), self.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0).values
            score = torch.exp(self.temperature*(elite_value - max_value))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
            std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
            std = std.clamp(self.min_std, self.max_std)

        # Select action
        rand_idx = gumbel_softmax_sample(score.squeeze(1))  # gumbel_softmax_sample is compatible with cuda graphs
        actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
        a, std = actions[0], std[0]  # use mean of the action distribution
        self._prev_mean.copy_(mean)
        
        return TensorDict(actions=actions.detach())

    def update(self, batch, step):
        # update dynamics model
        info = self._update_dynamics(batch)

        if not self.pretrained_vf:
            # update value
            info.update(self._update_v(batch))
            # update q-value
            info.update(self._update_q(batch))
            # update the target networks
            # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
            self.qnet_target_params.lerp_(self.qnet_params.data, self.tau)
        
        if not self.pretrained_policy:
            # update actor
            info.update(self._update_pi(batch))
        
        return info.detach().mean()

    def _update_dynamics(self, batch):
        obs = batch["obs"]
        action = batch["action"]
        next_obs = batch["next_obs"]

        next_obs_pred = self._forward_dynamics(obs, action)
        dynamics_loss = F.mse_loss(next_obs_pred, next_obs)

        self.dynamics_optimizer.zero_grad()
        dynamics_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), self.grad_clip_norm)
        self.dynamics_optimizer.step()

        return TensorDict(dynamics_loss=dynamics_loss.detach(), dynamics_gradnorm=grad_norm)

    def _update_q(self, batch):
        obs = batch["obs"]
        action = batch["action"]
        next_obs = batch["next_obs"]
        reward = batch["reward"]
        done = batch["done"]
        goal = batch["value_goal"]
        
        vf_next_target = self.vf_detach(next_obs, goal)
        next_q_value = reward.flatten() + (1 - done.flatten()) * self.cfg.gamma * vf_next_target.flatten()  # whether a "done" signal is recieved depends on config
        next_q_value = torch.clamp(next_q_value, min=-1/(1 - self.cfg.gamma), max=0)  # NOTE: clamp to possible range based on the reward, assumes reward in [-1, 0]

        qf_loss = torch.vmap(self._batched_qf, (0, None, None, None, None))(self.qnet_params, obs, action, goal, next_q_value)
        qf_loss = qf_loss.sum(0)

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.qnet_params.values(include_nested=True, leaves_only=True), self.grad_clip_norm)
        self.q_optimizer.step()

        return TensorDict(qf_loss=qf_loss.detach(), qf_gradnorm=grad_norm)

    def _batched_qf(self, params, obs, action, goal, next_q_value=None):
            with params.to_module(self.qnet):
                vals = self.qnet(obs, action, goal)
                if next_q_value is not None:
                    loss_val = F.mse_loss(vals.view(-1), next_q_value)
                    return loss_val
                return vals
            
    def _update_v(self, batch):
        obs = batch["obs"]
        action = batch["action"]
        goal = batch["value_goal"]

        qf_target = torch.vmap(self._batched_qf, (0, None, None, None))(self.qnet_target_params, obs, action, goal)
        min_qf_target = qf_target.min(0).values

        value = self.vf(obs, goal)
        
        value_loss = torch.mean(torch.abs(self.expectile - ((min_qf_target - value) < 0).float()) * (min_qf_target - value)**2)  # expectile loss
        
        self.v_optimizer.zero_grad()
        value_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.vf.parameters(), self.grad_clip_norm)
        self.v_optimizer.step()
        
        return TensorDict(vf_loss=value_loss.detach(), vf_gradnorm=grad_norm)

    def _update_pi(self, batch):
        obs = batch["obs"]
        action = batch["action"]
        goal = batch["policy_subgoal"] if self.pi_subgoal else batch["policy_goal"]

        # NOTE: different from OGBench in that uses qf1 and not min(qf1, qf2) for policy training
        self.actor_optimizer.zero_grad()

        pi_out = self.pi(obs, goal)
        bc_losses = F.mse_loss(pi_out, action, reduction='none').mean(dim=-1, keepdim=True)  # [B, 1]

        if self.pi_extract_type == "ddpgbc":
            with self.qnet_params.data[0].to_module(self.qnet):
                q_out = self.qnet(obs, pi_out, goal)
                q_loss = -q_out.mean() / (q_out.abs().mean().detach() + 1e-6)
                
                bc_loss = bc_losses.mean()
                pi_loss = q_loss + self.alpha * bc_loss
        
        elif self.pi_extract_type == "awr":
            qf_target = torch.vmap(self._batched_qf, (0, None, None, None))(self.qnet_target_params, obs, action, goal)
            min_qf_target = qf_target.min(0).values
            value = self.vf_detach(obs, goal)
            adv = min_qf_target - value
            exp_adv = torch.exp(self.alpha * adv.detach()).clamp(max=100.0)
            pi_loss = torch.mean(exp_adv * bc_losses)
        
        else:
            raise ValueError(f"Invalid pi_extract_type: {self.pi_extract_type}")

        pi_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.pi.parameters(), self.grad_clip_norm)
        self.actor_optimizer.step()

        return_dict = TensorDict(pi_loss=pi_loss.detach(), pi_gradnorm=grad_norm)
        if self.pi_extract_type == "ddpgbc":
            return_dict["pi_bc_loss"] = bc_loss.detach()
            return_dict["pi_q_loss"] = q_loss.detach()
            return_dict["q_scale"] = q_out.abs().mean().detach()
        elif self.pi_extract_type == "awr":
            return_dict["pi_adv"] = adv.mean().detach()
            return_dict["pi_adv_max"] = adv.max().detach()
            return_dict["pi_adv_min"] = adv.min().detach()
        return return_dict

    def save(self, fp, obs_mean=0, obs_std=1):
        """
        Save state dict of the agent to filepath.

        Args:
            fp (str): Filepath to save state dict to.
        """
        torch.save({
            "dynamics": self.dynamics.state_dict(),
            "vf": self.vf.state_dict(),
            "qfs": self.qnet_params.state_dict(),
            "pi": self.pi.state_dict(),
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            }, fp)

    def load(self, fp):
        """
        Load a checkpoint saved by save() into the agent.

        Args:
            fp (str or Path): checkpoint file.
        """
        state_dict = torch.load(fp)
        self.dynamics.load_state_dict(state_dict["dynamics"])
        if not self.pretrained_vf:
            self.vf.load_state_dict(state_dict["vf"])
            self.qnet_params.load_state_dict(state_dict["qfs"])
            self.qnet_target_params = self.qnet_params.data.clone()
            self.qnet_params.to_module(self.qnet)
        if not self.pretrained_policy:
            self.pi.load_state_dict(state_dict["pi"])
        self.obs_mean = state_dict["obs_mean"]
        self.obs_std = state_dict["obs_std"]

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
