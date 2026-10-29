"""Action-Chunk World Model (ACWM) agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule

from agents.modules.actor_critic import QNetwork, VNetwork, Actor
from agents.modules.diffusion_policy import MLPDiffusionPolicy


class DynamicsModel(nn.Module):
    """D(s_{t+K} | s_t, a_{t:t+K-1}): MLP from a state and a flattened action chunk to the state K steps later."""

    def __init__(self, cfg, device=None):
        super().__init__()
        obs_dim, ac_length, a_dim, h_dim = cfg.obs_shape[-1], cfg.acwm['wm_timescale'], cfg.action_dim, cfg.arch_mlp['h_dim']

        self.dynamics = nn.Sequential(
            nn.Linear(obs_dim + ac_length * a_dim, h_dim, device=device),
            nn.GELU(),
            nn.LayerNorm(h_dim, device=device),
            nn.Linear(h_dim, h_dim, device=device),
            nn.GELU(),
            nn.LayerNorm(h_dim, device=device),
            nn.Linear(h_dim, h_dim, device=device),
            nn.GELU(),
            nn.LayerNorm(h_dim, device=device),
            nn.Linear(h_dim, h_dim, device=device),
            nn.GELU(),
            nn.LayerNorm(h_dim, device=device),
            nn.Linear(h_dim, obs_dim, device=device),
        )

    def forward(self, obs, action_chunk):
        # obs: (B, obs_dim)
        # action_chunk: (B, K * action_dim) — pre-flattened, K = wm_timescale
        x = torch.cat([obs, action_chunk], dim=-1)
        return self.dynamics(x)


class ACWM(nn.Module):
    """
    Plans subgoals by rolling out action chunks through a dynamics model.

      - BC policy: a diffusion policy p(a_{t:t+K-1} | s_t) over action chunks (K = wm_timescale).
      - Dynamics model D(s_{t+K} | s_t, a_{t:t+K-1}): predicts the state the chunk leads to.
      - Planning: sample action chunks from the BC policy, roll out with D for planning_steps steps, score the
        states with the value function toward the goal.
      - The chosen subgoal is handed to a low-level policy pi(a | s, subgoal); the value function and low-level
        policy are goal-conditioned IQL, frozen (pretrained) or trained jointly.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda:0')

        #### Hyperparameters ####
        self.grad_clip_norm = self.cfg.grad_clip_norm
        self.pretrained_vf = self.cfg.acwm["pretrained_vf"]
        self.pretrained_policy = self.cfg.acwm["pretrained_policy"]
        # subgoals / planning
        self.n_sg_samples = self.cfg.acwm["n_sg_samples"]
        self.planning_steps = self.cfg.acwm["planning_steps"]
        self.planning_score = self.cfg.acwm["planning_score"]
        self.subgoal_state = self.cfg.acwm["subgoal_state"]
        self.action_chunk_repeat = self.cfg.acwm["action_chunk_repeat"]
        self.subgoal_reached_threshold = self.cfg.acwm["subgoal_reached_threshold"]
        self.goal_vicinity_threshold = self.cfg.acwm["goal_vicinity_threshold"]
        self.wm_timescale = self.cfg.acwm["wm_timescale"]  # action chunk length K
        self.subgoal_steps = self.cfg.acwm["subgoal_steps"]  # test-time replan interval
        self.filter_subgoals = self.cfg.acwm["filter_subgoals"]
        self.debug_subgoals = self.cfg.acwm["debug_subgoals"]
        self.disable_subgoal = self.cfg.acwm["disable_subgoal"]
        # iql
        self.tau = self.cfg.acwm['tau']
        self.expectile = self.cfg.acwm['expectile']
        self.alpha = self.cfg.acwm['alpha']
        self.pi_extract_type = self.cfg.acwm["pi_extract_type"]
        self.pi_subgoal = self.cfg.acwm["pi_subgoal"]
        self.policy_type = self.cfg.acwm['policy_type']
        self.pi_noise = self.cfg.acwm['pi_noise']
        self.pi_noise_clip = self.cfg.acwm['pi_noise_clip']

        #### Value Functions ####
        self.vf = VNetwork(self.cfg, self.device)
        self.vf_detach = VNetwork(self.cfg, self.device)
        from_module(self.vf).data.to_module(self.vf_detach)

        qf1 = QNetwork(self.cfg, self.device)
        qf2 = QNetwork(self.cfg, self.device)
        self.qnet_params = from_modules(qf1, qf2, as_module=True)
        self.qnet_target_params = self.qnet_params.data.clone()
        self.qnet = QNetwork(self.cfg, device="meta")
        self.qnet_params.to_module(self.qnet)

        #### Low-level Policy ####
        self.pi = Actor(self.cfg, device=self.device)
        self.pi_detach = Actor(self.cfg, device=self.device)
        from_module(self.pi).data.to_module(self.pi_detach)

        #### World Model and BC (diffusion) Policy ####
        # WM: (s_t, a_{t:t+K-1}) -> s_{t+K}
        # BC: s_t -> a_{t:t+K-1} (K = wm_timescale primitive actions)
        self.wm = DynamicsModel(self.cfg, self.device)
        self.wm_detach = DynamicsModel(self.cfg, self.device)
        from_module(self.wm).data.to_module(self.wm_detach)

        self.bc = MLPDiffusionPolicy(self.cfg, self.device)
        self.bc_detach = MLPDiffusionPolicy(self.cfg, self.device)
        from_module(self.bc).data.to_module(self.bc_detach)

        #### Pretrained Weight Initialization ####
        if self.cfg.train_from_checkpoint:
            self.load(self.cfg.checkpoint)
        if self.pretrained_vf:
            print(f"Loading pretrained value function from {self.cfg.vf_chkpt}")
            vf_state_dict = torch.load(self.cfg.vf_chkpt)
            self.vf.load_state_dict(vf_state_dict["vf"])
            self.qnet_params.load_state_dict(vf_state_dict["qfs"])
            self.qnet_target_params = self.qnet_params.data.clone()
        if self.pretrained_policy:
            print(f"Loading pretrained policy from {self.cfg.policy_chkpt}")
            policy_state_dict = torch.load(self.cfg.policy_chkpt)
            self.pi.load_state_dict(policy_state_dict["pi"])

        #### Optimizers ####
        self.wm_optimizer = optim.Adam(
            list(self.wm.parameters()), lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile
        )
        self.bc_optimizer = optim.Adam(
            list(self.bc.parameters()), lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile
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
            self._update_wm = torch.compile(self._update_wm, mode=mode)
            if not self.pretrained_vf:
                self._update_q = torch.compile(self._update_q, mode=mode)
                self._update_v = torch.compile(self._update_v, mode=mode)
            if not self.pretrained_policy:
                self._update_pi = torch.compile(self._update_pi, mode=mode)
            self._act = torch.compile(self._act, mode=mode)

        if self.cfg.cudagraphs:
            self._update_wm = CudaGraphModule(self._update_wm, in_keys=[], out_keys=[], warmup=5)
            if not self.pretrained_vf:
                self._update_q = CudaGraphModule(self._update_q, in_keys=[], out_keys=[], warmup=5)
                self._update_v = CudaGraphModule(self._update_v, in_keys=[], out_keys=[], warmup=5)
            if not self.pretrained_policy:
                self._update_pi = CudaGraphModule(self._update_pi, in_keys=[], out_keys=[], warmup=5)
            self._act = CudaGraphModule(self._act, in_keys=[], out_keys=[], warmup=5)

    @torch.no_grad()
    def act(self, td):
        td = td.to(self.device, non_blocking=True)
        step = td["step"]
        goal = td["goal"]
        obs = td["obs"]
        obs_dim = obs.shape[-1]

        if step == 0:
            if self.debug_subgoals:
                print("\n\nNew Episode")
            self._subgoal = goal.clone()  # [1, obs_dim]
            self._subgoal_samples = torch.zeros(self.n_sg_samples, obs_dim, device=obs.device, dtype=obs.dtype)
            self._best_wm_traj = torch.zeros(self.planning_steps + 1, obs_dim, device=obs.device, dtype=obs.dtype)
            self._last_replan_step = 0

        if self.disable_subgoal or (step >= (self.cfg.env_episode_length - 100)) or (self.vf_detach(obs, goal).abs() < self.goal_vicinity_threshold):
            self._subgoal = goal.clone()
            self._last_replan_step = step
        else:
            # Replan at the start of the episode, after subgoal_steps steps (timeout), or when the subgoal is reached
            should_replan = False
            if step == 0:
                should_replan, reason = True, "start"
            elif (step - self._last_replan_step) >= self.subgoal_steps:
                should_replan, reason = True, "timeout"
            elif self.vf_detach(obs, self._subgoal).abs() < self.subgoal_reached_threshold:
                should_replan, reason = True, "reached"
            if should_replan:
                if self.debug_subgoals:
                    print(f"Step {step} — replanning reason={reason}")
                self._subgoal = self._plan_with_wm(obs, goal)
                self._last_replan_step = step

        td["subgoal"] = self._subgoal
        td_out = self._act(td)
        action = td_out["action"]

        info = {"subgoal": self._subgoal.detach().clone()}
        if hasattr(self, "_best_wm_traj"):
            info["wm_traj"] = self._best_wm_traj.detach().clone()
        if hasattr(self, "_subgoal_samples"):
            info["subgoal_samples"] = self._subgoal_samples.detach().clone()

        return action, info

    @torch.no_grad()
    def _plan_with_wm(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Plan by rolling out action-chunk trajectories from the current obs.

        For each of n_sg_samples trajectories, over H = planning_steps steps:
          - sample an action chunk from the BC policy at each planning step
            (or once if action_chunk_repeat is True)
          - propagate with the dynamics model: s' = D(s, a_chunk)
          - score the states with the value function against the goal

        Args:
            obs: current state  [1, obs_dim]
            goal: planning goal [1, obs_dim]

        Returns:
            chosen_subgoal: [1, obs_dim]
        """
        horizon = self.planning_steps
        n_traj = self.n_sg_samples
        obs_dim = obs.shape[-1]

        bc_plan = self.bc_detach
        wm_plan = self.wm_detach

        # Initialise all trajectories from current obs
        states = obs.clone().repeat(n_traj, 1)  # (n_traj, obs_dim)
        traj_tensor = torch.empty(horizon + 1, n_traj, obs_dim, device=states.device, dtype=states.dtype)
        traj_tensor[0] = states

        if self.action_chunk_repeat:
            action_chunk = bc_plan.sample(states).flatten(-2)  # (n_traj, K * action_dim)

        for t in range(1, horizon + 1):
            if not self.action_chunk_repeat:
                action_chunk = bc_plan.sample(states).flatten(-2)  # (n_traj, K * action_dim)
            states = wm_plan(states, action_chunk)  # (n_traj, obs_dim)
            traj_tensor[t] = states

        # Score future states (exclude t=0) against the goal
        future_traj = traj_tensor[1:] # (H, n_traj, obs_dim)
        flat_states = future_traj.view(horizon * n_traj, obs_dim)
        flat_goals = goal.view(1, obs_dim).repeat(horizon * n_traj, 1)
        flat_values = self.vf_detach(flat_states, flat_goals).view(horizon, n_traj)

        if self.planning_score == "best":
            traj_scores, _ = flat_values.max(dim=0) # (n_traj,)
        elif self.planning_score == "last":
            traj_scores = flat_values[-1] # (n_traj,)
        elif self.planning_score == "all":
            traj_scores = flat_values.sum(dim=0) # (n_traj,)
        else:
            raise ValueError(f"Unknown planning_score '{self.planning_score}' (expected 'best', 'last', or 'all').")

        best_traj_idx = traj_scores.argmax()

        # Store diagnostics
        self._subgoal_samples = traj_tensor[-1, :, :] # all final states
        best_traj_states = traj_tensor[:, best_traj_idx, :] # (H+1, obs_dim)
        self._best_wm_traj = best_traj_states

        # Subgoal: final or first state from the best trajectory
        if self.subgoal_state == "final":
            chosen_subgoal = best_traj_states[-1:, :]  # [1, obs_dim]
        elif self.subgoal_state == "first":
            chosen_subgoal = best_traj_states[1:2, :]  # [1, obs_dim]
        else:
            raise ValueError(f"Unknown subgoal_state: '{self.subgoal_state}' (expected 'final' or 'first')")

        if self.debug_subgoals:
            best_score = traj_scores[best_traj_idx]
            print(f"subgoal->goal value: {best_score.int().item()}")
            obs_to_subgoal_values = self.vf_detach(
                obs.view(1, obs_dim).repeat(n_traj, 1), self._subgoal_samples
            )
            print(f"obs->subgoal value: {obs_to_subgoal_values[best_traj_idx].int().item()}")
            print(f"obs->subgoal sample values range: [{obs_to_subgoal_values.min().int().item()}, {obs_to_subgoal_values.max().int().item()}]")

        return chosen_subgoal

    @torch.no_grad()
    def _act(self, td):
        obs = td["obs"]
        subgoal = td["subgoal"]

        action = self.pi_detach(obs, subgoal)

        if self.policy_type == "noisy":
            noise = torch.randn_like(action)
            clipped_noise = noise.mul(self.pi_noise).clamp(-self.pi_noise_clip, self.pi_noise_clip)
            action = (action + clipped_noise).clamp(self.cfg.action_low, self.cfg.action_high)
        elif self.policy_type == "deterministic":
            pass
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")

        return TensorDict(action=action.detach(), subgoal=subgoal.detach())

    def update(self, batch, step):
        info = self._update_wm(batch)
        info.update(self._update_bc(batch))

        if not self.pretrained_vf:
            info.update(self._update_v(batch))
            info.update(self._update_q(batch))
            # update the target networks
            # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
            self.qnet_target_params.lerp_(self.qnet_params.data, self.tau)

        if not self.pretrained_policy:
            info.update(self._update_pi(batch))

        return info.detach().mean()

    def _update_wm(self, batch):
        """
        MSE between D(s_t, a_{t:t+K-1}) and s_{t+K}.
        Buffer provides:
            wm_action_chunks: (B, wm_timescale, action_dim)
            wm_next_obs: (B, obs_dim)
        """
        pred_next_obs = self.wm(batch["obs"], batch["wm_action_chunks"].flatten(-2))
        wm_loss = F.mse_loss(pred_next_obs, batch["wm_next_obs"])

        self.wm_optimizer.zero_grad()
        wm_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.grad_clip_norm)
        self.wm_optimizer.step()

        return TensorDict(wm_loss=wm_loss.detach(), wm_gradnorm=grad_norm)

    def _update_bc(self, batch):
        """
        Diffusion loss of the BC policy: bc(obs) -> action_chunk (target from the replay buffer)
        Buffer provides:
            wm_action_chunks: (B, wm_timescale, action_dim)
        """
        bc_loss, _ = self.bc.loss(batch["obs"], batch["wm_action_chunks"])

        self.bc_optimizer.zero_grad()
        bc_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.bc.parameters(), self.grad_clip_norm)
        self.bc_optimizer.step()

        return TensorDict(bc_loss=bc_loss.detach(), bc_gradnorm=grad_norm)

    def _update_q(self, batch):
        obs = batch["obs"]
        action = batch["action"]
        next_obs = batch["next_obs"]
        reward = batch["reward"]
        done = batch["done"]
        goal = batch["value_goal"]

        vf_next_target = self.vf_detach(next_obs, goal)
        next_q_value = reward.flatten() + (1 - done.flatten()) * self.cfg.gamma * vf_next_target.flatten()
        next_q_value = torch.clamp(next_q_value, min=-1 / (1 - self.cfg.gamma), max=0)

        qf_loss = torch.vmap(self._batched_qf, (0, None, None, None, None))(
            self.qnet_params, obs, action, goal, next_q_value
        ).sum(0)

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.qnet_params.values(include_nested=True, leaves_only=True), self.grad_clip_norm
        )
        self.q_optimizer.step()

        return TensorDict(qf_loss=qf_loss.detach(), qf_gradnorm=grad_norm)

    def _batched_qf(self, params, obs, action, goal, next_q_value=None):
        with params.to_module(self.qnet):
            vals = self.qnet(obs, action, goal)
            if next_q_value is not None:
                return F.mse_loss(vals.view(-1), next_q_value)
            return vals

    def _update_v(self, batch):
        obs = batch["obs"]
        action = batch["action"]
        goal = batch["value_goal"]

        qf_target = torch.vmap(self._batched_qf, (0, None, None, None))(self.qnet_target_params, obs, action, goal)
        min_qf_target = qf_target.min(0).values

        value = self.vf(obs, goal)
        value_loss = torch.mean(
            torch.abs(self.expectile - ((min_qf_target - value) < 0).float()) * (min_qf_target - value) ** 2
        )

        self.v_optimizer.zero_grad()
        value_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.vf.parameters(), self.grad_clip_norm)
        self.v_optimizer.step()

        return TensorDict(vf_loss=value_loss.detach(), vf_gradnorm=grad_norm)

    def _update_pi(self, batch):
        obs = batch["obs"]
        action = batch["action"]
        goal = batch["policy_subgoal"] if self.pi_subgoal else batch["policy_goal"]

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
        torch.save({
            "wm": self.wm.state_dict(),
            "bc": self.bc.state_dict(),
            "wm_timescale": self.wm_timescale,
            "vf": self.vf.state_dict(),
            "qfs": self.qnet_params.state_dict(),
            "pi": self.pi.state_dict(),
            "obs_mean": obs_mean,
            "obs_std": obs_std,
        }, fp)

    def load(self, fp):
        state_dict = torch.load(fp)
        self.wm.load_state_dict(state_dict["wm"])
        self.bc.load_state_dict(state_dict["bc"])
        if not self.pretrained_vf:
            self.vf.load_state_dict(state_dict["vf"])
            self.qnet_params.load_state_dict(state_dict["qfs"])
            self.qnet_target_params = self.qnet_params.data.clone()
            self.qnet_params.to_module(self.qnet)
        if not self.pretrained_policy:
            self.pi.load_state_dict(state_dict["pi"])
        self.obs_mean = state_dict["obs_mean"]
        self.obs_std  = state_dict["obs_std"]

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
