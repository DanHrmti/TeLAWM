"""TeLAWM agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule

from agents.modules.actor_critic import QNetwork, VNetwork, Actor
from agents.modules.lam import LatentActionModel


class TeLAWM(nn.Module):
    """
    Plans subgoals in the latent action space of a latent action model (LAM).

      - LAM: latent action z_t inferred from (s_t, s_{t+k}), forward dynamics D(s_{t+k} | s_t, z_t), k = lam_timescale.
      - Planning: sample z from the LAM prior, roll out with D for planning_steps steps, and take the subgoal from the
        trajectory with the highest value toward the goal; it is handed to an IQL low-level policy pi(a | s, subgoal).
      - obs=latent trains only the LAM; obs=rgb decodes its latent subgoals for a pixel low level.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda:0')

        #### Hyperparameters ####
        # general
        self.grad_clip_norm = self.cfg.grad_clip_norm
        self.pretrained_vf = self.cfg.telawm["pretrained_vf"]
        self.pretrained_policy = self.cfg.telawm["pretrained_policy"]
        # subgoals
        self.n_sg_samples = self.cfg.telawm["n_sg_samples"]
        self.planning_steps = self.cfg.telawm["planning_steps"]
        self.planning_score = self.cfg.telawm["planning_score"]
        self.subgoal_state = self.cfg.telawm["subgoal_state"]
        self.latent_action_repeat = self.cfg.telawm["latent_action_repeat"]  # if true, sample the latent action once from the LAM prior at t=0 and reuse it for the entire rollout horizon
        self.subgoal_reached_threshold = self.cfg.telawm["subgoal_reached_threshold"]
        self.goal_vicinity_threshold = self.cfg.telawm["goal_vicinity_threshold"]
        self.lam_timescale = self.cfg.telawm["lam_timescale"]  # step skip k of the LAM: (s_t, s_{t+k})
        self.subgoal_steps = self.cfg.telawm["subgoal_steps"]  # test-time replan interval
        self.filter_subgoals = self.cfg.telawm["filter_subgoals"]
        self.debug_subgoals = self.cfg.telawm["debug_subgoals"]
        self.disable_subgoal = self.cfg.telawm["disable_subgoal"]
        # visual tasks: obs=latent trains only the LAM; obs=rgb plans with it over a pretrained pixel low level
        self.train_vf = not self.pretrained_vf and self.cfg.obs != 'latent'
        self.train_pi = not self.pretrained_policy and self.cfg.obs != 'latent'
        # iql
        self.tau = self.cfg.telawm['tau']
        self.expectile = self.cfg.telawm['expectile']
        self.alpha = self.cfg.telawm['alpha']
        self.pi_extract_type = self.cfg.telawm["pi_extract_type"]
        self.pi_subgoal = self.cfg.telawm["pi_subgoal"]
        self.policy_type = self.cfg.telawm['policy_type']
        self.pi_noise = self.cfg.telawm['pi_noise']
        self.pi_noise_clip = self.cfg.telawm['pi_noise_clip']

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

        #### Low-level Policy ####
        self.pi = Actor(self.cfg, self.device)
        self.pi_detach = Actor(self.cfg, self.device)
        from_module(self.pi).data.to_module(self.pi_detach)  # Copy params to pi_detach without grad

        #### Frozen autoencoder (obs=rgb): plan in the LAM's latent, decode subgoals for the pixel low level ####
        self.ae = None
        lam_cfg = self.cfg  # the config the LAM is built from (latent-dimensional for obs=rgb)
        if self.cfg.obs == "rgb":
            assert self.cfg.get("latent_encoder_checkpoint"), "telawm with obs=rgb needs latent_encoder_checkpoint (the LAM's autoencoder)"
            assert self.pretrained_vf and self.pretrained_policy, (
                "telawm with obs=rgb runs the pixel IQL of vf_chkpt / policy_chkpt: set telawm.pretrained_vf=true and telawm.pretrained_policy=true"
            )
            from agents.modules.autoencoder import FrozenAE
            self.ae = FrozenAE(self.cfg, device=self.device)
            lam_cfg = self.ae.latent_cfg
            print(f"TeLAWM plans in the latent of {self.cfg.latent_encoder_checkpoint} (z_dim={self.ae.z_dim}); subgoals are decoded to images")

        #### Latent Action Model ####
        # learns (s_t, s_{t+k}) -> z and (s, z) -> s' at timescale k = lam_timescale
        self.lam = LatentActionModel(lam_cfg, self.device)
        self.lam_detach = LatentActionModel(lam_cfg, self.device)
        from_module(self.lam).data.to_module(self.lam_detach)

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
            if self.ae is not None:
                # frames are normalized with the pixel low level's statistics
                self.obs_mean, self.obs_std = policy_state_dict["obs_mean"], policy_state_dict["obs_std"]
        
        #### Optimizers ####
        self.lam_optimizer = optim.Adam(
            list(self.lam.parameters()), lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile
        )

        if self.train_vf:
            self.v_optimizer = optim.Adam(
                list(self.vf.parameters()), lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile
            )

            self.q_optimizer = optim.Adam(
                self.qnet_params.values(include_nested=True, leaves_only=True),
                lr=self.cfg.lr,
                capturable=self.cfg.cudagraphs and not self.cfg.compile,
            )

        if self.train_pi:
            self.actor_optimizer = optim.Adam(
                list(self.pi.parameters()), lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile
            )

        #### torch.compile ####
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            self._update_lam = torch.compile(self._update_lam, mode=mode)
            if self.train_vf:
                self._update_q = torch.compile(self._update_q, mode=mode)
                self._update_v = torch.compile(self._update_v, mode=mode)
            if self.train_pi:
                self._update_pi = torch.compile(self._update_pi, mode=mode)
            self._act = torch.compile(self._act, mode=mode)
            
        if self.cfg.cudagraphs:
            self._update_lam = CudaGraphModule(self._update_lam, in_keys=[], out_keys=[], warmup=5)
            if self.train_vf:
                self._update_q = CudaGraphModule(self._update_q, in_keys=[], out_keys=[], warmup=5)
                self._update_v = CudaGraphModule(self._update_v, in_keys=[], out_keys=[], warmup=5)
            if self.train_pi:
                self._update_pi = CudaGraphModule(self._update_pi, in_keys=[], out_keys=[], warmup=5)
            self._act = CudaGraphModule(self._act, in_keys=[], out_keys=[], warmup=5)

    @property
    def eval_policy(self):
        # obs=latent runs train only the LAM and have no low level to roll out
        return self.cfg.obs != 'latent'

    @torch.no_grad()
    def act(self, td):
        td = td.to(self.device, non_blocking=True)
        step = td["step"]
        goal = td["goal"]
        obs = td["obs"]
        plan_dim = self.lam.obs_dim  # the LAM's state space (the autoencoder latent for an rgb low level)

        if step == 0:
            if self.debug_subgoals:
                print("\n\nNew Episode")
            # current subgoal [1, *obs_shape], in the low level's observation space
            self._subgoal = goal.clone()
            self._subgoal_samples = torch.zeros(self.n_sg_samples, plan_dim, device=obs.device, dtype=torch.float32)
            self._best_lam_traj = torch.zeros(self.planning_steps + 1, plan_dim, device=obs.device, dtype=torch.float32)
            self._last_replan_step = 0

        if self.disable_subgoal or (step >= (self.cfg.env_episode_length - 100)) or (self.vf_detach(obs, goal).abs() < self.goal_vicinity_threshold):
            self._subgoal = goal.clone()
            self._last_replan_step = step

        else:
            # Replan at the start of the episode, after subgoal_steps steps (timeout), or when the current subgoal is
            # reached according to the value function.
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
                self._subgoal = self._plan_with_lam(obs, self._encode(obs), goal)
                self._last_replan_step = step

        td["subgoal"] = self._subgoal
        td_out = self._act(td)
        action = td_out["action"]

        info = {"subgoal": self._subgoal.detach().clone()}
        if self.ae is not None:
            info["subgoal_image"] = self._subgoal.detach().clone()  # exactly the image handed to the low level
        if hasattr(self, "_best_lam_traj"):
            info["lam_traj"] = self._best_lam_traj.detach().clone()
        if hasattr(self, "_subgoal_samples"):
            info["subgoal_samples"] = self._subgoal_samples.detach().clone()

        return action, info

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Low-level observation -> the LAM's (normalized latent) space; identity unless obs=rgb."""
        return x if self.ae is None else self.ae.encode(x)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """LAM's space -> low-level observation (decoded 0..255 frames); identity unless obs=rgb."""
        return z if self.ae is None else self.ae.decode(z)

    @torch.no_grad()
    def _plan_with_lam(self, obs: torch.Tensor, obs_plan: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Plan by rolling out latent-action trajectories from the current obs using the LAM.

        For each of n_sg_samples trajectories:
          - sequentially apply latent actions sampled from the LAM prior
          - propagate with LAM forward dynamics
          - score the trajectory based on planning score config

        The best-scoring trajectory is stored in self._best_lam_traj and self._subgoal_samples.

        With an rgb low level the rollout is in the autoencoder latent, and every planned state is decoded to an image
        before it is scored by the (pixel) value function; the chosen subgoal is returned as the decoded image.

        Args:
            obs: current observation in the low level's space [1, *obs_shape]
            obs_plan: current state in the LAM's space [1, plan_dim] (obs itself unless obs=rgb)
            goal: goal to plan toward, in the low level's space [1, *obs_shape].

        Returns:
            chosen_subgoal: [1, *obs_shape] — the selected subgoal, in the low level's space.
        """
        horizon = self.planning_steps
        n_traj = self.n_sg_samples

        # Initial states for all trajectories
        states = obs_plan.clone().repeat(n_traj, 1)  # [n_traj, obs_dim]

        # Preallocate trajectory tensor [T+1, n_traj, obs_dim]
        obs_dim = states.shape[-1]
        traj_tensor = torch.empty(horizon + 1, n_traj, obs_dim, device=states.device, dtype=states.dtype)
        
        traj_tensor[0] = states

        lam_plan = self.lam_detach

        if self.latent_action_repeat:
            # Sample once from the prior given the initial states, then reuse.
            out = lam_plan.sample_policy(states)
            latent = out["z"]  # [n_traj, latent_dim]

        for t in range(1, horizon + 1):
            if not self.latent_action_repeat:
                out = lam_plan.sample_policy(states)
                latent = out["z"]  # [n_traj, latent_dim]
            states = lam_plan.forward_dynamics(states, latent)  # [n_traj, obs_dim]
            traj_tensor[t] = states

        # traj_tensor shape: [T+1, n_traj, obs_dim]
        _, n_traj, obs_dim = traj_tensor.shape

        # Evaluate values only for FUTURE states (exclude t=0 current state)
        future_traj = traj_tensor[1:]  # [T, n_traj, obs_dim]
        flat_states = self._decode(future_traj.reshape(horizon * n_traj, obs_dim))  # low-level space (images for rgb)
        flat_goals = goal.expand(horizon * n_traj, *goal.shape[1:])
        flat_values = self.vf_detach(flat_states, flat_goals).view(horizon, n_traj)

        # Score each trajectory according to config:
        #  - "best": max value over future time steps
        #  - "last": value at last future time step
        #  - "all":  sum of values over planned steps
        if self.planning_score == "best":
            traj_scores, _ = flat_values.max(dim=0)      # [n_traj]
        elif self.planning_score == "last":
            traj_scores = flat_values[-1]                # [n_traj]
        elif self.planning_score == "all":
            traj_scores = flat_values.sum(dim=0)         # [n_traj]
        else:
            raise ValueError(f"Unknown planning_score '{self.planning_score}' (expected 'best', 'last', or 'all').")

        best_traj_idx = traj_scores.argmax()             # scalar

        # All last states across trajectories (subgoal candidates)
        self._subgoal_samples = traj_tensor[-1, :, :]  # [n_traj, obs_dim]
        # Best trajectory of states [T+1, obs_dim]
        best_traj_states = traj_tensor[:, best_traj_idx, :]
        self._best_lam_traj = best_traj_states

        # Subgoal: final or first state from the best trajectory (the decoded state already scored above)
        if self.subgoal_state == "final":
            chosen_t = horizon - 1
        elif self.subgoal_state == "first":
            chosen_t = 0
        else:
            raise ValueError(f"Unknown subgoal_state: '{self.subgoal_state}' (expected 'final' or 'first')")
        chosen_subgoal = flat_states.view(horizon, n_traj, *flat_states.shape[1:])[chosen_t, best_traj_idx].unsqueeze(0)  # [1, *obs_shape]

        if self.debug_subgoals:
            print(f"subgoal->goal value: {traj_scores[best_traj_idx].int().item()}")
            obs_to_subgoal_values = self.vf_detach(obs.expand(n_traj, *obs.shape[1:]), self._decode(self._subgoal_samples))
            print(f"obs->subgoal value:  {obs_to_subgoal_values[best_traj_idx].int().item()}")
            print(f"obs->subgoal sample values range:  [{obs_to_subgoal_values.min().int().item()}, {obs_to_subgoal_values.max().int().item()}]")
        return chosen_subgoal

    @torch.no_grad()
    def _act(self, td):
        obs = td["obs"]
        subgoal = td["subgoal"]

        action = self.pi_detach(obs, subgoal)

        if self.policy_type == "noisy":
            noise = torch.randn_like(action)
            clipped_noise = noise.mul(self.pi_noise).clamp(-self.pi_noise_clip, self.pi_noise_clip)  # NOTE: action scale is assumed to be 1
            action = (action + clipped_noise).clamp(self.cfg.action_low, self.cfg.action_high)
        elif self.policy_type == "deterministic":
            pass
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")
        
        return TensorDict(action=action.detach(), subgoal=subgoal.detach())
    
    def update(self, batch, step):
        # update lam
        info = self._update_lam(batch)

        if self.train_vf:
            # update value
            info.update(self._update_v(batch))
            # update q-value
            info.update(self._update_q(batch))
            # update the target networks
            # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
            self.qnet_target_params.lerp_(self.qnet_params.data, self.tau)
        
        if self.train_pi:
            # update actor
            info.update(self._update_pi(batch))
        
        return info.detach().mean()
    
    def _update_lam(self, batch):
        # Buffer provides lam_next_obs = s_{t+k} (B, obs_dim) for the start state obs = s_t (B, obs_dim)
        losses = self.lam.losses(batch["obs"], batch["lam_next_obs"])

        self.lam_optimizer.zero_grad()
        losses["lam_loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.lam.parameters(), self.grad_clip_norm)
        self.lam_optimizer.step()

        td = TensorDict(lam_gradnorm=grad_norm)
        for k in ("lam_loss", "lam_recon", "lam_kl", "lam_mu_mean", "lam_var_mean", "lam_p_mu_mean", "lam_p_var_mean"):
            if losses.get(k) is not None:
                td[k] = losses[k].detach()
        return td
    
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
            "lam": self.lam.state_dict(),
            "lam_timescale": self.lam_timescale,
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
        self.lam.load_state_dict(state_dict["lam"])
        if not self.pretrained_vf:
            self.vf.load_state_dict(state_dict["vf"])
            self.qnet_params.load_state_dict(state_dict["qfs"])
            self.qnet_target_params = self.qnet_params.data.clone()
            self.qnet_params.to_module(self.qnet)
        if not self.pretrained_policy:
            self.pi.load_state_dict(state_dict["pi"])
        if self.ae is not None:
            # obs=rgb: the checkpoint is an obs=latent run; its normalization is the LAM's latent's
            self.ae.set_stats(state_dict["obs_mean"], state_dict["obs_std"])
        else:
            self.obs_mean = state_dict["obs_mean"]
            self.obs_std = state_dict["obs_std"]

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    