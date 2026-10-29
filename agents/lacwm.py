"""Latent Action-Chunk World Model (LACWM) agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule

from agents.modules.actor_critic import QNetwork, VNetwork, Actor
from agents.modules.lacm import LatentActionChunkModel


class LACWMDynamicsModel(nn.Module):
    """Separate planning model for lacwm.variant=chunk_vae: MLP D(s_{t+K} | s_t, z_t), z_t the encoder mean."""

    def __init__(self, cfg, device=None):
        super().__init__()
        obs_dim = cfg.obs_shape[-1]
        h_dim = cfg.arch_mlp['h_dim']

        latent_dim = cfg.lacwm['latent_dim']

        self.dynamics = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, h_dim, device=device),
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

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.dynamics(torch.cat([obs, z], dim=-1))


class LACWM(nn.Module):
    """
    Plans subgoals in the latent space of an action-chunk VAE.

      - LACM (LatentActionChunkModel): z_t summarizes the action chunk a_{t:t+K-1} (K = wm_timescale), encoded by
        q(z_t | a_{t:t+K-1}) (VAE) or q(z_t | a_{t:t+K-1}, s_t) (CVAE, state_conditioned).
      - Planning model D(s_{t+K} | s_t, z_t), depending on lacwm.variant:
          dynamics:   the LACM's own decoder
          chunk_vae:  the LACM reconstructs the action chunk; a separate dynamics model (LACWMDynamicsModel) is
                      trained on the encoder mean z_t
      - Planning: sample z from the LACM prior (N(0, I), or p(z | s_t) if learned), roll out with D for
        planning_steps steps, score the states with the value function toward the goal.
      - The chosen subgoal is handed to a low-level policy pi(a | s, subgoal); the value function and low-level
        policy are goal-conditioned IQL, frozen (pretrained) or trained jointly.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda:0')

        #### Hyperparameters ####
        self.grad_clip_norm = cfg.grad_clip_norm
        self.pretrained_vf = cfg.lacwm["pretrained_vf"]
        self.pretrained_policy = cfg.lacwm["pretrained_policy"]
        self.pretrained_lacm = cfg.lacwm["pretrained_lacm"]
        self.state_conditioned = cfg.lacwm["state_conditioned"]
        self.variant = cfg.lacwm["variant"]  # [dynamics, chunk_vae], see the class docstring
        self.separate_wm = self.variant == "chunk_vae"
        # subgoals / planning
        self.n_sg_samples = cfg.lacwm["n_sg_samples"]
        self.planning_steps = cfg.lacwm["planning_steps"]
        self.planning_score = cfg.lacwm["planning_score"]
        self.subgoal_state = cfg.lacwm["subgoal_state"]
        self.latent_action_repeat = cfg.lacwm["latent_action_repeat"]
        self.subgoal_reached_threshold = cfg.lacwm["subgoal_reached_threshold"]
        self.goal_vicinity_threshold = cfg.lacwm["goal_vicinity_threshold"]
        self.wm_timescale = cfg.lacwm["wm_timescale"]  # action chunk length K
        self.subgoal_steps = cfg.lacwm["subgoal_steps"]  # test-time replan interval
        self.filter_subgoals = cfg.lacwm["filter_subgoals"]
        self.debug_subgoals = cfg.lacwm["debug_subgoals"]
        self.disable_subgoal = cfg.lacwm["disable_subgoal"]
        # iql
        self.tau = cfg.lacwm['tau']
        self.expectile = cfg.lacwm['expectile']
        self.alpha = cfg.lacwm['alpha']
        self.pi_extract_type = cfg.lacwm["pi_extract_type"]
        self.pi_subgoal = cfg.lacwm["pi_subgoal"]
        self.policy_type = cfg.lacwm['policy_type']
        self.pi_noise = cfg.lacwm['pi_noise']
        self.pi_noise_clip = cfg.lacwm['pi_noise_clip']

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

        #### Latent Action-Chunk Model ####
        # encodes a_{t:t+K-1} -> z (VAE/CVAE)
        self.lacm = LatentActionChunkModel(self.cfg, self.device)
        self.lacm_detach = LatentActionChunkModel(self.cfg, self.device)
        from_module(self.lacm).data.to_module(self.lacm_detach)

        #### Dynamics Model (variant=chunk_vae) ####
        # D(s_{t+K} | s_t, z_t)
        if self.separate_wm:
            self.wm = LACWMDynamicsModel(self.cfg, self.device)
            self.wm_detach = LACWMDynamicsModel(self.cfg, self.device)
            from_module(self.wm).data.to_module(self.wm_detach)

        #### Pretrained Weight Initialization ####
        if self.cfg.train_from_checkpoint:
            self.load(cfg.checkpoint)
        if self.pretrained_vf:
            print(f"Loading pretrained value function from {cfg.vf_chkpt}")
            vf_sd = torch.load(cfg.vf_chkpt)
            self.vf.load_state_dict(vf_sd["vf"])
            self.qnet_params.load_state_dict(vf_sd["qfs"])
            self.qnet_target_params = self.qnet_params.data.clone()
        if self.pretrained_policy:
            print(f"Loading pretrained policy from {cfg.policy_chkpt}")
            pi_sd = torch.load(cfg.policy_chkpt)
            self.pi.load_state_dict(pi_sd["pi"])
        if self.pretrained_lacm:
            print(f"Loading pretrained LACM from {cfg.lacwm['lacm_chkpt']}")
            lacm_sd = torch.load(cfg.lacwm["lacm_chkpt"])
            self.lacm.load_state_dict(lacm_sd["lacm"])
            from_module(self.lacm).data.to_module(self.lacm_detach)

        #### Optimizers ####
        # LACM and WM are trained jointly (both conditioned on the same data)
        if not self.pretrained_lacm:
            self.lacm_optimizer = optim.Adam(
                list(self.lacm.parameters()),
                lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile,
            )
        if self.separate_wm:
            self.wm_optimizer = optim.Adam(
                list(self.wm.parameters()),
                lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile,
            )
        if not self.pretrained_vf:
            self.v_optimizer = optim.Adam(
                list(self.vf.parameters()),
                lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile,
            )
            self.q_optimizer = optim.Adam(
                self.qnet_params.values(include_nested=True, leaves_only=True),
                lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile,
            )
        if not self.pretrained_policy:
            self.actor_optimizer = optim.Adam(
                list(self.pi.parameters()),
                lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile,
            )

        #### torch.compile ####
        if self.cfg.compile:
            mode = "reduce-overhead" if not cfg.cudagraphs else None
            if not self.pretrained_lacm:
                self._update_lacm = torch.compile(self._update_lacm, mode=mode)
            if self.separate_wm:
                self._update_wm = torch.compile(self._update_wm, mode=mode)
            if not self.pretrained_vf:
                self._update_q = torch.compile(self._update_q, mode=mode)
                self._update_v = torch.compile(self._update_v, mode=mode)
            if not self.pretrained_policy:
                self._update_pi = torch.compile(self._update_pi, mode=mode)
            self._act = torch.compile(self._act, mode=mode)

        if self.cfg.cudagraphs:
            if not self.pretrained_lacm:
                self._update_lacm = CudaGraphModule(self._update_lacm, in_keys=[], out_keys=[], warmup=5)
            if self.separate_wm:
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

        if (
            self.disable_subgoal
            or (step >= (self.cfg.env_episode_length - 100))
            or (self.vf_detach(obs, goal).abs() < self.goal_vicinity_threshold)
        ):
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
                self._subgoal = self._plan_with_lacwm(obs, goal)
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
    def _plan_with_lacwm(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Roll out latent-z trajectories using the LACM prior and the planning model D.

        For each of n_sg_samples trajectories, over H = planning_steps steps:
          - sample z from the prior (N(0,I) or p(z|s)) at each planning step
            (or once if latent_action_repeat is True)
          - propagate with D: s' = D(s, z)
          - score the states with the value function against the goal

        Args:
            obs:  current state [1, obs_dim]
            goal: planning goal [1, obs_dim]

        Returns:
            chosen_subgoal: [1, obs_dim]
        """
        horizon = self.planning_steps
        n_traj = self.n_sg_samples
        obs_dim = obs.shape[-1]

        lacm_plan = self.lacm_detach
        # planning model D(s_{t+K} | s_t, z): the separate WM (chunk_vae) or the LACM decoder (dynamics)
        wm_plan = self.wm_detach if self.separate_wm else lacm_plan.forward_dynamics

        states = obs.clone().repeat(n_traj, 1) # [n_traj, obs_dim]
        traj_tensor = torch.empty(
            horizon + 1, n_traj, obs_dim, device=states.device, dtype=states.dtype
        )
        traj_tensor[0] = states

        if self.latent_action_repeat:
            # Sample z once from the prior at t=0, reuse for all planning steps
            out = lacm_plan.sample_prior(states)
            z = out["z"] # [n_traj, latent_dim]

        for t in range(1, horizon + 1):
            if not self.latent_action_repeat:
                out = lacm_plan.sample_prior(states)
                z = out["z"]
            states = wm_plan(states, z) # [n_traj, obs_dim]
            traj_tensor[t] = states

        # Score future states (exclude t=0) against the goal
        future_traj = traj_tensor[1:] # [H, n_traj, obs_dim]
        flat_states = future_traj.view(horizon * n_traj, obs_dim)
        flat_goals = goal.view(1, obs_dim).repeat(horizon * n_traj, 1)
        flat_values = self.vf_detach(flat_states, flat_goals).view(horizon, n_traj)

        if self.planning_score == "best":
            traj_scores, _ = flat_values.max(dim=0) # [n_traj]
        elif self.planning_score == "last":
            traj_scores = flat_values[-1] # [n_traj]
        elif self.planning_score == "all":
            traj_scores = flat_values.sum(dim=0) # [n_traj]
        else:
            raise ValueError(
                f"Unknown planning_score '{self.planning_score}' "
                "(expected 'best', 'last', or 'all')."
            )

        best_traj_idx = traj_scores.argmax()

        # Store diagnostics
        self._subgoal_samples = traj_tensor[-1, :, :] # [n_traj, obs_dim]
        best_traj_states = traj_tensor[:, best_traj_idx, :] # [H+1, obs_dim]
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
                obs.view(1, obs_dim).repeat(n_traj, 1),
                self._subgoal_samples,
            )
            print(f"obs->subgoal value: {obs_to_subgoal_values[best_traj_idx].int().item()}")
            print(
                f"obs->subgoal sample values range: "
                f"[{obs_to_subgoal_values.min().int().item()}, "
                f"{obs_to_subgoal_values.max().int().item()}]"
            )

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
        if not self.pretrained_lacm:
            info = self._update_lacm(batch)
        else:
            info = TensorDict()
        if self.separate_wm:
            info.update(self._update_wm(batch))

        if not self.pretrained_vf:
            info.update(self._update_v(batch))
            info.update(self._update_q(batch))
            self.qnet_target_params.lerp_(self.qnet_params.data, self.tau)

        if not self.pretrained_policy:
            info.update(self._update_pi(batch))

        return info.detach().mean()

    def _update_lacm(self, batch):
        """
        Train the LACM: reconstruction of its decoder target (s_{t+K} for variant=dynamics, the action chunk for
        chunk_vae) + beta_kl * KL( q(z | a_chunk [, s]) || prior ).

        Buffer provides:
            obs:               (B, obs_dim)
            wm_action_chunks:  (B, wm_timescale, action_dim)
            wm_next_obs:       (B, obs_dim)
        """
        losses = self.lacm.losses(batch["obs"], batch["wm_action_chunks"].flatten(-2), batch["wm_next_obs"])

        self.lacm_optimizer.zero_grad()
        losses["lacm_loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.lacm.parameters(), self.grad_clip_norm)
        self.lacm_optimizer.step()

        # Sync detach copy
        from_module(self.lacm).data.to_module(self.lacm_detach)

        td = TensorDict(lacm_gradnorm=grad_norm)
        for k in ("lacm_loss", "lacm_recon", "lacm_kl", "lacm_mu_mean", "lacm_var_mean", "lacm_p_mu_mean", "lacm_p_var_mean"):
            if losses.get(k) is not None:
                td[k] = losses[k].detach()
        return td

    def _update_wm(self, batch):
        """
        Train the separate dynamics model (variant=chunk_vae): MSE between D(s_t, z_t) and s_{t+K}.

        z_t is obtained from the encoder in deterministic mode (mean of q).
        This keeps the WM training signal stable and decoupled from the
        reparameterisation noise in the LACM update.

        Buffer provides same keys as _update_lacm.
        """
        obs = batch["obs"]

        # Deterministic encode → z (no grad into LACM from WM loss)
        with torch.no_grad():
            enc = self.lacm.encode(obs, batch["wm_action_chunks"].flatten(-2))
            z   = enc["mu"]

        wm_loss = F.mse_loss(self.wm(obs, z), batch["wm_next_obs"])

        self.wm_optimizer.zero_grad()
        wm_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.grad_clip_norm)
        self.wm_optimizer.step()

        # Sync detach copy
        from_module(self.wm).data.to_module(self.wm_detach)

        return TensorDict(wm_loss=wm_loss.detach(), wm_gradnorm=grad_norm)

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
        torch.save(
            {
                "lacm": self.lacm.state_dict(),
                **({"wm": self.wm.state_dict()} if self.separate_wm else {}),
                "variant": self.variant,
                "wm_timescale": self.wm_timescale,
                "vf": self.vf.state_dict(),
                "qfs": self.qnet_params.state_dict(),
                "pi": self.pi.state_dict(),
                "obs_mean": obs_mean,
                "obs_std": obs_std,
            },
            fp,
        )

    def load(self, fp):
        state_dict = torch.load(fp)
        assert state_dict["variant"] == self.variant, \
            f"{fp} holds a lacwm.variant={state_dict['variant']} model, but the config has {self.variant}"
        self.lacm.load_state_dict(state_dict["lacm"])
        from_module(self.lacm).data.to_module(self.lacm_detach)
        if self.separate_wm:
            self.wm.load_state_dict(state_dict["wm"])
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
