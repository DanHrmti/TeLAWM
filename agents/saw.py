"""Subgoal-advantage-weighted policy bootstrapping (SAW) agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule

from agents.modules.actor_critic import QNetwork, VNetwork, Actor


class SAW(nn.Module):
    """
    Flat goal-conditioned policy pi(a | s, g) bootstrapped from a waypoint policy; no subgoals at test time.

      - A waypoint policy pi_w(a | s, w) is trained with AWR toward dataset subgoals w, k = subgoal_k steps ahead.
      - The flat policy is extracted from IQL (ddpgbc or AWR) plus a term pulling it toward pi_w(a | s, w), weighted
        by exp(beta * (V(w, g) - V(s, g))).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda:0')

        #### Hyperparameters ####
        # general
        self.grad_clip_norm = self.cfg.grad_clip_norm
        self.pretrained_vf = self.cfg.saw['pretrained_vf']
        self.alpha = self.cfg.saw['alpha']
        self.beta = self.cfg.saw['beta']
        # iql
        self.tau = self.cfg.saw['tau']
        self.expectile = self.cfg.saw['expectile']
        self.pi_extract_type = self.cfg.saw['pi_extract_type']
        self.policy_type = self.cfg.saw['policy_type']
        self.pi_noise = self.cfg.saw['pi_noise']
        self.pi_noise_clip = self.cfg.saw['pi_noise_clip']

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
        self.pi = Actor(self.cfg, device=self.device)
        self.pi_detach = Actor(self.cfg, device=self.device)
        from_module(self.pi).data.to_module(self.pi_detach)  # Copy params to pi_detach without grad

        #### High-level Policy ####
        self.pi_waypoint = Actor(self.cfg, device=self.device)
        self.pi_waypoint_detach = Actor(self.cfg, device=self.device)
        from_module(self.pi_waypoint).data.to_module(self.pi_waypoint_detach)  # Copy params to pi_waypoint_detach without grad

        #### Pretrained Weight Initialization ####
        if self.cfg.train_from_checkpoint:
            self.load(self.cfg.checkpoint)
        if self.pretrained_vf:
            print(f"Loading pretrained value function from {self.cfg.vf_chkpt}")
            vf_state_dict = torch.load(self.cfg.vf_chkpt)
            self.vf.load_state_dict(vf_state_dict["vf"])
            self.qnet_params.load_state_dict(vf_state_dict["qfs"])
            self.qnet_target_params = self.qnet_params.data.clone()
            
        #### Optimizers ####
        self.waypoint_actor_optimizer = optim.Adam(
            list(self.pi_waypoint.parameters()), lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile
        )

        self.actor_optimizer = optim.Adam(
            list(self.pi.parameters()),
            lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile
        )

        if not self.pretrained_vf:
            self.v_optimizer = optim.Adam(
                list(self.vf.parameters()),
                lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile
            )

            self.q_optimizer = optim.Adam(
                self.qnet_params.values(include_nested=True, leaves_only=True),
                lr=self.cfg.lr,
                capturable=self.cfg.cudagraphs and not self.cfg.compile,
            )

        # torch.compile
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            self._update_pi_waypoint = torch.compile(self._update_pi_waypoint, mode=mode)
            self._update_pi = torch.compile(self._update_pi, mode=mode)
            if not self.pretrained_vf:
                self._update_q = torch.compile(self._update_q, mode=mode)
                self._update_v = torch.compile(self._update_v, mode=mode)
            self._act = torch.compile(self._act, mode=mode)

        if self.cfg.cudagraphs:
            self._update_pi_waypoint = CudaGraphModule(self._update_pi_waypoint, in_keys=[], out_keys=[], warmup=5)
            self._update_pi = CudaGraphModule(self._update_pi, in_keys=[], out_keys=[], warmup=5)
            if not self.pretrained_vf:
                self._update_q = CudaGraphModule(self._update_q, in_keys=[], out_keys=[], warmup=5)
                self._update_v = CudaGraphModule(self._update_v, in_keys=[], out_keys=[], warmup=5)
            self._act = CudaGraphModule(self._act, in_keys=[], out_keys=[], warmup=5)

    @torch.no_grad()
    def act(self, td):
        td_out = self._act(td.to(self.device, non_blocking=True))
        return td_out["action"], {}

    @torch.no_grad()
    def _act(self, td):
        obs = td["obs"]
        goal = td["goal"]

        action = self.pi_detach(obs, goal)

        if self.policy_type == "noisy":
            noise = torch.randn_like(action)
            clipped_noise = noise.mul(self.pi_noise).clamp(-self.pi_noise_clip, self.pi_noise_clip)  # NOTE: action scale is assumed to be 1
            action = (action + clipped_noise).clamp(self.cfg.action_low, self.cfg.action_high)
        elif self.policy_type == "deterministic":
            pass
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")

        return TensorDict(action=action.detach())

    def update(self, batch, step):
        # update actors
        info = self._update_pi_waypoint(batch)

        info.update(self._update_pi(batch))

        if not self.pretrained_vf:
            # update value
            info.update(self._update_v(batch))

            # update q-value
            info.update(self._update_q(batch))

            # update the target networks
            # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
            self.qnet_target_params.lerp_(self.qnet_params.data, self.tau)
            for p, tp in zip(self.vf.parameters(), self.vf_detach.parameters()):
                tp.data.lerp_(p.data, self.tau)

        return info.detach().mean()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = torch.where(adv >= 0, expectile, (1.0 - expectile))
        return weight * (diff ** 2)

    def _update_q(self, batch):
        obs = batch["obs"]
        action = batch["action"]
        next_obs = batch["next_obs"]
        reward = batch["reward"]
        done = batch["done"]
        goal = batch["value_goal"]

        vf_next_target = self.vf_detach(next_obs, goal)
        next_q_value = reward.flatten() + (1 - done.flatten()) * self.cfg.gamma * vf_next_target.flatten()  # whether a "done" signal is received depends on config
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

        value_loss = self.expectile_loss(min_qf_target - value, min_qf_target - value, self.expectile).mean()

        self.v_optimizer.zero_grad()
        value_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(list(self.vf.parameters()), self.grad_clip_norm)
        self.v_optimizer.step()

        return TensorDict(vf_loss=value_loss.detach(), vf_gradnorm=grad_norm, v_mean=value.mean().detach(), v_max=value.max().detach(), v_min=value.min().detach())

    def _update_pi_waypoint(self, batch):
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        action = batch["action"]
        goal = batch["policy_subgoal"]

        v_current = self.vf_detach(obs, goal)
        v_next = self.vf_detach(next_obs, goal)
        adv = v_next - v_current

        pi_out = self.pi_waypoint(obs, goal)
        exp_adv = torch.exp(self.alpha * adv.detach()).clamp(max=100.0)
        bc_losses = F.mse_loss(pi_out, action, reduction='none').sum(dim=-1, keepdim=True)  # [B, 1]
        pi_loss = torch.mean(exp_adv * bc_losses)

        self.waypoint_actor_optimizer.zero_grad()
        pi_loss.backward()
        self.waypoint_actor_optimizer.step()

        return TensorDict(pi_waypoint_loss=pi_loss.detach(), pi_waypoint_adv=adv.mean().detach(), pi_waypoint_bc_loss=bc_losses.mean().detach())

    def _update_pi(self, batch):
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        action = batch["action"]
        goal = batch["policy_goal"]
        subgoal = batch["policy_subgoal"]

        pi_out = self.pi(obs, goal)
        bc_losses = F.mse_loss(pi_out, action, reduction='none').sum(dim=-1, keepdim=True)  # [B, 1]

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

        # waypoint loss
        with torch.no_grad():
            v = self.vf(obs, goal)
            wv = self.vf(subgoal, goal)
            wadv = wv - v
            exp_w = torch.exp(wadv * self.beta).clamp(max=100.0)

            # waypoint actor target (stop grad)
            pi_waypoint_out = self.pi_waypoint(obs, subgoal).detach()

        # waypoint KL loss
        kld = F.mse_loss(pi_out, pi_waypoint_out, reduction='none').sum(dim=-1, keepdim=True)
        waypoint_loss = (exp_w * kld).mean()

        loss = pi_loss + waypoint_loss

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return_dict = TensorDict(
            pi_loss=pi_loss.detach(),
            waypoint_loss=waypoint_loss.detach(),
            wadv=wadv.mean().detach(),
            kld=kld.mean().detach(),
        )
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
            "pi": self.pi.state_dict(),
            "pi_waypoint": self.pi_waypoint.state_dict(),
            "vf": self.vf.state_dict(),
            "qfs": self.qnet_params.state_dict(),
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
        self.pi.load_state_dict(state_dict["pi"])
        self.pi_waypoint.load_state_dict(state_dict["pi_waypoint"])
        self.vf.load_state_dict(state_dict["vf"])
        from_module(self.vf).data.to_module(self.vf_detach)
        self.qnet_params.load_state_dict(state_dict["qfs"])
        self.qnet_target_params = self.qnet_params.data.clone()
        self.qnet_params.to_module(self.qnet)
        self.obs_mean = state_dict["obs_mean"]
        self.obs_std = state_dict["obs_std"]

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    