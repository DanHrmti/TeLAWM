"""Goal-conditioned implicit Q-learning (IQL) agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule

from agents.modules.actor_critic import QNetwork, VNetwork, Actor


class IQL(nn.Module):
    """
    Flat goal-conditioned policy pi(a | s, g) on top of expectile-trained value functions.

      - Q and V are trained with expectile regression (optional n_step TD targets for Q).
      - The policy is extracted with ddpgbc (Q maximization + alpha * BC) or AWR, per pi_extract_type.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda:0')

        # hyperparameters
        self.tau = self.cfg.iql['tau']
        self.expectile = self.cfg.iql['expectile']
        self.alpha = self.cfg.iql['alpha']
        self.n_step = self.cfg.iql['n_step']
        self.grad_clip_norm = self.cfg.grad_clip_norm
        self.pi_extract_type = self.cfg.iql['pi_extract_type']
        self.pi_subgoal = self.cfg.iql['pi_subgoal']
        self.pretrained_vf = self.cfg.iql["pretrained_vf"]
        self.policy_type = self.cfg.iql['policy_type']
        self.pi_noise = self.cfg.iql['pi_noise']
        self.pi_noise_clip = self.cfg.iql['pi_noise_clip']

        # actor
        self.pi = Actor(self.cfg, self.device)
        
        self.pi_detach = Actor(self.cfg, self.device)
        from_module(self.pi).data.to_module(self.pi_detach)  # Copy params to pi_detach without grad
        
        # value
        self.vf = VNetwork(self.cfg, self.device)

        self.vf_detach = VNetwork(self.cfg, self.device)
        from_module(self.vf).data.to_module(self.vf_detach)  # Copy params to vf_detach without grad

        # q-value
        qf1 = QNetwork(self.cfg, self.device)
        qf2 = QNetwork(self.cfg, self.device)

        self.qnet_params = from_modules(qf1, qf2, as_module=True)
        self.qnet_target_params = self.qnet_params.data.clone()
        # discard params of net
        self.qnet = QNetwork(self.cfg, device="meta")
        self.qnet_params.to_module(self.qnet)

        # pretrained weight initialization
        if self.cfg.train_from_checkpoint:
            self.load(self.cfg.checkpoint)
        if self.pretrained_vf:
            print(f"Loading pretrained value function from {self.cfg.vf_chkpt}")
            vf_state_dict = torch.load(self.cfg.vf_chkpt)
            self.vf.load_state_dict(vf_state_dict["vf"])
            self.qnet_params.load_state_dict(vf_state_dict["qfs"])
            self.qnet_target_params = self.qnet_params.data.clone()

        # optimizers
        self.actor_optimizer = optim.Adam(
            list(self.pi.parameters()), lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile
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

        # torch.compile
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            if not self.pretrained_vf:
                self._update_q = torch.compile(self._update_q, mode=mode)
                self._update_v = torch.compile(self._update_v, mode=mode)
            self._update_pi = torch.compile(self._update_pi, mode=mode)
            self._act = torch.compile(self._act, mode=mode)

        if self.cfg.cudagraphs:
            if not self.pretrained_vf:
                self._update_q = CudaGraphModule(self._update_q, in_keys=[], out_keys=[], warmup=5)
                self._update_v = CudaGraphModule(self._update_v, in_keys=[], out_keys=[], warmup=5)
            self._update_pi = CudaGraphModule(self._update_pi, in_keys=[], out_keys=[], warmup=5)
            self._act = CudaGraphModule(self._act, in_keys=[], out_keys=[], warmup=5)

    @torch.no_grad()
    def act(self, td):
        td_out = self._act(td.to(self.device, non_blocking=True))
        action = td_out["action"]
        return action, {}
    
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
        info = TensorDict()

        if not self.pretrained_vf:
            # update value
            info.update(self._update_v(batch))
            # update q-value
            info.update(self._update_q(batch))
            # update the target networks
            # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
            self.qnet_target_params.lerp_(self.qnet_params.data, self.tau)
        
        # update actor
        info.update(self._update_pi(batch))
        
        return info.detach().mean()
    
    def _update_q(self, batch):
        obs = batch["obs"]
        action = batch["action"]
        next_obs = batch["next_obs"]
        reward = batch["reward"]
        done = batch["done"]
        goal = batch["value_goal"]
        
        vf_next_target = self.vf_detach(next_obs, goal)
        next_q_value = reward.flatten() + (1 - done.flatten()) * (self.cfg.gamma ** self.n_step) * vf_next_target.flatten()  # whether a "done" signal is recieved depends on config; reward is n-step aggregated when n_step>1
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
            "pi": self.pi.state_dict(),
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
        self.vf.load_state_dict(state_dict["vf"])
        self.qnet_params.load_state_dict(state_dict["qfs"])
        self.qnet_target_params = self.qnet_params.data.clone()
        self.qnet_params.to_module(self.qnet)
        self.obs_mean = state_dict["obs_mean"]
        self.obs_std = state_dict["obs_std"]

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    