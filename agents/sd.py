"""Subgoal Diffuser (SD) agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule

from agents.modules.diffuser import SubgoalDiffuser
from agents.modules.actor_critic import Actor, QNetwork, VNetwork


class SD(nn.Module):
    """
    Samples candidate subgoals from a diffusion model p(s_{t+k} | s_t [, g]) and picks the one with
    the highest value toward the goal.

      - The chosen subgoal is handed to a goal-conditioned IQL low-level policy pi(a | s, subgoal).
      - obs=latent trains only the diffusion model; obs=rgb decodes its latent subgoals for a pixel low level.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda:0')

        #### Hyperparameters ####
        # general
        self.grad_clip_norm = cfg.grad_clip_norm
        self.pretrained_vf = self.cfg.sd["pretrained_vf"]
        self.pretrained_policy = self.cfg.sd["pretrained_policy"]
        # subgoals
        self.goal_conditioned = cfg.sd['goal_conditioned']
        self.n_sg_samples = cfg.sd['n_sg_samples']
        self.subgoal_steps = cfg.sd['subgoal_steps']
        self.goal_vicinity_threshold = cfg.sd['goal_vicinity_threshold']
        self.subgoal_reached_threshold = cfg.sd['subgoal_reached_threshold']
        self.disable_subgoal = cfg.sd['disable_subgoal']
        self.debug_subgoals = cfg.sd.get('debug_subgoals', False)
        # iql
        self.tau = self.cfg.sd['tau']
        self.expectile = self.cfg.sd['expectile']
        self.alpha = self.cfg.sd['alpha']
        self.pi_extract_type = self.cfg.sd["pi_extract_type"]
        self.pi_subgoal = self.cfg.sd["pi_subgoal"]
        self.policy_type = self.cfg.sd['policy_type']
        self.pi_noise = self.cfg.sd['pi_noise']
        self.pi_noise_clip = self.cfg.sd['pi_noise_clip']
        # visual tasks: obs=latent trains only the diffuser; obs=rgb samples with it over a pretrained pixel low level
        self.train_vf = not self.pretrained_vf and self.cfg.obs != 'latent'
        self.train_pi = not self.pretrained_policy and self.cfg.obs != 'latent'

        #### Low-level Value Functions ####
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

        #### Frozen autoencoder (obs=rgb): sample latent subgoals, decode them for the pixel low level ####
        self.ae = None
        sd_cfg = self.cfg
        if self.cfg.obs == "rgb":
            assert self.cfg.get("latent_encoder_checkpoint"), "sd with obs=rgb needs latent_encoder_checkpoint"
            assert self.pretrained_vf and self.pretrained_policy, (
                "sd with obs=rgb runs the pixel IQL of vf_chkpt / policy_chkpt: set sd.pretrained_vf=true and sd.pretrained_policy=true"
            )
            from agents.modules.autoencoder import FrozenAE
            self.ae = FrozenAE(self.cfg, device=self.device)
            sd_cfg = self.ae.latent_cfg
            print(f"SD samples subgoals in the latent of {self.cfg.latent_encoder_checkpoint} (z_dim={self.ae.z_dim}); decoded to images")

        # subgoal diffuser
        self.subgoal_diffuser = SubgoalDiffuser(sd_cfg, device=self.device)

        #### Low-level Policy ####
        self.pi = Actor(self.cfg, self.device)
        self.pi_detach = Actor(self.cfg, self.device)
        from_module(self.pi).data.to_module(self.pi_detach)  # Copy params to pi_detach without grad

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
            if self.ae is not None:
                # frames are normalized with the pixel low level's statistics
                self.obs_mean, self.obs_std = policy_state_dict["obs_mean"], policy_state_dict["obs_std"]

        #### Optimizers ####
        self.diffuser_optimizer = optim.Adam(
            self.subgoal_diffuser.parameters(),
            lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile,
        )
        
        if self.train_vf:
            self.v_optimizer = optim.Adam(
                list(self.vf.parameters()),
                lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile
            )

            self.q_optimizer = optim.Adam(
                self.qnet_params.values(include_nested=True, leaves_only=True),
                lr=self.cfg.lr, capturable=self.cfg.cudagraphs and not self.cfg.compile,
            )

        if self.train_pi:
            self.actor_optimizer = optim.Adam(
                list(self.pi.parameters()), lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile
            )

        if cfg.compile:
            mode = "reduce-overhead" if not cfg.cudagraphs else None
            self._update_diffuser = torch.compile(self._update_diffuser, mode=mode)
            if self.train_vf:
                self._update_q = torch.compile(self._update_q, mode=mode)
                self._update_v = torch.compile(self._update_v, mode=mode)
            if self.train_pi:
                self._update_pi = torch.compile(self._update_pi, mode=mode)
            self._act = torch.compile(self._act, mode=mode)
            self._sample_subgoals = torch.compile(self._sample_subgoals, mode=mode)

        if cfg.cudagraphs:
            self._update_diffuser = CudaGraphModule(self._update_diffuser, in_keys=[], out_keys=[], warmup=5)
            if self.train_vf:
                self._update_q = CudaGraphModule(self._update_q, in_keys=[], out_keys=[], warmup=5)
                self._update_v = CudaGraphModule(self._update_v, in_keys=[], out_keys=[], warmup=5)
            if self.train_pi:
                self._update_pi = CudaGraphModule(self._update_pi, in_keys=[], out_keys=[], warmup=5)
            self._act = CudaGraphModule(self._act, in_keys=[], out_keys=[], warmup=5)
            self._sample_subgoals = CudaGraphModule(self._sample_subgoals, in_keys=[], out_keys=[], warmup=5)

    @property
    def eval_policy(self):
        # obs=latent runs train only the diffuser and have no low level to roll out
        return self.cfg.obs != 'latent'

    def _encode(self, x):
        """Low-level observation -> the diffuser's (normalized latent) space; identity unless obs=rgb."""
        return x if self.ae is None else self.ae.encode(x)

    def _decode(self, z):
        """Diffuser's space -> low-level observation (decoded 0..255 frames); identity unless obs=rgb."""
        return z if self.ae is None else self.ae.decode(z)

    @torch.no_grad()
    def act(self, td):
        td = td.to(self.device, non_blocking=True)
        step = td["step"]
        goal = td["goal"]
        obs = td["obs"]
        
        if step == 0:
            if self.debug_subgoals:
                print("\n\nNew Episode")
            self._subgoal_samples = None
            self._last_subgoal_gen_step = 0
            self._subgoal = None # will fallback to generating a subgoal
        
        if self.disable_subgoal or (step >= (self.cfg.env_episode_length - 100)) or (self.vf_detach(obs, goal).abs() < self.goal_vicinity_threshold):
            self._subgoal = goal
            self._last_subgoal_gen_step = step

        else:
            # Replan triggers if:
            # - timeout: subgoal_steps steps since last replan
            # - reached its current subgoal based on value-function closeness to the target
            should_generate_subgoal = False
            if self._subgoal is None:
                should_generate_subgoal, reason = True, "no subgoal proposed so far"
            elif (step - self._last_subgoal_gen_step) >= self.subgoal_steps:
                should_generate_subgoal, reason = True, "timeout"
            elif self.vf_detach(obs, self._subgoal).abs() < self.subgoal_reached_threshold:
                should_generate_subgoal, reason = True, "reached"
            if should_generate_subgoal:

                N = self.n_sg_samples
                obs_rep = self._encode(obs).expand(N, -1)
                goal_rep = self._encode(goal).expand(N, -1)

                # sampled through _sample_subgoals so it can be compiled / cudagraph-captured
                sg_td = TensorDict({"obs": obs_rep, "goal": goal_rep}, batch_size=[])
                sg_out = self._sample_subgoals(sg_td)
                subgoal_samples = self._decode(sg_out["subgoal_samples"])  # low-level space (decoded images for rgb)
                
                subgoal_values = self.vf_detach(subgoal_samples, goal.expand(N, *goal.shape[1:]))

                subgoal_value, argmax_idx = torch.max(subgoal_values, dim=0)
                self._subgoal = subgoal_samples[argmax_idx]
                self._subgoal_samples = subgoal_samples.detach().clone()

                if self.debug_subgoals:
                    print(f"Step {step} - replanning due to reason={reason}, chose subgoal with value={subgoal_value}")

        td["subgoal"] = self._subgoal
        td_out = self._act(td)
        action = td_out["action"]

        info = {"subgoal": self._subgoal.detach().clone(), "plot_subgoals": True}
        if self.ae is not None:
            info["subgoal_image"] = self._subgoal.reshape(1, *self._subgoal.shape[-3:]).detach().clone()
        if self._subgoal_samples is not None:
            info["subgoal_samples"] = self._subgoal_samples
        return action, info

    def _sample_subgoals(self, td):
        subgoal_samples = self.subgoal_diffuser.sample_fast(td["obs"], td["goal"])
        return TensorDict(subgoal_samples=subgoal_samples)

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
        info = TensorDict()

        if self.train_vf:
            # update value
            info = self._update_v(batch)
            # update q-value
            info.update(self._update_q(batch))
            # update the target networks
            # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
            self.qnet_target_params.lerp_(self.qnet_params.data, self.tau)

        if self.train_pi:
            # update actor
            info.update(self._update_pi(batch))

        # update subgoal diffuser
        info.update(self._update_diffuser(batch))
        
        return info.detach().mean()

    def _update_diffuser(self, batch):
        obs = batch["obs"]
        subgoal = batch["policy_subgoal"] if self.goal_conditioned else batch["sd_next_obs"]
        goal = batch["policy_goal"]

        self.diffuser_optimizer.zero_grad()
        loss, _ = self.subgoal_diffuser.loss(obs, subgoal, goal)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.subgoal_diffuser.parameters(), self.grad_clip_norm
        )
        self.diffuser_optimizer.step()

        return TensorDict(
            diffusion_loss=loss.detach(),
            diffusion_gradnorm=grad_norm,
        )
    
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
        torch.save({
            "subgoal_diffuser": self.subgoal_diffuser.state_dict(),
            "vf": self.vf.state_dict(),
            "qfs": self.qnet_params.state_dict(),
            "pi": self.pi.state_dict(),
            "obs_mean": obs_mean,
            "obs_std": obs_std,
        }, fp)

    def load(self, fp):
        state_dict = torch.load(fp)
        self.subgoal_diffuser.load_state_dict(state_dict["subgoal_diffuser"])
        if not self.pretrained_vf:
            self.vf.load_state_dict(state_dict["vf"])
            self.qnet_params.load_state_dict(state_dict["qfs"])
            self.qnet_target_params = self.qnet_params.data.clone()
            self.qnet_params.to_module(self.qnet)
        if not self.pretrained_policy:
            self.pi.load_state_dict(state_dict["pi"])
        if self.ae is not None:
            # obs=rgb: the checkpoint is an obs=latent run; its normalization is the diffuser's latent's
            self.ae.set_stats(state_dict["obs_mean"], state_dict["obs_std"])
        else:
            self.obs_mean = state_dict["obs_mean"]
            self.obs_std = state_dict["obs_std"]

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        