"""Image autoencoder (AE) agent."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule

from agents.modules.autoencoder import Autoencoder


class AE(nn.Module):
    """
    Learns a compact latent observation (obs=latent) to use in place of raw images.

      - Loss: frame reconstruction + a goal-reached classifier on (z_{s'}, z_g), which keeps task-relevant
        information in the latent + a small L2 penalty on z (+ KL if kl_weight > 0).
      - Learns no policy (no rollouts); its checkpoint is loaded via latent_encoder_checkpoint.
      - Requires obs=rgb and reward=env (the goal-reached label).
    """

    eval_policy = False  # trains no policy: the trainer skips rollout evaluation and best.pt

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda:0')

        assert cfg.obs == 'rgb', f"the ae agent encodes images: set obs=rgb (got obs={cfg.obs})"
        assert cfg.reward != 'unsupervised', (
            "the ae reward head needs the supervised goal-reached label: set reward=env"
        )

        self.grad_clip_norm = cfg.grad_clip_norm
        self.reward_weight = float(cfg.ae['reward_weight'])
        self.l2_weight = float(cfg.ae['l2_weight'])
        self.recon_every = int(cfg.ae['recon_every'] or 0)  # 0 disables the reconstruction grids

        self.ae = Autoencoder(Autoencoder.arch_from_cfg(cfg), device=self.device)

        if cfg.train_from_checkpoint:
            self.load(cfg.checkpoint)

        self.optimizer = optim.Adam(self.ae.parameters(), lr=cfg.lr,
                                    capturable=cfg.cudagraphs and not cfg.compile)

        if cfg.compile:
            mode = "reduce-overhead" if not cfg.cudagraphs else None
            self._update_ae = torch.compile(self._update_ae, mode=mode)
        if cfg.cudagraphs:
            self._update_ae = CudaGraphModule(self._update_ae, in_keys=[], out_keys=[], warmup=5)

        self._recon_dir = Path(cfg.log_dir) / 'recon'

    @torch.no_grad()
    def act(self, td):
        raise NotImplementedError("the ae agent trains a representation, not a policy")

    def update(self, batch, step):
        # Consume the update's outputs before anything else runs: under cudagraphs they live in static buffers.
        info = self._update_ae(batch).detach().mean()
        if self.recon_every and step % self.recon_every == 0:
            self._save_recon_grid(batch, step)
        return info

    def _update_ae(self, batch):
        obs = batch["obs"]                 # [B, H, W, C] uint8
        next_obs = batch["next_obs"]
        goal = batch["value_goal"]
        at_goal = (batch["reward"] == 0).float()  # [B, 1]: 1 at the goal, 0 otherwise

        losses = self.ae.losses(obs, next_obs, goal, at_goal, self.l2_weight, self.reward_weight)
        total_loss = losses["ae_loss"]

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.ae.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        td = TensorDict(ae_loss=total_loss.detach(), ae_gradnorm=grad_norm)
        for key, val in losses.items():
            if key != "ae_loss":
                td[key] = val
        return td

    @torch.no_grad()
    def _save_recon_grid(self, batch, step, n=8):
        """Write a PNG of n frames above their reconstructions, for eyeballing what the latent keeps."""
        from PIL import Image

        obs = batch["obs"][:n]
        recon = self.ae.decode(self.ae.encode(obs)).clamp(0, 1)
        top = obs.float().cpu().numpy()
        bottom = (recon * 255.0).cpu().numpy()
        grid = np.concatenate([
            np.concatenate(list(top), axis=1),
            np.concatenate(list(bottom), axis=1),
        ], axis=0).astype(np.uint8)
        self._recon_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(grid).save(self._recon_dir / f'{step:08d}.png')

    def save(self, fp, obs_mean=0, obs_std=1):
        torch.save({
            "ae_arch": self.ae.arch,
            "ae": self.ae.state_dict(),
            "obs_mean": obs_mean,
            "obs_std": obs_std,
        }, fp)

    def load(self, fp):
        state_dict = torch.load(fp, weights_only=False)
        assert state_dict["ae_arch"] == self.ae.arch, (
            f"{fp} was saved with ae_arch={state_dict['ae_arch']} but this run builds {self.ae.arch}"
        )
        self.ae.load_state_dict(state_dict["ae"])
        self.obs_mean = state_dict["obs_mean"]
        self.obs_std = state_dict["obs_std"]

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
