"""Image autoencoder with a goal-conditioned reward head on the latent.

Maps image frames to a low-dimensional latent z that the latent-space agents (obs=latent) use as their observation.
The latent is trained with two losses:
  - pixel reconstruction (MSE on [0, 1] images);
  - a goal-reached classifier r(z, z_g) on the task's success label. Its gradients reach the encoder, so z keeps what
    distinguishes being at the goal, not only what is cheap to reconstruct.
A small L2 penalty keeps the latent scale bounded. With kl_weight > 0 the encoder becomes a Gaussian VAE posterior;
the default is deterministic.
"""

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.modules.actor_critic import ResnetStack


class ConvEncoder(nn.Module):
    """Impala-style image encoder (a stack of residual conv blocks over a single frame): [B, H, W, C] uint8
    frames -> [B, out_dim] features. Each stack halves the spatial resolution."""

    def __init__(self, img_shape, conv_channels, h_dim, out_dim, device=None):
        super().__init__()
        img_h, img_w, img_c = img_shape
        c = [img_c, *conv_channels]
        self.resnet = nn.Sequential(*[ResnetStack(c[i], c[i + 1], device) for i in range(len(c) - 1)])

        with torch.no_grad():
            dummy = torch.zeros(1, img_c, img_h, img_w, device=device)
            feat = self.resnet(dummy)
            self.conv_out_shape = tuple(feat.shape[1:])  # [C, H', W']
            conv_out_flat_dim = feat.numel()

        self.fc1 = nn.Linear(conv_out_flat_dim, h_dim, device=device)
        self.fc_out = nn.Linear(h_dim, out_dim, device=device)

    def forward(self, img):
        x = img.permute(0, 3, 1, 2).float() / 255.0  # [B, C, H, W] in [0, 1]
        x = F.relu(self.resnet(x))
        x = F.gelu(self.fc1(x.reshape(x.shape[0], -1)))
        return self.fc_out(x)


class ConvDecoder(nn.Module):
    """Mirror of ConvEncoder: [B, z_dim] -> [B, H, W, C] images in [0, 1]. Nearest-neighbour upsampling followed by
    a 3x3 convolution per stack (no transposed convolutions, which cause checkerboard artifacts). Reproduces the
    input size when H and W are divisible by 2 ** len(conv_channels)."""

    def __init__(self, z_dim, conv_out_shape, conv_channels, h_dim, img_shape, device=None):
        super().__init__()
        self.conv_out_shape = conv_out_shape
        img_c = img_shape[-1]
        flat_dim = math.prod(conv_out_shape)

        self.fc1 = nn.Linear(z_dim, h_dim, device=device)
        self.fc2 = nn.Linear(h_dim, flat_dim, device=device)

        channels = list(reversed(conv_channels))  # deepest first, mirroring the encoder
        blocks = []
        for i in range(len(channels)):
            c_out = channels[i + 1] if i + 1 < len(channels) else channels[-1]
            blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(channels[i], c_out, kernel_size=3, stride=1, padding=1, device=device),
                nn.ReLU(),
            ))
        self.blocks = nn.Sequential(*blocks)
        self.conv_out = nn.Conv2d(channels[-1], img_c, kernel_size=3, stride=1, padding=1, device=device)

    def forward(self, z):
        x = F.gelu(self.fc1(z))
        x = F.gelu(self.fc2(x)).view(-1, *self.conv_out_shape)
        x = self.blocks(x)
        x = torch.sigmoid(self.conv_out(x))          # [B, C, H, W] in [0, 1]
        return x.permute(0, 2, 3, 1)                 # [B, H, W, C]


class RewardHead(nn.Module):
    """Goal-reached classifier on the latent: logit of P(at goal | z, z_g)."""

    def __init__(self, z_dim, h_dim, n_layers, device=None):
        super().__init__()
        dims = [2 * z_dim] + [h_dim] * n_layers
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], device=device) for i in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(h_dim, device=device) for _ in range(n_layers)])
        self.fc_out = nn.Linear(h_dim, 1, device=device)

    def forward(self, z, z_goal):
        x = torch.cat([z, z_goal], dim=-1)
        for fc, ln in zip(self.layers, self.norms):
            x = ln(F.gelu(fc(x)))
        return self.fc_out(x)


class Autoencoder(nn.Module):
    """Encoder + decoder + goal-conditioned reward head.

    `arch` is the full architecture spec, stamped into every checkpoint so `Autoencoder.from_checkpoint` can rebuild
    the module without a hydra config.
    """

    ARCH_KEYS = ('img_shape', 'z_dim', 'conv_channels', 'h_dim', 'reward_h_dim', 'reward_n_layers', 'kl_weight')

    def __init__(self, arch, device=None):
        super().__init__()
        self.arch = dict(arch)
        self.img_shape = tuple(arch['img_shape'])
        self.z_dim = int(arch['z_dim'])
        self.kl_weight = float(arch['kl_weight'])
        self.is_vae = self.kl_weight > 0.0

        enc_out_dim = 2 * self.z_dim if self.is_vae else self.z_dim
        self.encoder = ConvEncoder(self.img_shape, arch['conv_channels'], arch['h_dim'], enc_out_dim, device=device)
        self.decoder = ConvDecoder(self.z_dim, self.encoder.conv_out_shape, arch['conv_channels'], arch['h_dim'],
                                   self.img_shape, device=device)
        self.reward_head = RewardHead(self.z_dim, arch['reward_h_dim'], arch['reward_n_layers'], device=device)

    @classmethod
    def arch_from_cfg(cls, cfg):
        """Architecture spec from the `ae:` config block plus the environment's image shape."""
        ae = cfg.ae
        return dict(
            img_shape=tuple(cfg.obs_shape),
            z_dim=int(ae['z_dim']),
            conv_channels=list(ae['conv_channels']),
            h_dim=int(ae['h_dim']),
            reward_h_dim=int(ae['reward_h_dim']),
            reward_n_layers=int(ae['reward_n_layers']),
            kl_weight=float(ae['kl_weight']),
        )

    @classmethod
    def from_checkpoint(cls, fp, device=None):
        """Rebuild a trained autoencoder from any checkpoint that carries 'ae_arch' + 'ae' (as saved by its training run)."""
        ckpt = torch.load(fp, map_location=device or 'cpu', weights_only=False)
        assert 'ae_arch' in ckpt and 'ae' in ckpt, f"{fp} is not an autoencoder checkpoint (no 'ae_arch' / 'ae')"
        ae = cls(ckpt['ae_arch'], device=device)
        ae.load_state_dict(ckpt['ae'])
        ae.eval()
        ae.requires_grad_(False)
        return ae

    def encode(self, img, sample=False):
        """[B, H, W, C] uint8 frames -> [B, z_dim] latents. The VAE path returns the posterior mean unless
        `sample` (dataset conversion and control always use the mean, so the representation is deterministic)."""
        out = self.encoder(img)
        if not self.is_vae:
            return out
        mu, log_std = out.chunk(2, dim=-1)
        if not sample:
            return mu
        log_std = log_std.clamp(-5.0, 2.0)
        return mu + torch.randn_like(mu) * log_std.exp()

    def _encode_dist(self, img):
        """Encoder output for training: (z, kl) with kl = 0 on the deterministic path."""
        out = self.encoder(img)
        if not self.is_vae:
            return out, out.new_zeros(())
        mu, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(-5.0, 2.0)
        z = mu + torch.randn_like(mu) * log_std.exp()
        kl = (-log_std + 0.5 * (log_std.exp() ** 2 + mu ** 2) - 0.5).sum(dim=-1).mean()
        return z, kl

    def decode(self, z):
        return self.decoder(z)

    def losses(self, obs, next_obs, goal, at_goal, l2_weight, reward_weight):
        """Reconstruct the three frames the batch provides and classify (next_obs, goal) as at-goal.

        Args:
            obs, next_obs, goal: [B, H, W, C] uint8 frames.
            at_goal: [B, 1] float labels in {0, 1} (buffer reward == 0).
        """
        b = obs.shape[0]
        imgs = torch.cat([obs, goal, next_obs], dim=0)       # one encoder pass over all three
        z, kl = self._encode_dist(imgs)

        # Reconstruct observations and goals; next_obs is encoded only for the classifier below (it is drawn from
        # the same frame distribution as obs, so decoding it would cost a third of the decoder for nothing).
        recon = self.decode(z[:2 * b])
        target = imgs[:2 * b].float() / 255.0                # [2B, H, W, C] in [0, 1]
        recon_loss = F.mse_loss(recon, target)

        z_goal, z_next = z[b:2 * b], z[2 * b:]
        logit = self.reward_head(z_next, z_goal)
        reward_loss = F.binary_cross_entropy_with_logits(logit, at_goal)

        l2 = z.pow(2).mean()
        total = recon_loss + reward_weight * reward_loss + l2_weight * l2 + self.kl_weight * kl

        with torch.no_grad():
            prob = torch.sigmoid(logit)
            pred = prob > 0.5
            acc = (pred.float() == at_goal).float().mean()
            # Class-wise accuracy: the positives are rare, so overall accuracy alone hides a collapsed head.
            # Kept branch-free (no .any() / boolean masking) so this stays inside the compiled + cudagraph region.
            pos = at_goal > 0.5
            acc_pos = (pred & pos).sum() / pos.sum().clamp(min=1)
            acc_neg = (~pred & ~pos).sum() / (~pos).sum().clamp(min=1)

        return dict(
            ae_loss=total,
            recon_loss=recon_loss.detach(),
            reward_loss=reward_loss.detach(),
            kl=kl.detach(),
            latent_l2=l2.detach(),
            latent_std=z.std(dim=0).mean().detach(),
            reward_acc=acc,
            reward_acc_pos=acc_pos,
            reward_acc_neg=acc_neg,
            reward_pos_rate=at_goal.mean().detach(),
        )


class FrozenAE:
    """Frozen autoencoder that lets the high level of a pixel (obs=rgb) run work in its latent.

    encode() maps frames to latents normalized as an obs=latent run normalizes its observations (clipped to [-5, 5]);
    decode() maps them back to frames on the pixel agents' 0..255 scale. The normalization statistics come from a
    checkpoint (set_stats) or, for a new run, from the encoded dataset (fit_stats).
    """

    def __init__(self, cfg, device=None):
        self.cfg = cfg
        self.device = device
        self.model = Autoencoder.from_checkpoint(cfg.latent_encoder_checkpoint, device=device)
        self.z_dim = self.model.z_dim
        # the run's config with the latent as observation space, for the networks that work in it
        self.latent_cfg = dataclasses.replace(cfg, obs='latent', obs_shape=(self.z_dim,))
        self.mean = self.std = None

    def set_stats(self, mean, std):
        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=self.device)
        self.std = torch.as_tensor(std, dtype=torch.float32, device=self.device)

    def fit_stats(self):
        """Mean / std of the encoded dataset (built if missing), as the buffer of an obs=latent run computes them."""
        from common.utils.dataset import load_latent_dataset
        z = torch.from_numpy(load_latent_dataset(self.cfg, self.model)['observations'])
        self.set_stats(z.mean(0), torch.sqrt(z.var(0)))

    @torch.no_grad()
    def encode(self, frames):
        """[..., H, W, C] frames -> [..., z_dim] normalized latents."""
        assert self.mean is not None, "FrozenAE: no latent statistics (set_stats / fit_stats)"
        lead = frames.shape[:-3]
        z = self.model.encode(frames.reshape(-1, *frames.shape[-3:])).reshape(*lead, -1)
        return torch.clip((z - self.mean) / self.std, -5, 5)

    @torch.no_grad()
    def decode(self, z):
        """[..., z_dim] normalized latents -> [..., H, W, C] frames (0..255 floats)."""
        lead = z.shape[:-1]
        img = self.model.decode((z * self.std + self.mean).reshape(-1, z.shape[-1])).clamp(0, 1) * 255.0
        return img.reshape(*lead, *img.shape[1:])
