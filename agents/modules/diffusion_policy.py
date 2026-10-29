"""
A Gaussian diffusion model over an action chunk [a_0, ..., a_{T-1}], FiLM-conditioned on the
observation (and the diffusion timestep). Trained with an L1 noise-prediction loss (optionally min-SNR-gamma weighted);
sampled with DDIM, or with the full DDPM reverse process when ddim_sampling_steps is null.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#### Diffusion process utilities ####

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """Cosine schedule (Nichol & Dhariwal 2021)."""
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02, dtype=torch.float32):
    """Linear schedule (Ho et al. 2020)."""
    betas = np.linspace(beta_start, beta_end, timesteps)
    return torch.tensor(betas, dtype=dtype)


#### Denoiser network ####

class FiLMBlock(nn.Module):
    def __init__(self, in_dim, out_dim, cond_dim, device=None):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, device=device)
        self.ln = nn.LayerNorm(out_dim, elementwise_affine=False, device=device) # no learned parameters
        self.film = nn.Linear(cond_dim, 2 * out_dim, device=device)

        # zero-initialize the FiLM projections
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, cond):
        h = self.ln(F.gelu(self.linear(x)))
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        return h * (1 + gamma) + beta


class DiffusionPolicyDenoiser(nn.Module):
    """MLP denoiser over a flattened action chunk, FiLM-conditioned on [obs, t]."""

    def __init__(self, obs_dim, action_dim, horizon, h_dim, device=None):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        in_dim = horizon * action_dim

        self.cond_mlp = nn.Sequential(
            nn.Linear(obs_dim + 1, h_dim, device=device),
            nn.GELU(),
            nn.Linear(h_dim, h_dim, device=device),
        )
        self.blocks = nn.ModuleList([
            FiLMBlock(in_dim if i == 0 else h_dim, h_dim, h_dim, device=device)
            for i in range(4)
        ])
        self.out = nn.Linear(h_dim, in_dim, device=device)

    def forward(self, x, obs, t):
        # x: (B, horizon, action_dim), obs: (B, obs_dim), t: (B,)
        B = x.shape[0]
        h = x.reshape(B, -1)
        cond_emb = self.cond_mlp(torch.cat([obs, t.view(B, 1).to(obs.dtype)], dim=1))
        for block in self.blocks:
            h = block(h, cond_emb)
        return self.out(h).view(B, self.horizon, self.action_dim)


#### Diffusion process ####

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        n_timesteps=1000,
        clip_denoised=True,
        predict_epsilon=True,
        beta_schedule='cosine',  # 'cosine' | 'linear'
        snr_gamma=None,
        clip_denoised_range=(-1., 1.),
    ):
        super().__init__()
        self.model = model

        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule '{beta_schedule}'. Choose 'cosine' or 'linear'.")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.clip_denoised_range = tuple(clip_denoised_range)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        # min-SNR-gamma loss weighting (Hang et al. 2023): w(t) = min(SNR(t), gamma) / SNR(t)
        if snr_gamma is not None:
            snr = alphas_cumprod / (1. - alphas_cumprod)
            loss_weight = torch.minimum(snr, torch.full_like(snr, snr_gamma)) / snr
            self.register_buffer('loss_weight', loss_weight)
        else:
            self.loss_weight = None

    # sampling #

    def predict_start_from_noise(self, x_t, t, noise):
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    @torch.no_grad()
    def ddpm_sample(self, shape, obs, temperature=1.0):
        """Full DDPM reverse process over all n_timesteps."""
        x = torch.randn(shape, device=self.betas.device)
        for i in reversed(range(0, self.n_timesteps)):
            t = torch.full((shape[0],), i, device=x.device, dtype=torch.long)
            x_recon = self.predict_start_from_noise(x, t, self.model(x, obs, t))
            if self.clip_denoised:
                x_recon.clamp_(*self.clip_denoised_range)
            model_mean, model_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
            noise = torch.randn_like(x) * temperature  # < 1 sharpens, > 1 broadens
            noise[t == 0] = 0
            x = model_mean + torch.exp(0.5 * model_log_variance) * noise
        return x

    @torch.no_grad()
    def ddim_sample(self, shape, obs, ddim_steps=50, eta=0.0, temperature=1.0):
        """DDIM reverse process (Song et al. 2020). eta=0 is deterministic, eta=1 recovers DDPM-equivalent noise."""
        device = self.betas.device
        batch_size = shape[0]

        step_ratio = max(self.n_timesteps // ddim_steps, 1)
        ddim_timesteps = np.arange(0, self.n_timesteps, step_ratio)[::-1].copy()

        x = torch.randn(shape, device=device)
        for i, t_val in enumerate(ddim_timesteps):
            t = torch.full((batch_size,), int(t_val), device=device, dtype=torch.long)
            t_prev_val = int(ddim_timesteps[i + 1]) if i + 1 < len(ddim_timesteps) else -1

            x_recon = self.predict_start_from_noise(x, t, self.model(x, obs, t))
            if self.clip_denoised:
                x_recon.clamp_(*self.clip_denoised_range)

            # eps re-derived from the (possibly clipped) x0, so the DDIM direction is consistent with it
            sqrt_alpha_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
            sqrt_one_minus_alpha_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            eps = (x - sqrt_alpha_t * x_recon) / sqrt_one_minus_alpha_t.clamp(min=1e-8)

            alpha_t = extract(self.alphas_cumprod, t, x.shape)
            if t_prev_val >= 0:
                t_prev = torch.full((batch_size,), t_prev_val, device=device, dtype=torch.long)
                alpha_t_prev = extract(self.alphas_cumprod, t_prev, x.shape)
            else:
                alpha_t_prev = torch.ones_like(alpha_t)  # last step collapses to x = x_recon

            sigma = eta * torch.sqrt(
                ((1 - alpha_t_prev) / (1 - alpha_t).clamp(min=1e-8)) *
                (1 - alpha_t / alpha_t_prev.clamp(min=1e-8))
            ).clamp(min=0)

            dir_xt = torch.sqrt((1 - alpha_t_prev - sigma ** 2).clamp(min=0)) * eps
            noise = torch.randn_like(x) * temperature if t_val > 0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_t_prev) * x_recon + dir_xt + sigma * noise
        return x

    # training #

    def q_sample(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def loss(self, x_start, obs):
        t = torch.randint(0, self.n_timesteps, (len(x_start),), device=x_start.device).long()
        noise = torch.randn_like(x_start)
        x_recon = self.model(self.q_sample(x_start, t, noise), obs, t)
        target = noise if self.predict_epsilon else x_start

        per_sample_loss = F.l1_loss(x_recon, target, reduction='none').mean(dim=(1, 2))
        if self.loss_weight is not None:
            per_sample_loss = per_sample_loss * extract(self.loss_weight, t, per_sample_loss.shape)
        return per_sample_loss.mean(), {}


#### Agent-facing facade ####

class MLPDiffusionPolicy(nn.Module):
    """Diffusion policy over an action chunk of length wm_timescale, conditioned on the observation."""

    def __init__(self, cfg, device=None):
        super().__init__()
        assert cfg.obs == 'state', 'MLPDiffusionPolicy only supports state observations.'
        acwm = cfg.acwm
        self.action_dim = cfg.action_dim
        self._horizon = acwm['wm_timescale']

        model = DiffusionPolicyDenoiser(
            cfg.obs_shape[0], self.action_dim, self._horizon, cfg.arch_mlp['h_dim'], device=device,
        )
        self.diffusion = GaussianDiffusion(
            model,
            n_timesteps=acwm['n_diffusion_steps'],
            clip_denoised=acwm['clip_denoised'],
            predict_epsilon=acwm['predict_epsilon'],
            beta_schedule=acwm['beta_schedule'],
            snr_gamma=acwm['snr_gamma'],
            clip_denoised_range=acwm['clip_denoised_range'],
        ).to(device)

        self._ddim_steps = acwm['ddim_sampling_steps']  # null -> full DDPM
        self._ddim_eta = acwm['ddim_eta']  # 0 = deterministic DDIM, 1 = DDPM-equivalent noise
        self._temperature = acwm['sampling_temperature']

    def loss(self, obs, actions):
        """actions: ground-truth action chunk, shape (B, horizon, action_dim)."""
        return self.diffusion.loss(actions, obs)

    def sample(self, obs):
        """obs: (B, obs_dim) -> sampled action chunks (B, horizon, action_dim)."""
        shape = (obs.shape[0], self._horizon, self.action_dim)
        if self._ddim_steps is None:
            return self.diffusion.ddpm_sample(shape, obs, temperature=self._temperature)
        return self.diffusion.ddim_sample(
            shape, obs, ddim_steps=self._ddim_steps, eta=self._ddim_eta, temperature=self._temperature,
        )
