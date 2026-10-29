"""
Subgoal diffuser: a Gaussian diffusion model over the short sequence [obs, subgoal] (or [obs, subgoal, goal] if
goal-conditioned). obs (and goal) are held at their given values throughout denoising; only the subgoal slot is
generated and supervised.
"""

import math

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


def apply_conditioning(x, conditions):
    for t, val in conditions.items():
        x[:, t, :] = val.clone()
    return x


#### Denoiser network ####

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class FiLMBlock(nn.Module):
    def __init__(self, in_dim, out_dim, cond_dim, device=None):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, device=device)
        self.ln = nn.LayerNorm(out_dim, elementwise_affine=False, device=device)
        self.film = nn.Linear(cond_dim, 2 * out_dim, device=device)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, cond):
        h = self.ln(F.gelu(self.linear(x)))
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        return h * (1 + gamma) + beta


class MLPDenoiser(nn.Module):
    """MLP denoiser over a fixed-length sequence of elem_dim-dimensional vectors, FiLM-conditioned on the timestep."""

    def __init__(self, seq_len, elem_dim, h_dim, n_blocks=4, device=None):
        super().__init__()
        self.seq_len = seq_len
        self.elem_dim = elem_dim
        in_dim = seq_len * elem_dim

        self.time_emb = SinusoidalPosEmb(h_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim, device=device),
            nn.GELU(),
            nn.Linear(h_dim, h_dim, device=device),
        )
        self.blocks = nn.ModuleList([
            FiLMBlock(in_dim if i == 0 else h_dim, h_dim, h_dim, device=device)
            for i in range(n_blocks)
        ])
        self.out = nn.Linear(h_dim, in_dim, device=device)

    def forward(self, x, cond, t):
        B = x.shape[0]
        h = x.reshape(B, -1)
        cond_emb = self.cond_mlp(self.time_emb(t.float()))
        for block in self.blocks:
            h = block(h, cond_emb)
        return self.out(h).view(B, self.seq_len, self.elem_dim)


#### Core diffusion process ####

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        horizon,
        elem_dim,
        subgoal_slot,  # the only slot that is generated / supervised; the others are held at their values in cond
        n_timesteps=1000,
        clip_denoised=True,
        predict_epsilon=True,
        beta_schedule='cosine',  # 'cosine' | 'linear'
    ):
        super().__init__()
        self.horizon = horizon
        self.elem_dim = elem_dim
        self.model = model
        self.subgoal_slot = subgoal_slot

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
    def ddpm_sample(self, shape, cond, temperature=1.0):
        """Full DDPM reverse process over all n_timesteps."""
        x = apply_conditioning(torch.randn(shape, device=self.betas.device), cond)
        for i in reversed(range(0, self.n_timesteps)):
            t = torch.full((shape[0],), i, device=x.device, dtype=torch.long)
            x_recon = self.predict_start_from_noise(x, t, self.model(x, cond, t))
            if self.clip_denoised:
                x_recon.clamp_(-1., 1.)
            model_mean, model_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
            noise = torch.randn_like(x) * temperature  # < 1 sharpens, > 1 broadens
            noise[t == 0] = 0
            x = apply_conditioning(model_mean + torch.exp(0.5 * model_log_variance) * noise, cond)
        return x

    @torch.no_grad()
    def ddim_sample(self, shape, cond, ddim_steps=50, eta=0.0, temperature=1.0):
        """DDIM reverse process (Song et al. 2020). eta=0 is deterministic, eta=1 recovers DDPM-equivalent noise."""
        device = self.betas.device
        batch_size = shape[0]

        step_ratio = max(self.n_timesteps // ddim_steps, 1)
        ddim_timesteps = np.arange(0, self.n_timesteps, step_ratio)[::-1].copy()

        x = apply_conditioning(torch.randn(shape, device=device), cond)
        for i, t_val in enumerate(ddim_timesteps):
            t = torch.full((batch_size,), int(t_val), device=device, dtype=torch.long)
            t_prev_val = int(ddim_timesteps[i + 1]) if i + 1 < len(ddim_timesteps) else -1

            x_recon = self.predict_start_from_noise(x, t, self.model(x, cond, t))
            if self.clip_denoised:
                x_recon.clamp_(-1., 1.)

            sqrt_alpha_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
            sqrt_one_minus_alpha_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            eps = (x - sqrt_alpha_t * x_recon) / sqrt_one_minus_alpha_t.clamp(min=1e-8)

            alpha_t = extract(self.alphas_cumprod, t, x.shape)
            if t_prev_val >= 0:
                t_prev = torch.full((batch_size,), t_prev_val, device=device, dtype=torch.long)
                alpha_t_prev = extract(self.alphas_cumprod, t_prev, x.shape)
            else:
                alpha_t_prev = torch.ones_like(alpha_t)

            sigma = eta * torch.sqrt(
                ((1 - alpha_t_prev) / (1 - alpha_t).clamp(min=1e-8)) *
                (1 - alpha_t / alpha_t_prev.clamp(min=1e-8))
            ).clamp(min=0)

            dir_xt = torch.sqrt((1 - alpha_t_prev - sigma ** 2).clamp(min=0)) * eps
            noise = torch.randn_like(x) * temperature if t_val > 0 else torch.zeros_like(x)

            x = apply_conditioning(torch.sqrt(alpha_t_prev) * x_recon + dir_xt + sigma * noise, cond)
        return x

    @torch.no_grad()
    def conditional_sample(self, cond, use_ddim=False, ddim_steps=50, ddim_eta=0.0, temperature=1.0):
        shape = (len(cond[0]), self.horizon, self.elem_dim)
        if use_ddim:
            return self.ddim_sample(shape, cond, ddim_steps=ddim_steps, eta=ddim_eta, temperature=temperature)
        return self.ddpm_sample(shape, cond, temperature=temperature)

    # training #

    def q_sample(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def loss(self, x_start, cond):
        t = torch.randint(0, self.n_timesteps, (len(x_start),), device=x_start.device).long()
        noise = torch.randn_like(x_start)
        x_noisy = apply_conditioning(self.q_sample(x_start, t, noise), cond)
        x_recon = apply_conditioning(self.model(x_noisy, cond, t), cond)
        target = noise if self.predict_epsilon else x_start

        # L1 on the subgoal slot only
        per_sample_loss = F.l1_loss(x_recon[:, self.subgoal_slot], target[:, self.subgoal_slot], reduction='none').mean(dim=-1)
        return per_sample_loss.mean(), {}


#### Agent-facing facade ####

class SubgoalDiffuser(nn.Module):
    """
    Subgoal planner: generates a subgoal conditioned on obs and, if cfg.sd['goal_conditioned'] is set,
    also on goal (as a third slot held at its given value).
    goal_conditioned is a build-time flag -- fixed for the lifetime of the model, safe
    for torch.compile / cudagraphs.
    """

    def __init__(self, cfg, device=None):
        super().__init__()
        assert cfg.obs in ('state', 'latent'), 'SubgoalDiffuser only supports vector (state / autoencoder latent) observations.'
        self.cfg = cfg
        self.obs_dim = cfg.obs_shape[0]
        self.goal_conditioned = cfg.sd['goal_conditioned']
        self.subgoal_slot = 1
        self._horizon = 3 if self.goal_conditioned else 2

        model = MLPDenoiser(
            seq_len=self._horizon, elem_dim=self.obs_dim, h_dim=cfg.arch_mlp['h_dim'], device=device,
        )

        sd = cfg.sd
        self.diffusion = GaussianDiffusion(
            model,
            horizon=self._horizon,
            elem_dim=self.obs_dim,
            subgoal_slot=self.subgoal_slot,
            n_timesteps=sd['n_diffusion_steps'],
            clip_denoised=sd['clip_denoised'],
            predict_epsilon=sd['predict_epsilon'],
            beta_schedule=sd['beta_schedule'],
        ).to(device)

        ddim_steps = sd['ddim_sampling_steps']
        self._sample_kwargs = dict(
            use_ddim=ddim_steps is not None,
            ddim_steps=ddim_steps or 50,
            ddim_eta=sd['ddim_eta'],
            temperature=sd['sampling_temperature'],
        )

    def _build_cond(self, obs, goal):
        B = obs.shape[0]
        cond = {0: obs.view(B, -1).detach()}
        if self.goal_conditioned:
            cond[2] = goal.view(B, -1).detach()
        return cond

    def loss(self, obs, subgoal, goal=None):
        B = obs.shape[0]
        slots = [obs.unsqueeze(1), subgoal.unsqueeze(1)]
        if self.goal_conditioned:
            slots.append(goal.unsqueeze(1))
        x = torch.cat(slots, dim=1).view(B, self._horizon, -1)
        return self.diffusion.loss(x, self._build_cond(obs, goal))

    def sample_fast(self, obs, goal=None):
        """Static-shape sampling path, safe for compile/cudagraph capture."""
        cond = self._build_cond(obs, goal)
        subgoal = self.diffusion.conditional_sample(cond, **self._sample_kwargs)[:, self.subgoal_slot, :]
        assert torch.isfinite(subgoal).all(), "subgoal sample contains NaN/inf -- unclipped diffusion diverged"
        return subgoal
