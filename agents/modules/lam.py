"""Latent Action Model (LAM): a VAE that infers a latent action from a state transition and predicts the transition from it."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentActionModel(nn.Module):
    """
    Latent action model over transitions (s_t, s_{t+K}), K = telawm.lam_timescale, with continuous (Gaussian,
    reparameterised) latent actions z_t.

      - Encoder (inverse dynamics):  q(z_t | s_t, s_{t+K})
      - Decoder (forward dynamics):  D(s_{t+K} | s_t, z_t), used as the planning model
      - Prior over z: N(0, I), or a learned p(z_t | s_t) (learned_prior), used for the KL term during training and
        to sample latent actions at planning time.
    """

    def __init__(self, cfg, device=None):
        super().__init__()
        self.cfg = cfg
        self.device = device

        #### Hyperparameters ####
        self.beta_kl = cfg.telawm["beta_kl"]
        self.learned_prior = cfg.telawm["learned_prior"]
        h_dim = cfg.telawm["h_dim"]

        #### Dimensions ####
        self.obs_dim = cfg.obs_shape[-1]
        self.latent_action_dim = cfg.telawm["latent_dim"]

        #### Encoder ####
        self.encoder = nn.Sequential(
            nn.Linear(2 * self.obs_dim, h_dim, device=device),
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
            nn.Linear(h_dim, 2 * self.latent_action_dim, device=device),
        )

        #### Decoder ####
        self.decoder = nn.Sequential(
            nn.Linear(self.obs_dim + self.latent_action_dim, h_dim, device=device),
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
            nn.Linear(h_dim, self.obs_dim, device=device),
        )

        #### Prior ####
        if self.learned_prior:
            self.prior_policy = nn.Sequential(
                nn.Linear(self.obs_dim, h_dim, device=device),
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
                nn.Linear(h_dim, 2 * self.latent_action_dim, device=device),
            )

    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor, deterministic: bool = False) -> Dict[str, Any]:
        """
        Encode the transition, sample z (or take its mean if deterministic) and decode the next state.

        Args:
            obs: s_t, [B, obs_dim]
            next_obs: s_{t+K}, [B, obs_dim]
        Returns:
            dict with the inputs, the posterior mu / logvar, the latent z, the predicted next_obs_pred and, with a
            learned prior, its p_mu / p_logvar.
        """
        enc = self.inverse_dynamics(obs, next_obs)
        pr = self.policy(obs)

        mu, logvar = enc["mu"], enc["logvar"]
        z = mu if deterministic else self._reparameterize(mu, logvar)
        next_obs_pred = self.forward_dynamics(obs, z)
        return {
            "obs": obs,
            "next_obs": next_obs,
            "next_obs_pred": next_obs_pred,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            **pr,
        }

    def losses(self, obs: torch.Tensor, next_obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Training loss: MSE reconstruction of s_{t+K} + beta_kl * KL(q(z | s_t, s_{t+K}) || prior), plus diagnostics."""
        out = self.forward(obs, next_obs, deterministic=False)
        recon = F.mse_loss(out["next_obs_pred"], out["next_obs"])

        mu, logvar = out["mu"], out["logvar"]
        if self.learned_prior:
            p_mu, p_logvar = out["p_mu"], out["p_logvar"]
            # KL( q || p ) for diagonal Gaussians
            kl = 0.5 * (
                (torch.exp(logvar) + (mu - p_mu).pow(2)) / torch.exp(p_logvar)
                + (p_logvar - logvar) - 1.0
            ).sum(dim=-1).mean()
        else:
            # KL( q || N(0,1) )
            kl = 0.5 * (
                torch.exp(logvar) + mu.pow(2)
                 - logvar - 1.0
            ).sum(dim=-1).mean()

        total = recon + self.beta_kl * kl

        return {
            "lam_loss": total,
            "lam_recon": recon,
            "lam_kl": kl,
            "lam_mu_mean": out["mu"].mean(),
            "lam_var_mean": torch.exp(out["logvar"]).mean(),
            "lam_p_mu_mean": out["p_mu"].mean() if self.learned_prior else None,
            "lam_p_var_mean": torch.exp(out["p_logvar"]).mean() if self.learned_prior else None,
        }

    def inverse_dynamics(self, obs: torch.Tensor, next_obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Posterior q(z | s_t, s_{t+K}): mean and (clamped) log-variance."""
        h = self.encoder(torch.cat([obs, next_obs], dim=-1))
        mu, logvar_raw = torch.chunk(h, 2, dim=-1)
        logvar = logvar_raw.clamp(min=-20.0, max=10.0)
        return {"mu": mu, "logvar": logvar}

    def forward_dynamics(self, obs: torch.Tensor, latent_action: torch.Tensor) -> torch.Tensor:
        """D(s_{t+K} | s_t, z_t): the predicted next state."""
        return self.decoder(torch.cat([obs, latent_action], dim=-1))

    def policy(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Learned prior p(z | s_t): mean and (clamped) log-variance; empty without learned_prior."""
        if not self.learned_prior:
            return {}
        h = self.prior_policy(obs)
        mu, logvar_raw = torch.chunk(h, 2, dim=-1)
        logvar = logvar_raw.clamp(min=-20.0, max=10.0)
        return {"p_mu": mu, "p_logvar": logvar}

    @torch.no_grad()
    def sample_policy(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        *,
        return_params: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a latent action from the prior (learned p(z | s_t), or N(0, I)) for planning in latent-action space.

        Args:
            obs: s_t, shape [B, obs_dim]
            deterministic: if True, use mode/mean instead of sampling
            return_params: if True, also return the prior's parameters (mu / logvar)

        Returns:
            {"z": [B, latent_dim]}, plus "p_mu" / "p_logvar" if return_params
        """
        if self.learned_prior:
            pr = self.policy(obs)
            mu, logvar = pr["p_mu"], pr["p_logvar"]
            z = mu if deterministic else self._reparameterize(mu, logvar)

        else:
            # Assumed non-learned prior: z ~ N(0, I)
            if deterministic:
                z = torch.zeros(obs.shape[0], self.latent_action_dim, device=obs.device, dtype=torch.float32)
            else:
                z = torch.randn(obs.shape[0], self.latent_action_dim, device=obs.device, dtype=torch.float32)
            mu, logvar = torch.zeros_like(z), torch.zeros_like(z)

        out = {"z": z}
        if return_params:
            out.update({"p_mu": mu, "p_logvar": logvar})
        return out

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
