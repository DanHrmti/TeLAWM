"""Latent Action-Chunk Model (LACM): a (conditional) VAE whose latent summarizes an action chunk."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentActionChunkModel(nn.Module):
    """
    Latent action-chunk model: z_t summarizes the action chunk a_{t:t+K-1} (K = wm_timescale), with continuous
    (Gaussian, reparameterised) latents.

      - Encoder:
          unconditional (state_conditioned=False):  q(z_t | a_{t:t+K-1})         (VAE)
          conditional   (state_conditioned=True):   q(z_t | a_{t:t+K-1}, s_t)    (CVAE)
      - Decoder (variant):
          dynamics:   D(s_{t+K} | s_t, z_t)  predicts the state the chunk leads to; serves as the planning model
          chunk_vae:  reconstructs the chunk itself; planning then needs a separate dynamics model trained on z
                        unconditional:  p(a_{t:t+K-1} | z_t)
                        conditional:    p(a_{t:t+K-1} | z_t, s_t)
      - Prior over z: N(0, I) (unconditional) or a learned p(z | s_t) (conditional), used for the KL term during
        training and to sample latents at planning time.
    """

    def __init__(self, cfg, device=None):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # Hyperparameters
        self.beta_kl = cfg.lacwm["beta_kl"]
        self.learned_prior = cfg.lacwm["learned_prior"]
        self.state_conditioned = cfg.lacwm["state_conditioned"]
        self.variant = cfg.lacwm["variant"]  # what the decoder reconstructs, see the class docstring
        assert self.variant in ("dynamics", "chunk_vae"), \
            f"lacwm.variant must be 'dynamics' or 'chunk_vae' (got {self.variant})"

        h_dim = cfg.lacwm["h_dim"]

        self.obs_dim = cfg.obs_shape[-1]
        self.chunk_len = cfg.lacwm["wm_timescale"] # K
        self.action_dim = cfg.action_dim
        self.chunk_flat_dim = self.chunk_len * self.action_dim # K * a_dim

        self.latent_dim = cfg.lacwm["latent_dim"]

        # Encoder input dimension:
        #   unconditional: just flattened action chunk
        #   conditional: action chunk + obs
        if self.state_conditioned:
            enc_in_dim = self.chunk_flat_dim + self.obs_dim
        else:
            enc_in_dim = self.chunk_flat_dim

        # Encoder  q(z | a_chunk [, s])
        self.encoder = nn.Sequential(
            nn.Linear(enc_in_dim, h_dim, device=device),
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
            nn.Linear(h_dim, 2 * self.latent_dim, device=device),
        )

        # Decoder
        #   dynamics:  D(s' | s, z), input [obs, z_t]
        #   chunk_vae: p(a_chunk | z [, s]), input [z_t (, obs)]
        if self.variant == "dynamics":
            dec_in_dim, dec_out_dim = self.obs_dim + self.latent_dim, self.obs_dim
        else:
            dec_in_dim = self.latent_dim + (self.obs_dim if self.state_conditioned else 0)
            dec_out_dim = self.chunk_flat_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_in_dim, h_dim, device=device),
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
            nn.Linear(h_dim, dec_out_dim, device=device),
        )

        # Learned Prior  p(z | s)  (conditional case only)
        # In the unconditional case the prior is N(0,I) by construction of the
        # VAE objective; no network is needed.
        # In the conditional case we learn p(z | s) and use it for KL
        # regularisation and planning samples.
        if self.learned_prior:
            assert self.state_conditioned, (
                "learned_prior=True requires state_conditioned=True; "
                "in the unconditional case the prior is N(0,I) by design."
            )
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
                nn.Linear(h_dim, 2 * self.latent_dim, device=device),
            )

    def forward(
        self,
        obs: torch.Tensor,
        action_chunk: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """
        Args:
            obs: s_t, shape [B, obs_dim]
            action_chunk: a_{t:t+K-1}, shape [B, K * action_dim]  (pre-flattened)
            deterministic: use mean/mode instead of sampling
        Returns:
            dict with latent parameters, sampled z, and the decoder's reconstruction (s_{t+K} or the chunk).
        """
        enc = self.encode(obs, action_chunk)
        pr  = self.prior(obs)

        mu, logvar = enc["mu"], enc["logvar"]
        z = mu if deterministic else self._reparameterize(mu, logvar)
        if self.variant == "dynamics":
            recon = self.forward_dynamics(obs, z)
        else:
            recon = self.decode_action_chunk(obs, z)
        return {
            "obs": obs,
            "action_chunk": action_chunk,
            "recon": recon,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            **pr,
        }

    def losses(
        self,
        obs: torch.Tensor,
        action_chunk: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE/CVAE training losses.

        Args:
            obs: s_t, [B, obs_dim]
            action_chunk: a_{t:t+K-1}, [B, K * action_dim]
            next_obs: s_{t+K}, [B, obs_dim]
        """
        out = self.forward(obs, action_chunk, deterministic=False)
        recon = F.mse_loss(out["recon"], next_obs if self.variant == "dynamics" else action_chunk)

        mu, logvar = out["mu"], out["logvar"]

        if self.learned_prior:
            # KL( q(z|a[,s]) || p(z|s) )  for diagonal Gaussians
            p_mu, p_logvar = out["p_mu"], out["p_logvar"]
            kl = 0.5 * (
                (torch.exp(logvar) + (mu - p_mu).pow(2)) / torch.exp(p_logvar)
                + (p_logvar - logvar) - 1.0
            ).sum(dim=-1).mean()
        else:
            # KL( q(z|a) || N(0,I) )
            kl = 0.5 * (
                torch.exp(logvar) + mu.pow(2) - logvar - 1.0
            ).sum(dim=-1).mean()

        total = recon + self.beta_kl * kl

        return {
            "lacm_loss":      total,
            "lacm_recon":     recon,
            "lacm_kl":        kl,
            "lacm_mu_mean":   mu.mean(),
            "lacm_var_mean":  torch.exp(logvar).mean(),
            "lacm_p_mu_mean":  out.get("p_mu", torch.zeros(1, device=obs.device)).mean()
                               if self.learned_prior else None,
            "lacm_p_var_mean": torch.exp(out["p_logvar"]).mean()
                               if self.learned_prior else None,
        }

    def encode(
        self, obs: torch.Tensor, action_chunk: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Encode action chunk (and optionally obs) into latent parameters."""
        if self.state_conditioned:
            h = self.encoder(torch.cat([action_chunk, obs], dim=-1))
        else:
            h = self.encoder(action_chunk)

        mu, logvar_raw = torch.chunk(h, 2, dim=-1)
        logvar = logvar_raw.clamp(min=-20.0, max=10.0)
        return {"mu": mu, "logvar": logvar}

    def decode_action_chunk(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """p(a_{t:t+K-1} | z [, s_t]) (variant=chunk_vae): [B, K * action_dim], pre-flattened."""
        return self.decoder(torch.cat([z, obs], dim=-1) if self.state_conditioned else z)

    def forward_dynamics(
        self, obs: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """D(s_{t+K} | s_t, z_t)."""
        return self.decoder(torch.cat([obs, z], dim=-1))

    def prior(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Evaluate the learned prior p(z | s).
        Returns empty dict if no learned prior (unconditional case).
        """
        if not self.learned_prior:
            return {}
        h = self.prior_policy(obs)
        mu, logvar_raw = torch.chunk(h, 2, dim=-1)
        logvar = logvar_raw.clamp(min=-20.0, max=10.0)
        return {"p_mu": mu, "p_logvar": logvar}

    @torch.no_grad()
    def sample_prior(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        *,
        return_params: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a latent z from the prior.

        Unconditional: z ~ N(0, I)
        Conditional: z ~ p(z | s) (requires learned_prior=True)

        Args:
            obs: s_t, [B, obs_dim]
            deterministic: use mean/mode instead of sampling
            return_params: also return prior distribution parameters

        Returns:
            {"z": [B, latent_dim], ...optional params...}
        """
        if self.learned_prior:
            pr = self.prior(obs)
            mu, logvar = pr["p_mu"], pr["p_logvar"]
            z = mu if deterministic else self._reparameterize(mu, logvar)
        else:
            if deterministic:
                z = torch.zeros(
                    obs.shape[0], self.latent_dim,
                    device=obs.device, dtype=torch.float32,
                )
            else:
                z = torch.randn(
                    obs.shape[0], self.latent_dim,
                    device=obs.device, dtype=torch.float32,
                )
            mu = logvar = torch.zeros_like(z)

        out = {"z": z}
        if return_params:
            out.update({"p_mu": mu, "p_logvar": logvar})
        return out

    def _reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
