"""Per-bin resolution loss with group-dependent priors.

ELBO decomposition:
    L = E_q[ log p(x | I, prf, bg) ]             — Poisson NLL
      - KL( q(prf) || p(prf) )                    — profile prior
      - KL( q(I_i)   || p_I(tau_{k(i)}) )         — per-group intensity prior
      - KL( q(bg_i)  || p_bg(lambda_{k(i)}) )     — per-group background prior

Intensity and background priors are controlled by ``pi_cfg`` and ``pbg_cfg``:
  - ``pi_cfg.name = "exponential"``:  I_k ~ Exp(tau_k)
  - ``pi_cfg.name = "gamma"``:       I_k ~ Gamma(alpha_k, alpha_k * tau_k)
    where alpha_k is fitted via Gamma MLE per bin.

Profile prior depends on the surrogate type:
  - Dirichlet surrogate:  KL( q(prf_i) || Dir(alpha_{k(i)}) )  per-bin
  - Latent decoder (ProfilePosterior):  KL( q(h) || N(0, sigma²I) )  global

All prior parameters are fixed (not learned), loaded from simulation output.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Dirichlet, Distribution, Gamma, Poisson

from integrator.model.distributions.logistic_normal import ProfilePosterior
from integrator.model.loss.loss import _kl


def _load_buffer(value: list[float] | str) -> Tensor:
    """Load a tensor from a list of floats or a .pt file path."""
    if isinstance(value, str):
        loaded = torch.load(value, weights_only=True)
        if isinstance(loaded, dict):
            # support dicts with a single key
            return next(iter(loaded.values())).float()
        return loaded.float()
    return torch.tensor(value, dtype=torch.float32)


class PerBinLoss(nn.Module):
    """ELBO loss with per-group priors for intensity, background, and profile.

    Parameters
    ----------
    mc_samples : int
        Monte Carlo samples for KL estimation.
    eps : float
        Numerical stability constant for Poisson rate.
    tau_per_group : list[float] or str
        Exponential rates for intensity prior, one per group.
    bg_rate_per_group : list[float] or str
        Exponential rates for background prior, one per group.
    concentration_per_group : str
        Path to .pt file with Dirichlet concentrations (n_groups, n_pixels).
    pi_cfg : PriorConfig or None
        Intensity prior config.  ``pi_cfg.name`` selects the distribution:
        ``"exponential"`` (default) or ``"gamma"`` (per-bin MLE alpha).
        ``pi_cfg.weight`` scales the intensity KL term.
    pbg_cfg : PriorConfig or None
        Background prior config.  ``pbg_cfg.name`` selects the distribution:
        ``"exponential"`` (default) or ``"gamma"`` (per-bin MLE alpha).
        ``pbg_cfg.weight`` scales the background KL term.
    pprf_cfg : PriorConfig or None
        Profile prior config.  ``pprf_cfg.weight`` scales the profile KL term.
    i_concentration_per_group : list[float] or str or None
        Per-bin Gamma shape (alpha) for intensity.  Auto-loaded when
        ``pi_cfg.name = "gamma"``.
    bg_concentration_per_group : list[float] or str or None
        Per-bin Gamma shape (alpha) for background.  Auto-loaded when
        ``pbg_cfg.name = "gamma"``.
    bg_concentration : float
        Scalar fallback shape for background Gamma prior.  Only used when
        ``bg_concentration_per_group`` is not provided.  Default 1.0 = Exp.
    pprf_weight, pbg_weight, pi_weight : float
        Scalar fallback weights when *_cfg is not provided.
    """

    def __init__(
        self,
        *,
        mc_samples: int = 4,
        eps: float = 1e-6,
        tau_per_group: list[float] | str,
        bg_rate_per_group: list[float] | str,
        concentration_per_group: str,
        i_concentration_per_group: list[float] | str | None = None,
        bg_concentration_per_group: list[float] | str | None = None,
        bg_concentration: float = 1.0,
        # Prior configs (from YAML pi_cfg / pbg_cfg / pprf_cfg)
        pi_cfg=None,
        pbg_cfg=None,
        pprf_cfg=None,
        # Scalar fallback weights
        pprf_weight: float = 1.0,
        pbg_weight: float = 1.0,
        pi_weight: float = 1.0,
        dataset_size: int = 1,
    ):
        super().__init__()
        self.mc_samples = mc_samples
        self.eps = eps
        self.dataset_size = dataset_size
        self.bg_concentration = bg_concentration

        # Resolve weights: prefer *_cfg.weight, fall back to scalar
        self.pprf_weight = pprf_cfg.weight if pprf_cfg is not None else pprf_weight
        self.pbg_weight = pbg_cfg.weight if pbg_cfg is not None else pbg_weight
        self.pi_weight = pi_cfg.weight if pi_cfg is not None else pi_weight

        # -- Fixed buffers (per-bin) ------------------------------------------
        self.register_buffer("tau_per_group", _load_buffer(tau_per_group))
        self.register_buffer(
            "bg_rate_per_group", _load_buffer(bg_rate_per_group)
        )
        self.register_buffer(
            "concentration_per_group",
            _load_buffer(concentration_per_group).clamp(min=1e-6),
        )

        # -- Intensity concentration (Gamma MLE alpha per bin) ----------------
        if i_concentration_per_group is not None:
            self.register_buffer(
                "i_concentration_per_group",
                _load_buffer(i_concentration_per_group).clamp(min=0.1),
            )
        else:
            self.i_concentration_per_group = None

        # -- Background concentration (Gamma MLE alpha per bin) ---------------
        if bg_concentration_per_group is not None:
            self.register_buffer(
                "bg_concentration_per_group",
                _load_buffer(bg_concentration_per_group).clamp(min=0.1),
            )
        else:
            self.bg_concentration_per_group = None

    def forward(
        self,
        rate: Tensor,
        counts: Tensor,
        qp: Distribution | ProfilePosterior,
        qi: Distribution,
        qbg: Distribution,
        mask: Tensor,
        group_labels: Tensor,
        **kwargs,
    ) -> dict[str, Tensor]:
        device = rate.device
        batch_size = rate.shape[0]
        counts = counts.to(device)
        mask = mask.to(device)
        groups = group_labels.long()

        kl = torch.zeros(batch_size, device=device)
        kl_prf = torch.zeros(batch_size, device=device)
        kl_i = torch.zeros(batch_size, device=device)
        kl_bg = torch.zeros(batch_size, device=device)

        # Profile KL: per-bin Dirichlet or latent Normal (global or per-bin)
        # Use profile_group_label (2D binning) if available, else group_labels
        if isinstance(qp, ProfilePosterior):
            meta = kwargs.get("metadata", {})
            pgl = meta.get("profile_group_label") if isinstance(meta, dict) else None
            prf_groups = pgl.long().to(device) if pgl is not None else groups
            kl_prf = qp.kl_divergence(prf_groups) * self.pprf_weight
        else:
            meta = kwargs.get("metadata", {})
            pgl = meta.get("profile_group_label") if isinstance(meta, dict) else None
            prf_groups = pgl.long().to(device) if pgl is not None else groups
            alpha = self.concentration_per_group[prf_groups]  # (B, n_pixels)
            p_prf = Dirichlet(alpha)
            kl_prf = (
                _kl(qp, p_prf, self.mc_samples, eps=self.eps)
                * self.pprf_weight
            )
        kl = kl + kl_prf

        # Intensity KL: KL(q(I_i) || Gamma(alpha_k, alpha_k * tau_k))
        # When i_concentration_per_group is None, alpha_k = 1 (Exponential)
        tau_per_refl = self.tau_per_group[groups]  # (B,)
        if self.i_concentration_per_group is not None:
            alpha_i = self.i_concentration_per_group[groups]  # (B,)
        else:
            alpha_i = torch.ones_like(tau_per_refl)
        p_i = Gamma(
            concentration=alpha_i,
            rate=alpha_i * tau_per_refl,
        )
        kl_i = _kl(qi, p_i, self.mc_samples, eps=self.eps) * self.pi_weight
        kl = kl + kl_i

        # Background KL: KL(q(bg_i) || Gamma(alpha_bg, alpha_bg * lambda_k))
        bg_rate_per_refl = self.bg_rate_per_group[groups]  # (B,)
        if self.bg_concentration_per_group is not None:
            alpha_bg = self.bg_concentration_per_group[groups]  # (B,)
        else:
            alpha_bg = torch.full_like(bg_rate_per_refl, self.bg_concentration)
        p_bg = Gamma(
            concentration=alpha_bg,
            rate=alpha_bg * bg_rate_per_refl,
        )
        kl_bg = _kl(qbg, p_bg, self.mc_samples, eps=self.eps) * self.pbg_weight
        kl = kl + kl_bg

        # Poisson NLL
        ll = Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))
        ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
        neg_ll = (-ll_mean).sum(1)

        # Total loss
        loss = (neg_ll + kl).mean()

        return {
            "loss": loss,
            "neg_ll_mean": neg_ll.mean(),
            "kl_mean": kl.mean(),
            "kl_prf_mean": kl_prf.mean(),
            "kl_i_mean": kl_i.mean(),
            "kl_bg_mean": kl_bg.mean(),
        }
