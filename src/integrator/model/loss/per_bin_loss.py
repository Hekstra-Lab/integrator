import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution, Gamma, Poisson

from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)
from integrator.model.loss.kl_helpers import (
    _kl,
    compute_bg_kl,
    compute_profile_kl,
)


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

    Args:
        mc_samples: Monte Carlo samples for KL estimation.
        eps: Numerical stability constant for Poisson rate.
        tau_per_group: Exponential rates for intensity prior, one per group.
        bg_rate_per_group: Exponential rates for background prior, one per group.
        pi_cfg: Intensity prior config. `pi_cfg.name` selects the distribution:
            "exponential" (default) or "gamma" (per-bin MLE alpha).
            `pi_cfg.weight` scales the intensity KL term.
        pbg_cfg: Background prior config. `pbg_cfg.name` selects the distribution:
            "exponential" (default) or "gamma" (per-bin MLE alpha).
            `pbg_cfg.weight` scales the background KL term.
        pprf_cfg: Profile prior config. `pprf_cfg.weight` scales the profile KL term.
        i_concentration_per_group: Per-bin Gamma shape (alpha) for intensity.
            Auto-loaded when `pi_cfg.name = "gamma"`.
        bg_concentration_per_group: Per-bin Gamma shape (alpha) for background.
            Auto-loaded when `pbg_cfg.name = "gamma"`.
        bg_concentration: Scalar fallback shape for background Gamma prior.
            Only used when `bg_concentration_per_group` is not provided.
            Default 1.0 = Exp.
        pprf_weight, pbg_weight, pi_weight: Scalar fallback weights when
            *_cfg is not provided.
    """

    def __init__(
        self,
        *,
        mc_samples: int = 4,
        eps: float = 1e-6,
        tau_per_group: list[float] | str,
        bg_rate_per_group: list[float] | str,
        i_concentration_per_group: list[float] | str | None = None,
        bg_concentration_per_group: list[float] | str | None = None,
        bg_concentration: float = 1.0,
        # Profile prior from basis file
        profile_basis_per_bin: str | None = None,
        profile_sigma_prior: float = 3.0,
        # Prior configs (from YAML pi_cfg / pbg_cfg / pprf_cfg)
        pi_cfg=None,
        pbg_cfg=None,
        pprf_cfg=None,
        # Scalar fallback weights
        pprf_weight: float = 1.0,
        pbg_weight: float = 1.0,
        pi_weight: float = 1.0,
    ):
        super().__init__()
        self.mc_samples = mc_samples
        self.eps = eps
        self.bg_concentration = bg_concentration
        self.profile_sigma_prior = profile_sigma_prior

        # Resolve weights
        self.pprf_weight = (
            pprf_cfg.weight if pprf_cfg is not None else pprf_weight
        )
        self.pbg_weight = pbg_cfg.weight if pbg_cfg is not None else pbg_weight
        self.pi_weight = pi_cfg.weight if pi_cfg is not None else pi_weight

        # Profile prior params from basis file
        if profile_basis_per_bin is not None:
            basis = torch.load(profile_basis_per_bin, weights_only=False)
            self.profile_sigma_prior = float(
                basis.get("sigma_prior", profile_sigma_prior)
            )
            if "mu_per_group" in basis:
                self.register_buffer("profile_mu_prior", basis["mu_per_group"])
                self.register_buffer(
                    "profile_std_prior", basis["std_per_group"]
                )
            else:
                self.profile_mu_prior = None
                self.profile_std_prior = None
        else:
            self.profile_mu_prior = None
            self.profile_std_prior = None

        # Fixed buffers (per-bin)
        self.register_buffer("tau_per_group", _load_buffer(tau_per_group))
        self.register_buffer(
            "bg_rate_per_group", _load_buffer(bg_rate_per_group)
        )

        #  Intensity concentration (Gamma MLE alpha per bin)
        if i_concentration_per_group is not None:
            self.register_buffer(
                "i_concentration_per_group",
                _load_buffer(i_concentration_per_group).clamp(min=0.1),
            )
        else:
            self.i_concentration_per_group = None

        # Background concentration (Gamma MLE alpha per bin)
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
        qp: Distribution | ProfileSurrogateOutput,
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

        # Profile KL
        kl_prf = compute_profile_kl(
            qp,
            groups,
            self.profile_sigma_prior,
            self.profile_mu_prior,
            self.profile_std_prior,
            self.pprf_weight,
            device,
            metadata=kwargs.get("metadata"),
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

        # Background KL
        kl_bg = compute_bg_kl(
            qbg,
            groups,
            self.bg_rate_per_group,
            self.bg_concentration_per_group,
            self.bg_concentration,
            self.pbg_weight,
            self.mc_samples,
            self.eps,
        )
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
