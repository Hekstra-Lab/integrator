import math
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Distribution, Gamma, kl_divergence

from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)
from integrator.model.loss.count_likelihood import CountLikelihood
from integrator.model.loss.kl_helpers import compute_profile_kl

_DEFAULT_PROFILE_PRIOR_SCALE = 3.0


class WilsonLoss(nn.Module):
    """Base ELBO loss with Wilson intensity prior.

    Subclasses implement `_get_tau` to define how the Wilson prior rate
    is computed (scalar G for monochromatic, G(lambda) for polychromatic).
    """

    def __init__(
        self,
        *,
        # Background Gamma prior: scalar, or per-resolution-bin prior
        bg_rate: float | list[float] = 1.0,
        bg_concentration: float | list[float] = 1.0,
        # B factor
        init_log_B: float = 3.0,
        b_min: float = 0.0,
        # Scale stabilization (DIALS-free): log-space G + bounded B + a one-time
        # Wilson-plot scale init from raw counts. Off by default (legacy behavior,
        # checkpoint-compatible). See _maybe_init_scale / MonochromaticWilsonLoss.
        stabilize_scale: bool = False,
        b_max: float = 80.0,
        init_B: float = 30.0,
        init_scale_from_counts: bool = True,
        wilson_init_bins: int = 20,
        # Resolution bins for per-bin background prior
        n_bins: int = 1,
        # Pixel count likelihood: "poisson" (default) or "negative_binomial"
        likelihood: str = "poisson",
        nb_dispersion_init: float = 10.0,
        nb_dispersion_scope: str = "global",
        nb_dispersion_floor: float = 1e-3,
        # Prior configs from yaml
        pi_cfg=None,
        pbg_cfg=None,
        pprf_cfg=None,
        # KL weights
        profile_kl_weight: float = 1.0,
        background_kl_weight: float = 1.0,
        intensity_kl_weight: float = 1.0,
    ):
        super().__init__()
        self.b_min = b_min  # minimum B-factor
        self.b_max = b_max
        self.n_bins = n_bins
        self.stabilize_scale = stabilize_scale
        self.init_scale_from_counts = init_scale_from_counts
        self.wilson_init_bins = wilson_init_bins
        self.register_buffer(
            "bg_concentration",
            torch.as_tensor(bg_concentration, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "bg_rate",
            torch.as_tensor(bg_rate, dtype=torch.float32),
            persistent=False,
        )
        # keep the prior cfgs so run artifacts can record them
        self.pprf_cfg = pprf_cfg
        self.pbg_cfg = pbg_cfg
        self.pi_cfg = pi_cfg

        self.profile_kl_weight = (
            pprf_cfg.weight if pprf_cfg is not None else profile_kl_weight
        )
        self.background_kl_weight = pbg_cfg.weight if pbg_cfg is not None else background_kl_weight
        self.intensity_kl_weight = pi_cfg.weight if pi_cfg is not None else intensity_kl_weight

        # Point-estimate B factor
        # used by polychromatic and monochromatic loss classes
        if stabilize_scale:
            # B is sigmoid-bounded to [b_min, b_max]; init at init_B (physical,
            # data-free). raw_B is the inverse-sigmoid of that fraction.
            u = (float(init_B) - b_min) / (b_max - b_min)
            u = min(max(u, 1e-4), 1.0 - 1e-4)
            self.raw_B = nn.Parameter(torch.tensor(math.log(u / (1.0 - u))))
            # persistent so a resumed run does not re-init over trained params;
            # only registered when the feature is on, so legacy checkpoints load.
            self.register_buffer(
                "scale_initialized", torch.tensor(False), persistent=True
            )
        else:
            self.raw_B = nn.Parameter(torch.tensor(float(init_log_B)))

        self.count_likelihood = CountLikelihood(
            likelihood,
            dispersion_init=nb_dispersion_init,
            dispersion_scope=nb_dispersion_scope,
            n_bins=n_bins,
            dispersion_floor=nb_dispersion_floor,
        )

    def get_B(self) -> Tensor:
        if self.stabilize_scale:
            # bounded to [b_min, b_max]: never collapses to 0, gradient never dies
            return self.b_min + (self.b_max - self.b_min) * torch.sigmoid(
                self.raw_B
            )
        return F.softplus(self.raw_B) + self.b_min

    @abstractmethod
    def _get_tau(
        self, metadata: dict, s_sq: Tensor, device: torch.device
    ) -> Tensor:
        """Compute Wilson prior rate tau per reflection."""

    def _set_scale_from_fit(self, G0: float) -> None:
        """Set the overall scale from a data-driven estimate. Base: no-op.

        Overridden by MonochromaticWilsonLoss, whose scale is a single scalar G.
        """

    def _wilson_fit_G(self, i_hat: Tensor, s_sq: Tensor) -> float:
        """Intercept of a binned Wilson-plot fit: log(mean I per bin) vs s^2.

        Uses per-bin means (unbiased for the Wilson mean), so the intercept
        recovers G without the Euler-Mascheroni bias of a per-reflection log-mean.
        """
        valid = i_hat > 0
        i_hat, s2 = i_hat[valid], s_sq[valid]
        n_bins = self.wilson_init_bins
        if i_hat.numel() < 3 * n_bins:
            return float(i_hat.mean()) if i_hat.numel() else 1.0
        order = torch.argsort(s2)
        s2s, is_ = s2[order], i_hat[order]
        idx = torch.linspace(0, len(s2s), n_bins + 1).long()
        xb, yb = [], []
        for i in range(n_bins):
            lo, hi = int(idx[i]), int(idx[i + 1])
            if hi - lo < 3:
                continue
            xb.append(s2s[lo:hi].mean())
            yb.append(torch.log(is_[lo:hi].mean().clamp_min(1e-6)))
        x, y = torch.stack(xb), torch.stack(yb)
        xm, ym = x.mean(), y.mean()
        slope = ((x - xm) * (y - ym)).sum() / (x - xm).pow(2).sum().clamp_min(
            1e-12
        )
        return float(torch.exp(ym - slope * xm))

    def _maybe_init_scale(
        self, counts: Tensor, s_sq: Tensor, mask: Tensor
    ) -> None:
        """One-time DIALS-free scale init from raw counts (first batch only)."""
        if not (self.stabilize_scale and self.init_scale_from_counts):
            return
        if bool(self.scale_initialized):
            return
        with torch.no_grad():
            m = mask.squeeze(-1)  # (B, P)
            npix = m.sum(-1).clamp_min(1.0)
            bg_mean = (self.bg_concentration / self.bg_rate.clamp_min(1e-6))
            bg_mean = bg_mean.mean()  # scalar proxy (per-bin or scalar prior)
            # intensity proxy = summed counts above background; raw data, no DIALS
            i_hat = (counts * m).sum(-1) - bg_mean * npix
            g0 = self._wilson_fit_G(i_hat, s_sq)
            self._set_scale_from_fit(g0)
            self.scale_initialized.fill_(True)

    def forward(
        self,
        rate: Tensor,
        counts: Tensor,
        qp: Distribution | ProfileSurrogateOutput,
        qi: Distribution,
        qbg: Distribution,
        mask: Tensor,
        group_labels: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        device = rate.device
        batch_size = rate.shape[0]
        counts = counts.to(device)
        mask = mask.to(device)

        kl = torch.zeros(batch_size, device=device)

        # get metadata

        metadata = kwargs.get("metadata")
        if metadata is None or "d" not in metadata:
            raise ValueError("Wilson loss requires metadata['d'].")

        # profile kl-divergence
        prf_prior_scale = getattr(
            qp, "prior_scale", _DEFAULT_PROFILE_PRIOR_SCALE
        )
        kl_prf = compute_profile_kl(
            qp, prf_prior_scale, self.profile_kl_weight, device
        )
        kl = kl + kl_prf

        # Wilson intensity KL
        d = metadata["d"].to(device)
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))

        self._maybe_init_scale(counts, s_sq, mask)

        tau = self._get_tau(metadata, s_sq, device)

        p_i = Gamma(concentration=torch.ones_like(tau), rate=tau)

        kl_i = kl_divergence(qi, p_i) * self.intensity_kl_weight
        kl = kl + kl_i

        # background prior: shared Gamma, or per-resolution-bin
        if self.bg_concentration.ndim == 1:
            if group_labels is None:
                raise ValueError(
                    "per-bin background prior requires group_labels"
                )
            groups = group_labels.to(device).long()
            bg_conc = self.bg_concentration[groups]
            bg_rate = self.bg_rate[groups]
        else:
            bg_conc = self.bg_concentration
            bg_rate = self.bg_rate
        p_bg = Gamma(concentration=bg_conc, rate=bg_rate)
        kl_bg = kl_divergence(qbg, p_bg) * self.background_kl_weight
        kl = kl + kl_bg

        neg_ll = self.count_likelihood.neg_ll(
            rate, counts, mask, group_labels=group_labels
        )

        loss = (neg_ll + kl).mean()

        return {
            "loss": loss,
            "neg_ll_mean": neg_ll.mean(),
            "kl_mean": kl.mean(),
            "kl_prf_mean": kl_prf.mean(),
            "kl_i_mean": kl_i.mean(),
            "kl_bg_mean": kl_bg.mean(),
        }
