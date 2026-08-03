import math
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import (
    Distribution,
    Gamma,
    Normal,
    Poisson,
    StudentT,
    kl_divergence,
)

from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)
from integrator.model.loss.kl_helpers import compute_profile_kl

_DEFAULT_PROFILE_PRIOR_SCALE = 3.0


def _kl_divergence_with_mc_fallback(
    q: Distribution,
    p: Distribution,
    n_samples: int = 8,
    eps: float = 1.0e-8,
) -> Tensor:
    """Compute KL(q || p), with a sampling fallback for unsupported pairs.

    PyTorch has exact KL formulas for some distribution pairs, such as
    Gamma vs Gamma. For newer MFX tests, qi may be LogNormal or FoldedNormal
    while the Wilson intensity prior p_i is still Gamma. PyTorch does not
    always have a built-in exact KL formula for those pairs.

    This fallback estimates:

        KL(q || p) = E_q[log q(x) - log p(x)]

    by sampling x from q.
    """
    try:
        return kl_divergence(q, p)
    except NotImplementedError:
        if getattr(q, "has_rsample", False):
            x = q.rsample((n_samples,))
        else:
            x = q.sample((n_samples,))

        x = x.clamp_min(eps)

        log_q = q.log_prob(x)
        log_p = p.log_prob(x)

        return (log_q - log_p).mean(dim=0)


class ObservationLikelihood(nn.Module):
    """Pixel observation likelihood helper.

    Options:
        poisson:
            Original behavior. Use for nonnegative count-like data.

        normal:
            Continuous likelihood. Handles float and negative values.

        student_t:
            Continuous likelihood like Normal, but more robust to outliers.
    """

    def __init__(
        self,
        name: str = "poisson",
        init_scale: float = 1.0,
        student_t_df: float = 4.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        valid_names = {"poisson", "normal", "student_t"}
        if name not in valid_names:
            raise ValueError(
                f"Unknown observation_likelihood={name!r}. "
                f"Valid options are {sorted(valid_names)}."
            )

        self.name = name
        self.student_t_df = float(student_t_df)
        self.eps = eps

        if self.name in {"normal", "student_t"}:
            init_scale = max(float(init_scale), self.eps)
            raw_init = math.log(math.expm1(init_scale))
            self.raw_scale = nn.Parameter(
                torch.tensor(raw_init, dtype=torch.float32)
            )

    def get_scale(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Return positive scale for Normal/Student-t."""
        if not hasattr(self, "raw_scale"):
            raise RuntimeError(
                "Observation scale is only defined for normal/student_t."
            )

        return F.softplus(self.raw_scale).to(device=device, dtype=dtype) + self.eps

    def forward(self, rate: Tensor, counts: Tensor) -> Tensor:
        """Compute pixel log probability.

        Args:
            rate:
                Shape: (batch, mc_samples, pixels)

            counts:
                Shape: (batch, pixels)

        Returns:
            Tensor with shape: (batch, mc_samples, pixels)
        """
        y = counts.unsqueeze(1)

        if self.name == "poisson":
            return Poisson(rate.clamp(min=1e-12)).log_prob(y)

        if self.name == "normal":
            scale = self.get_scale(device=rate.device, dtype=rate.dtype)
            return Normal(loc=rate, scale=scale).log_prob(y)

        if self.name == "student_t":
            scale = self.get_scale(device=rate.device, dtype=rate.dtype)
            df = torch.tensor(
                self.student_t_df,
                device=rate.device,
                dtype=rate.dtype,
            )
            return StudentT(df=df, loc=rate, scale=scale).log_prob(y)

        raise RuntimeError(f"Unsupported likelihood: {self.name}")


class WilsonLoss(nn.Module):
    """Base ELBO loss with Wilson intensity prior.

    Subclasses implement `_get_tau` to define how the Wilson prior rate
    is computed.

    Examples:
        MonochromaticWilsonLoss:
            scalar G, optionally per-image B/G.

        PolychromaticWilsonLoss:
            wavelength-dependent G(lambda).
    """

    def __init__(
        self,
        *,
        # Background Gamma prior: scalar, or per-resolution-bin prior.
        bg_rate: float | list[float] = 1.0,
        bg_concentration: float | list[float] = 1.0,
        # B factor.
        init_log_B: float = 20.0,
        b_min: float = 0.0,
        # Resolution bins for per-bin background prior.
        n_bins: int = 1,
        # Prior configs from YAML.
        pi_cfg=None,
        pbg_cfg=None,
        pprf_cfg=None,
        # KL weights.
        profile_kl_weight: float = 1.0,
        background_kl_weight: float = 1.0,
        intensity_kl_weight: float = 1.0,
        # MFX/new observation likelihood options.
        observation_likelihood: str = "poisson",
        init_obs_scale: float = 1.0,
        student_t_df: float = 4.0,
        
        # Optional image-level Wilson options.
        image_level_wilson: bool = False,
        n_images: int | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.eps = eps
        self.b_min = b_min
        self.n_bins = n_bins

        # Store image-level Wilson settings for subclasses.
        self.image_level_wilson = image_level_wilson
        self.n_images = n_images
        self.init_log_B = init_log_B
        
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

        # Keep the prior cfgs so run artifacts can record them.
        self.pprf_cfg = pprf_cfg
        self.pbg_cfg = pbg_cfg
        self.pi_cfg = pi_cfg

        self.profile_kl_weight = (
            pprf_cfg.weight if pprf_cfg is not None else profile_kl_weight
        )
        self.background_kl_weight = (
            pbg_cfg.weight if pbg_cfg is not None else background_kl_weight
        )
        self.intensity_kl_weight = (
            pi_cfg.weight if pi_cfg is not None else intensity_kl_weight
        )

        # Global B factor.
        # MonochromaticWilsonLoss may also create per-image B embeddings.
        self.raw_B = nn.Parameter(torch.tensor(float(init_log_B)))

        self.observation_model = ObservationLikelihood(
            name=observation_likelihood,
            init_scale=init_obs_scale,
            student_t_df=student_t_df,
            eps=self.eps,
        )

    def get_B(self) -> Tensor:
        return F.softplus(self.raw_B) + self.b_min

    @abstractmethod
    def _get_tau(
        self,
        metadata: dict,
        s_sq: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Compute Wilson prior rate tau per reflection."""

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

        # Metadata is needed for Wilson prior calculation.
        metadata = kwargs.get("metadata")
        if metadata is None or "d" not in metadata:
            raise ValueError("Wilson loss requires metadata['d'].")

        # Profile KL-divergence.
        prf_prior_scale = getattr(
            qp,
            "prior_scale",
            _DEFAULT_PROFILE_PRIOR_SCALE,
        )
        kl_prf = compute_profile_kl(
            qp,
            prf_prior_scale,
            self.profile_kl_weight,
            device,
        )
        kl = kl + kl_prf

        # Wilson intensity KL.
        d = metadata["d"].to(device)
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))

        tau = self._get_tau(metadata, s_sq, device)

        p_i = Gamma(
            concentration=torch.ones_like(tau),
            rate=tau.clamp(min=self.eps),
        )

        kl_i = (
            _kl_divergence_with_mc_fallback(qi, p_i, eps=self.eps)
            * self.intensity_kl_weight
        )
        kl = kl + kl_i

        # Background prior: shared Gamma, or per-resolution-bin.
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

        p_bg = Gamma(
            concentration=bg_conc,
            rate=bg_rate,
        )
        kl_bg = kl_divergence(qbg, p_bg) * self.background_kl_weight
        kl = kl + kl_bg

        # Pixel negative log likelihood.
        ll = self.observation_model(rate, counts)

        # Average over Monte Carlo samples, then apply valid-pixel mask.
        ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)

        # Negative log likelihood per reflection.
        neg_ll = (-ll_mean).sum(1)

        loss = (neg_ll + kl).mean()

        return {
            "loss": loss,
            "neg_ll_mean": neg_ll.mean(),
            "kl_mean": kl.mean(),
            "kl_prf_mean": kl_prf.mean(),
            "kl_i_mean": kl_i.mean(),
            "kl_bg_mean": kl_bg.mean(),
        }