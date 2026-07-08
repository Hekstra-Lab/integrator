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

    #to check how different one probability distribution is from another.
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

    Why this helper exists:
        PyTorch already knows exact KL formulas for some distribution pairs,
        such as Gamma vs Gamma. That is why the original Gamma qi setup worked.

        But for the new MFX tests, qi may be LogNormal or FoldedNormal while
        the Wilson intensity prior p_i is still Gamma. PyTorch does not have a
        built-in exact formula for KL(LogNormal || Gamma), so the direct call
        kl_divergence(q, p) raises NotImplementedError.

    What the fallback does:
        It estimates the same quantity by sampling values x from q and averaging

            log q(x) - log p(x)

        This is the Monte Carlo estimate of KL(q || p).

    Plain-English meaning:
        We sample possible intensity values from the model's qi distribution.
        Then we ask: are those same values also likely under the Wilson prior?

        If yes, the difference is small.
        If no, the difference is large.
    """
    try:
        # Fast/default path: use PyTorch's exact analytic formula when it exists.
        # Example: Gamma qi vs Gamma Wilson prior.
        return kl_divergence(q, p)
    except NotImplementedError:
        # Fallback path: PyTorch does not know this distribution pair.
        # Example: LogNormal qi vs Gamma Wilson prior.

        # Draw n_samples possible values from q.
        # rsample keeps gradients through the random sample when supported.
        # This is useful because qi's parameters are learned by the neural net.
        if getattr(q, "has_rsample", False):
            x = q.rsample((n_samples,))
        else:
            x = q.sample((n_samples,))

        # The Wilson intensity prior p is Gamma, which only supports positive
        # values. Clamp protects against zero/negative values before log_prob.
        x = x.clamp_min(eps)

        # How likely are these sampled intensities under the model posterior q?
        log_q = q.log_prob(x)

        # How likely are the same sampled intensities under the Wilson prior p?
        log_p = p.log_prob(x)

        # Average over the Monte Carlo sample dimension.
        # Result shape should match the per-reflection KL shape expected below.
        return (log_q - log_p).mean(dim=0)


class ObservationLikelihood(nn.Module):
    """
    Pixel observation likelihood helper.

    Purpose:
        Keep Luis's original Poisson pixel likelihood as the default,
        but add optional Normal and Student-t likelihoods for MFX/Jungfrau data.

    Why MFX needs this:
        MFX/Jungfrau counts.npy can contain floating-point values and negative
        values. Poisson is designed for nonnegative count-like values, so it is
        not a good match for those MFX pixels.

    Options:
        poisson:
            Original behavior. Use this for old/default workflows.

        normal:
            Continuous likelihood. Handles float and negative values.

        student_t:
            Continuous likelihood like Normal, but more forgiving of outliers.
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

        # Normal and Student-t need a positive noise scale.
        # We make this trainable so the model can learn the detector noise size.
        #
        # For Poisson, we do NOT create this parameter, so Luis's original
        # Poisson behavior stays as unchanged as possible.
        if self.name in {"normal", "student_t"}:
            init_scale = max(float(init_scale), self.eps)

            # raw_scale can be any real number.
            # softplus(raw_scale) is always positive.
            #
            # This inverse-softplus initialization makes the starting scale
            # close to init_scale.
            raw_init = math.log(math.expm1(init_scale))
            self.raw_scale = nn.Parameter(
                torch.tensor(raw_init, dtype=torch.float32)
            )

    def get_scale(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """
        Return positive scale for Normal/Student-t.

        scale = allowed noise/spread around the predicted pixel value.
        """
        if not hasattr(self, "raw_scale"):
            raise RuntimeError(
                "Observation scale is only defined for normal/student_t."
            )

        return F.softplus(self.raw_scale).to(device=device, dtype=dtype) + self.eps

    def forward(self, rate: Tensor, counts: Tensor) -> Tensor:
        """
        Compute pixel log probability.

        Args:
            rate:
                Shape: (batch, mc_samples, pixels)

                For Poisson:
                    rate is the Poisson rate.

                For Normal/Student-t:
                    rate is used as the predicted pixel center/location.

            counts:
                Shape: (batch, pixels)

                Observed detector pixel values.

        Returns:
            Tensor with shape: (batch, mc_samples, pixels)
        """
        y = counts.unsqueeze(1)

        if self.name == "poisson":
            # Original Luis behavior.
            return Poisson(rate.clamp(min=1e-12)).log_prob(y)

        if self.name == "normal":
            # MFX option: handles float and negative pixel values.
            scale = self.get_scale(device=rate.device, dtype=rate.dtype)
            return Normal(loc=rate, scale=scale).log_prob(y)

        if self.name == "student_t":
            # MFX option: handles float/negative values and is more robust
            # to very bright/outlier pixels.
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
    is computed: scalar G for monochromatic, G(lambda) for polychromatic.
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
        # Resolution bins for per-bin background prior
        n_bins: int = 1,
        # Prior configs from yaml
        pi_cfg=None,
        pbg_cfg=None,
        pprf_cfg=None,
        # KL weights
        profile_kl_weight: float = 1.0,
        background_kl_weight: float = 1.0,
        intensity_kl_weight: float = 1.0,
        # MFX/new observation likelihood options.
        #
        # Default is "poisson", preserving Luis's original logic.
        # For MFX, use "normal" or "student_t" from YAML.
        observation_likelihood: str = "poisson",
        init_obs_scale: float = 1.0,
        student_t_df: float = 4.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.eps = eps
        self.b_min = b_min
        self.n_bins = n_bins

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

        # Point-estimate B factor.
        # Used by polychromatic and monochromatic loss classes.
        self.raw_B = nn.Parameter(torch.tensor(float(init_log_B)))

        # Observation likelihood helper.
        #
        # If observation_likelihood="poisson":
        #     same behavior as original Poisson(rate).log_prob(counts)
        #
        # If observation_likelihood="normal" or "student_t":
        #     MFX-friendly continuous likelihood for float/negative pixels.
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
        self, metadata: dict, s_sq: Tensor, device: torch.device
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
            qp, "prior_scale", _DEFAULT_PROFILE_PRIOR_SCALE
        )
        kl_prf = compute_profile_kl(
            qp, prf_prior_scale, self.profile_kl_weight, device
        )
        kl = kl + kl_prf

        # Wilson intensity KL.
        d = metadata["d"].to(device)
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))

        tau = self._get_tau(metadata, s_sq, device)

        p_i = Gamma(concentration=torch.ones_like(tau), rate=tau)

        # Compare the learned intensity posterior qi against the Wilson Gamma prior.
        #
        # Original Gamma setup:
        #     qi = Gamma, p_i = Gamma
        #     PyTorch has an exact KL formula, so this behaves like kl_divergence.
        #
        # New LogNormal/FoldedNormal qi tests:
        #     qi = LogNormal or FoldedNormal, p_i = Gamma
        #     PyTorch may not have an exact KL formula, so the helper estimates
        #     KL(qi || p_i) using samples instead of crashing.

        # we are comparing: qi = model's predicted intensity posterior with p_i = Wilson Gamma prior
        #if Luis runs orginal YAML with qi of gamma, then this helper should behave like the original code
        #kl_divergence(qi, p_i)

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

        p_bg = Gamma(concentration=bg_conc, rate=bg_rate)
        kl_bg = kl_divergence(qbg, p_bg) * self.background_kl_weight
        kl = kl + kl_bg

        # Pixel negative log likelihood.
        #
        # Original behavior:
        #     ll = Poisson(rate.clamp(min=1e-12)).log_prob(counts.unsqueeze(1))
        #
        # New wrapped behavior:
        #     poisson   -> same as original behavior
        #     normal    -> MFX float/negative pixel likelihood
        #     student_t -> MFX robust float/negative pixel likelihood
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