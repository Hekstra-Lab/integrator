import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Distribution, Gamma

from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)
from integrator.model.loss.kl_helpers import (
    compute_bg_kl,
    compute_profile_kl,
    compute_zi_intensity_kl,
)
from integrator.model.loss.spectral_wilson_loss import SpectralWilsonLoss


class MultinomialSpectralWilsonLoss(SpectralWilsonLoss):
    """SpectralWilsonLoss with conditional (multinomial) likelihood.

    Replaces the Poisson NLL with a multinomial NLL that conditions on
    the total observed count N per reflection. This eliminates the
    vanishing gradient problem at low signal-to-background ratios.

    The multinomial conditions on N = sum(counts) and models only the
    spatial distribution of counts. The gradient at I=0 is:

        dℓ/dI = Σ_j y_j (p_j/b_j - 1/B)

    which is non-zero whenever the profile differs from uniform background.
    This loses ~5% of Fisher information for peaked profiles but eliminates
    the intensity floor caused by background-dominated gradient noise.
    """

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
        counts = counts.to(device)
        mask = mask.to(device)
        groups = group_labels.long()

        kl = torch.zeros(counts.shape[0], device=device)

        # Profile KL (same as parent)
        kl_prf = compute_profile_kl(
            qp,
            groups,
            self.profile_sigma_prior,
            None,
            None,
            self.pprf_weight,
            device,
            metadata=kwargs.get("metadata"),
        )
        kl = kl + kl_prf

        # Wilson intensity KL (same as parent)
        metadata = kwargs.get("metadata")
        if (
            metadata is None
            or "d" not in metadata
            or "wavelength" not in metadata
        ):
            raise ValueError(
                "MultinomialSpectralWilsonLoss requires metadata['d'] and metadata['wavelength']."
            )
        d = metadata["d"].to(device)
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))
        wavelength = metadata["wavelength"].to(device)

        log_G = self.spectrum.get_log_G(wavelength)
        G = torch.exp(log_G)
        B = self.get_B()
        tau = (1.0 / G) * torch.exp(2.0 * B * s_sq)

        if self.learn_concentration:
            alpha_i = F.softplus(self.log_alpha_per_group[groups])
            p_i = Gamma(concentration=alpha_i, rate=alpha_i * tau)
        else:
            p_i = Gamma(
                concentration=torch.ones_like(tau),
                rate=tau,
            )
        kl_i = compute_zi_intensity_kl(qi, p_i, self.pi0, self.mc_samples, eps=self.eps)
        kl_i = kl_i * self.pi_weight
        kl = kl + kl_i

        # Background KL (same as parent)
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

        # Multinomial NLL (replaces Poisson NLL)
        # rate: (B, S, K) — per-pixel rates from MC samples
        # counts: (B, K)
        rate_clamped = rate.clamp(min=1e-12)
        mask_expanded = mask.unsqueeze(1)  # (B, 1, K)

        # Normalize rates to probabilities per reflection per MC sample
        total_rate = (rate_clamped * mask_expanded).sum(dim=-1, keepdim=True)
        log_probs = torch.log(rate_clamped / total_rate)  # (B, S, K)

        # Multinomial log-likelihood: Σ_j y_j * log(π_j)
        ll = (counts.unsqueeze(1) * log_probs) * mask_expanded  # (B, S, K)
        ll_per_sample = ll.sum(dim=-1)  # (B, S) — sum over pixels
        ll_mean = ll_per_sample.mean(dim=1)  # (B,) — mean over MC samples
        neg_ll = -ll_mean

        loss = (neg_ll + kl).mean()

        return {
            "loss": loss,
            "neg_ll_mean": neg_ll.mean(),
            "kl_mean": kl.mean(),
            "kl_prf_mean": kl_prf.mean(),
            "kl_i_mean": kl_i.mean(),
            "kl_bg_mean": kl_bg.mean(),
        }
