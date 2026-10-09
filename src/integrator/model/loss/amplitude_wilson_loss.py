"""Wilson loss with Normal-Normal KL for amplitude parameterization.

The structure factor amplitude F is modeled as F = |X| where X ~ N(mu, sigma^2).
The Wilson prior on F (Rayleigh for acentric) is induced by X ~ N(0, sigma_w^2),
giving a closed-form Normal-Normal KL:

    KL = 0.5 * (sigma^2/sigma_w^2 + mu^2/sigma_w^2 - 1 - log(sigma^2/sigma_w^2))

where sigma_w^2 = 1/(2*tau) and tau = (1/G) * exp(2*B*s^2) is the Wilson rate.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import (
    Distribution,
    Gamma,
    NegativeBinomial,
    Poisson,
    kl_divergence,
)

from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)
from integrator.model.loss.kl_helpers import compute_profile_kl
from integrator.model.loss.monochromatic_wilson_loss import (
    MonochromaticWilsonLoss,
)


class AmplitudeWilsonLoss(MonochromaticWilsonLoss):
    """Wilson loss for the amplitude (Normal/FoldedNormal) parameterization.

    Replaces the Gamma-Gamma intensity KL with a closed-form Normal-Normal
    KL in signed-amplitude space.  Profile KL, background KL, and Poisson
    NLL are inherited unchanged.
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
        batch_size = rate.shape[0]
        counts = counts.to(device)
        mask = mask.to(device)

        kl = torch.zeros(batch_size, device=device)

        metadata = kwargs.get("metadata")
        if metadata is None or "d" not in metadata:
            raise ValueError("AmplitudeWilsonLoss requires metadata['d'].")

        # Profile KL (inherited)
        if self.profile_prior is not None:
            x_px = metadata["xyzcal.px.0"].to(device)
            y_px = metadata["xyzcal.px.1"].to(device)
            prf_prior_scale = self.profile_prior(x_px, y_px)
        else:
            prf_prior_scale = self.profile_prior_scale
        kl_prf = compute_profile_kl(
            qp, prf_prior_scale, self.pprf_weight, device
        )
        kl = kl + kl_prf

        # Amplitude KL: KL(N(mu, sigma^2) || N(0, sigma_w^2))
        d = metadata["d"].to(device)
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))
        tau = self._get_tau(metadata, s_sq, device)

        f_mu = metadata["f_mu"].to(device)
        f_sigma = metadata["f_sigma"].to(device)

        sigma_w_sq = 1.0 / (2.0 * tau.clamp(min=1e-12))
        sigma_sq = f_sigma.pow(2)

        kl_i = 0.5 * (
            sigma_sq / sigma_w_sq
            + f_mu.pow(2) / sigma_w_sq
            - 1.0
            - torch.log(sigma_sq / sigma_w_sq + 1e-12)
        ) * self.pi_weight
        kl = kl + kl_i

        # Background KL (inherited)
        if self.bg_prior is not None:
            x_px = metadata["xyzcal.px.0"].to(device)
            y_px = metadata["xyzcal.px.1"].to(device)
            bg_rate, bg_alpha = self.bg_prior(x_px, y_px)
            p_bg = Gamma(concentration=bg_alpha, rate=bg_alpha * bg_rate)
        else:
            p_bg = Gamma(
                concentration=torch.tensor(
                    self.bg_concentration, device=device
                ),
                rate=torch.tensor(
                    self.bg_concentration * self.bg_rate, device=device
                ),
            )
        kl_bg = kl_divergence(qbg, p_bg) * self.pbg_weight
        kl = kl + kl_bg

        # NLL: Poisson or NegativeBinomial (inherited from WilsonLoss)
        mu = rate.clamp(min=1e-12)
        if self.raw_dispersion is not None:
            r = torch.nn.functional.softplus(self.raw_dispersion)
            probs = mu / (mu + r)
            ll = NegativeBinomial(
                total_count=r, probs=probs
            ).log_prob(counts.unsqueeze(1))
        else:
            ll = Poisson(mu).log_prob(counts.unsqueeze(1))
        ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
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
