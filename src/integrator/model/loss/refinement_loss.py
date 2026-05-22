import torch
from torch import Tensor
from torch.distributions import Distribution, Gamma, Poisson, kl_divergence

from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)
from integrator.model.loss.kl_helpers import compute_profile_kl
from integrator.model.loss.monochromatic_wilson_loss import (
    MonochromaticWilsonLoss,
)


class RefinementLoss(MonochromaticWilsonLoss):
    """ELBO loss for end-to-end refinement.

    Inherits profile KL, background KL, and Poisson NLL from WilsonLoss.
    Skips the intensity KL entirely since F^2 is deterministic from the
    atomic model (not a variational distribution).  Returns ``kl_i_mean=0``
    so all downstream logging works unchanged.
    """

    def forward(
        self,
        rate: Tensor,
        counts: Tensor,
        qp: Distribution | ProfileSurrogateOutput,
        qi,
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

        # Profile KL
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

        # No intensity KL — F^2 is deterministic from the atomic model
        kl_i = torch.zeros(1, device=device)

        # Background KL
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

        # Poisson NLL
        ll = Poisson(rate.clamp(min=1e-12)).log_prob(counts.unsqueeze(1))
        ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
        neg_ll = (-ll_mean).sum(1)

        loss = (neg_ll + kl).mean()

        return {
            "loss": loss,
            "neg_ll_mean": neg_ll.mean(),
            "kl_mean": kl.mean(),
            "kl_prf_mean": kl_prf.mean(),
            "kl_i_mean": kl_i,
            "kl_bg_mean": kl_bg.mean(),
        }
