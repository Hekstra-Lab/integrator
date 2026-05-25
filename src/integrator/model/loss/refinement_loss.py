import torch
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

        # NLL: Poisson or NegativeBinomial (inherited from WilsonLoss)
        mu = rate.clamp(min=1e-12)
        if self.raw_dispersion is not None:
            import torch.nn.functional as Fn

            r = Fn.softplus(self.raw_dispersion)
            probs = (mu / (mu + r)).clamp(max=1.0 - 1e-6)
            ll = NegativeBinomial(
                total_count=r, probs=probs
            ).log_prob(counts.unsqueeze(1))
        else:
            ll = Poisson(mu).log_prob(counts.unsqueeze(1))
        ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
        neg_ll = (-ll_mean).sum(1)

        # R-free: exclude flagged HKLs from the training loss
        rfree = metadata.get("rfree_flag") if metadata is not None else None
        if rfree is not None:
            rfree = rfree.bool().to(device)
            work_mask = ~rfree
            n_work = work_mask.sum().clamp(min=1)
            n_free = rfree.sum().clamp(min=1)
            loss = ((neg_ll + kl) * work_mask).sum() / n_work
            neg_ll_free = (neg_ll * rfree).sum() / n_free
        else:
            loss = (neg_ll + kl).mean()
            neg_ll_free = torch.zeros(1, device=device)

        return {
            "loss": loss,
            "neg_ll_mean": neg_ll.mean(),
            "neg_ll_free": neg_ll_free,
            "kl_mean": kl.mean(),
            "kl_prf_mean": kl_prf.mean(),
            "kl_i_mean": kl_i,
            "kl_bg_mean": kl_bg.mean(),
        }
