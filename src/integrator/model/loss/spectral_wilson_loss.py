import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Distribution, Gamma, Poisson

from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)
from integrator.model.loss.kl_helpers import (
    compute_bg_kl,
    compute_profile_kl,
    compute_zi_intensity_kl,
)
from integrator.model.loss.learned_spectrum import ChebyshevSpectrum
from integrator.model.loss.wilson_loss import WilsonLoss


class SpectralWilsonLoss(WilsonLoss):
    """ELBO loss with continuous learned spectrum G(λ).

    Uses a Chebyshev polynomial for log G(λ) and a point-estimate B factor.
    No variational inference over G or B — both are learned directly as
    parameters, since the dataset is large enough to determine them.
    """

    def __init__(
        self,
        *,
        degree: int = 4,
        lambda_min: float = 0.9,
        lambda_max: float = 1.1,
        spectrum_init_from: str | None = None,
        b_min: float = 1.0,
        k_prior: float = 1.0,
        pi0: float = 0.7,
        init_from_tau: bool = False,
        tau_per_group=None,
        s_squared_per_group=None,
        **kwargs,
    ):
        parent_params = set(
            inspect.signature(WilsonLoss.__init__).parameters
        ) - {"self"}
        parent_kwargs = {k: v for k, v in kwargs.items() if k in parent_params}
        super().__init__(
            init_from_tau=init_from_tau,
            tau_per_group=tau_per_group,
            s_squared_per_group=s_squared_per_group,
            **parent_kwargs,
        )

        self.b_min = b_min
        self.k_prior = k_prior
        self.pi0 = pi0

        # Replace parent's variational K params with the Chebyshev spectrum
        del self.q_log_K_loc
        del self.q_log_K_log_scale

        self.spectrum = ChebyshevSpectrum(
            degree=degree,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            init_from=spectrum_init_from,
        )

        # Replace parent's variational B with a point estimate
        del self.q_log_B_loc
        del self.q_log_B_log_scale

        self.raw_B = nn.Parameter(torch.tensor(3.0))

    def get_B(self) -> Tensor:
        return F.softplus(self.raw_B) + self.b_min

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

        # Profile KL
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

        # Wilson intensity KL — point estimates for G and B
        metadata = kwargs.get("metadata")
        if (
            metadata is None
            or "d" not in metadata
            or "wavelength" not in metadata
        ):
            raise ValueError(
                "SpectralWilsonLoss requires metadata['d'] and metadata['wavelength']."
            )
        d = metadata["d"].to(device)
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))
        wavelength = metadata["wavelength"].to(device)

        # Getting the K/G factors for each reflection
        log_G = self.spectrum.get_log_G(wavelength)
        G = torch.exp(log_G)
        B = self.get_B()

        tau = (1.0 / G) * torch.exp(2.0 * B * s_sq)

        if self.learn_concentration:
            alpha_i = F.softplus(self.log_alpha_per_group[groups])
            p_i = Gamma(concentration=alpha_i, rate=alpha_i * tau)
        else:
            k_prior = self.k_prior
            p_i = Gamma(
                concentration=torch.full_like(tau, k_prior),
                rate=k_prior * tau,
            )

        kl_i = compute_zi_intensity_kl(qi, p_i, self.pi0, self.mc_samples, eps=self.eps)
        kl_i = kl_i * self.pi_weight
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
        ll = Poisson(rate.clamp(min=1e-12)).log_prob(counts.unsqueeze(1))
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
