import inspect

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Distribution, Gamma, Poisson, kl_divergence

from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)
from integrator.model.loss.kl_helpers import (
    _kl,
    compute_bg_kl,
    compute_profile_kl,
)
from integrator.model.loss.learned_spectrum import LearnedSpectrum
from integrator.model.loss.wilson_loss import WilsonLoss


class SpectralWilsonLoss(WilsonLoss):
    """ELBO loss with continuous learned spectrum G(λ) instead of per-bin G_k.

    Uses a Gaussian RBF basis expansion for log G(λ), replacing the discrete
    wavelength binning in PolyWilsonLoss.
    """

    def __init__(
        self,
        *,
        n_basis: int = 16,
        lambda_min: float = 0.9,
        lambda_max: float = 1.1,
        overlap_factor: float = 1.5,
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

        init_log_K = float(self.q_log_K_loc.detach())

        # Replace parent's scalar K params with the continuous spectrum
        del self.q_log_K_loc
        del self.q_log_K_log_scale

        self.spectrum = LearnedSpectrum(
            n_basis=n_basis,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            overlap_factor=overlap_factor,
            init_log_K=init_log_K,
            hp_loc=float(self.hp_log_K_loc),
            hp_scale=float(self.hp_log_K_scale),
        )

    def kl_hyperparams(self) -> Tensor:
        """KL on spectrum coefficients (G) + scalar B."""
        kl_K = self.spectrum.kl()
        kl_B = kl_divergence(self.q_log_B(), self.p_log_B())
        return kl_K + kl_B

    def posterior_means(self) -> dict[str, float]:
        s_B = F.softplus(self.q_log_B_log_scale)
        log_G_at_centers = self.spectrum.mean_log_G(self.spectrum.centers)
        G_means = log_G_at_centers.exp().detach()
        out = {
            "K_mean": G_means.mean().item(),
            "K_min": G_means.min().item(),
            "K_max": G_means.max().item(),
            "B_mean": (self.q_log_B_loc + 0.5 * s_B**2).exp().item(),
            "B_std": self._lognormal_std(self.q_log_B_loc, s_B),
        }
        if self.learn_concentration:
            alphas = F.softplus(self.log_alpha_per_group).detach()
            out["alpha_mean"] = alphas.mean().item()
            out["alpha_min"] = alphas.min().item()
            out["alpha_max"] = alphas.max().item()
        return out

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
            None,
            None,
            self.pprf_weight,
            device,
            metadata=kwargs.get("metadata"),
        )
        kl = kl + kl_prf

        # Wilson intensity KL with continuous spectrum
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

        if self.learn_concentration:
            alpha_i = F.softplus(self.log_alpha_per_group[groups])
        else:
            alpha_i = None

        for _ in range(self.n_wilson_samples):
            log_K_per_refl = self.spectrum.sample_log_G(wavelength)
            log_B = self.q_log_B().rsample()
            K_per_refl = torch.exp(log_K_per_refl)
            B = torch.exp(log_B).clamp(min=self.b_min) if self.b_min > 0 else torch.exp(log_B)
            tau = (1.0 / K_per_refl) * torch.exp(2.0 * B * s_sq)
            if alpha_i is not None:
                p_i = Gamma(concentration=alpha_i, rate=alpha_i * tau)
            else:
                p_i = Gamma(
                    concentration=torch.ones_like(tau),
                    rate=tau,
                )
            kl_i = kl_i + _kl(qi, p_i, self.mc_samples, eps=self.eps)
        kl_i = kl_i / self.n_wilson_samples
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

        # Hyperprior KL
        kl_hyper = self.kl_hyperparams() / self.dataset_size

        # Poisson NLL
        ll = Poisson(rate.clamp(min=1e-12)).log_prob(counts.unsqueeze(1))
        ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
        neg_ll = (-ll_mean).sum(1)

        loss = (neg_ll + kl).mean() + kl_hyper

        return {
            "loss": loss,
            "neg_ll_mean": neg_ll.mean(),
            "kl_mean": kl.mean(),
            "kl_prf_mean": kl_prf.mean(),
            "kl_i_mean": kl_i.mean(),
            "kl_bg_mean": kl_bg.mean(),
            "kl_hyper": kl_hyper,
        }
