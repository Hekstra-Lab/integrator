import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import (
    Distribution,
    Gamma,
    Normal,
    Poisson,
    kl_divergence,
)

from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)
from integrator.model.loss.kl_helpers import (
    _kl,
    compute_bg_kl,
    compute_profile_kl,
)
from integrator.model.loss.per_bin_loss import _load_buffer


class WilsonLoss(nn.Module):
    """ELBO loss with Wilson intensity prior using per-reflection s^2.

    G and B are global hyperparameters inferred via variational inference.
    Background  priors remain per-bin empirical Bayes.

    Args:
        mc_samples: Monte Carlo samples for KL estimation.
        eps: Numerical stability constant for Poisson rate.
        bg_rate_per_group: Exponential rates for background prior, one per group.
        tau_per_group: Optional empirical per-bin rates, used only to
            auto-initialize q(log G) and q(log B) via linear regression.
            Not stored as a buffer.
        s_squared_per_group: Optional per-bin s^2, used only for
            initialization with tau_per_group. Not stored as a buffer.
        init_from_tau: Whether to initialize G, B from tau_per_group.
        init_log_K: Initial mean for q(log G). Used when init_from_tau=False.
        init_log_B: Initial mean for q(log B). Used when init_from_tau=False.
        hp_log_K_loc, hp_log_K_scale: Hyperprior p(log G) ~ Normal(loc, scale).
        hp_log_B_loc, hp_log_B_scale: Hyperprior p(log B) ~ Normal(loc, scale).
        n_wilson_samples: MC samples over (G, B) for the intensity KL.
        bg_concentration: Shape parameter for the background Gamma prior.
        learn_concentration: If True, learn per-bin alpha_k for Gamma intensity
            prior. Auto-set True when pi_cfg.name = "gamma".
        init_alpha: Initial value for alpha_k when learn_concentration=True.
        dataset_size: Total dataset size for amortizing the hyperprior KL.
    """

    def __init__(
        self,
        *,
        mc_samples: int = 4,
        eps: float = 1e-6,
        # Empirical Bayes priors (per-bin)
        bg_rate_per_group: list[float] | str,
        bg_concentration: float = 1.0,
        # Global Normal prior on the latent h (used by ProfileSurrogateOutput qp)
        profile_sigma_prior: float = 3.0,
        # Wilson initialization; optional
        tau_per_group: list[float] | str | None = None,
        s_squared_per_group: list[float] | str | None = None,
        init_from_tau: bool = True,
        # Wilson hyperprior config
        init_log_K: float = 0.0,
        init_log_B: float = 3.4,
        hp_log_K_loc: float = 0.0,
        hp_log_K_scale: float = 3.0,
        hp_log_B_loc: float = 3.4,
        hp_log_B_scale: float = 1.0,
        n_wilson_samples: int = 4,
        # Per-bin learnable concentration
        learn_concentration: bool = False,
        init_alpha: float = 1.0,
        i_concentration_per_group: list[float] | str | None = None,
        bg_concentration_per_group: list[float] | str | None = None,
        # Prior configs from yaml
        pi_cfg=None,
        pbg_cfg=None,
        pprf_cfg=None,
        # KL weights
        pprf_weight: float = 1.0,
        pbg_weight: float = 1.0,
        pi_weight: float = 1.0,
        dataset_size: int = 1,
    ):
        super().__init__()
        self.mc_samples = mc_samples
        self.eps = eps
        self.dataset_size = dataset_size
        self.bg_concentration = bg_concentration
        self.profile_sigma_prior = profile_sigma_prior
        self.pprf_weight = (
            pprf_cfg.weight if pprf_cfg is not None else pprf_weight
        )
        self.pbg_weight = pbg_cfg.weight if pbg_cfg is not None else pbg_weight
        self.pi_weight = pi_cfg.weight if pi_cfg is not None else pi_weight
        self.n_wilson_samples = n_wilson_samples

        # if pi_cfg.name == "gamma" -> learn per-bin alpha
        if (
            pi_cfg is not None
            and hasattr(pi_cfg, "name")
            and pi_cfg.name == "gamma"
        ):
            learn_concentration = True
        self.learn_concentration = learn_concentration

        # Fixed buffers (per-bin) — bg only
        self.bg_rate_per_group: Tensor
        self.register_buffer(
            "bg_rate_per_group", _load_buffer(bg_rate_per_group)
        )
        n_bins = int(self.bg_rate_per_group.shape[0])
        if bg_concentration_per_group is not None:
            self.register_buffer(
                "bg_concentration_per_group",
                _load_buffer(bg_concentration_per_group).clamp(min=0.1),
            )
        else:
            self.bg_concentration_per_group = None

        # Auto-initialize G, B from empirical tau via linear regression
        if init_from_tau and tau_per_group is not None:
            tau = _load_buffer(tau_per_group)
            if s_squared_per_group is not None:
                s_sq_init = _load_buffer(s_squared_per_group)
            else:
                raise ValueError(
                    "s_squared_per_group is required when init_from_tau=True"
                )
            init_log_K, init_log_B = self._fit_wilson_init(tau, s_sq_init)

        # Learnable variational parameters for q(log G), q(log B)
        self.q_log_K_loc = nn.Parameter(torch.tensor(float(init_log_K)))
        self.q_log_K_log_scale = nn.Parameter(torch.tensor(-2.0))
        self.q_log_B_loc = nn.Parameter(torch.tensor(float(init_log_B)))
        self.q_log_B_log_scale = nn.Parameter(torch.tensor(-2.0))

        # Per-bin learnable concentration (Gamma shape)
        if self.learn_concentration:
            if i_concentration_per_group is not None:
                # Initialize from Gamma MLE alpha values
                alpha_init = _load_buffer(i_concentration_per_group).clamp(
                    min=0.1
                )
                init_raw = torch.log(torch.expm1(alpha_init))
            else:
                init_raw = torch.full(
                    (n_bins,), math.log(math.expm1(init_alpha))
                )
            self.log_alpha_per_group = nn.Parameter(init_raw)

        # Fixed hyperprior parameters
        self.hp_log_K_loc: Tensor
        self.hp_log_K_scale: Tensor
        self.hp_log_B_loc: Tensor
        self.hp_log_B_scale: Tensor
        self.register_buffer("hp_log_K_loc", torch.tensor(hp_log_K_loc))
        self.register_buffer("hp_log_K_scale", torch.tensor(hp_log_K_scale))
        self.register_buffer("hp_log_B_loc", torch.tensor(hp_log_B_loc))
        self.register_buffer("hp_log_B_scale", torch.tensor(hp_log_B_scale))

    @staticmethod
    def _fit_wilson_init(tau: Tensor, s_sq: Tensor) -> tuple[float, float]:
        """Fit Wilson model to empirical tau via linear regression.

        log(tau_k) = -log(G) + 2B*s_k^2  ->  y = a + b*x
        where y = log(tau), x = s^2.  Then G = exp(-a), B = b/2.

        Returns (init_log_K, init_log_B) as floats.
        """
        y = torch.log(tau.clamp(min=1e-12))
        x = s_sq
        x_mean = x.mean()
        y_mean = y.mean()
        b = ((x - x_mean) * (y - y_mean)).sum() / (x - x_mean).pow(
            2
        ).sum().clamp(min=1e-12)
        a = y_mean - b * x_mean
        init_log_K = float(-a)
        B_val = float(b / 2.0)
        init_log_B = float(torch.log(torch.tensor(max(B_val, 1e-6))))
        return init_log_K, init_log_B

    #  Variational distributions
    def q_log_K(self) -> Normal:
        """Variational posterior q(log G)."""
        return Normal(self.q_log_K_loc, F.softplus(self.q_log_K_log_scale))

    def q_log_B(self) -> Normal:
        """Variational posterior q(log B)."""
        return Normal(self.q_log_B_loc, F.softplus(self.q_log_B_log_scale))

    # Hyperprior distributions
    def p_log_K(self) -> Normal:
        """Hyperprior p(log G)."""
        return Normal(self.hp_log_K_loc, self.hp_log_K_scale)

    def p_log_B(self) -> Normal:
        """Hyperprior p(log B)."""
        return Normal(self.hp_log_B_loc, self.hp_log_B_scale)

    #  KL terms
    def kl_hyperparams(self) -> Tensor:
        """KL(q(log G) || p(log G)) + KL(q(log B) || p(log B)).

        Closed-form Normal-Normal KL. Returns a scalar.
        """
        kl_K = kl_divergence(self.q_log_K(), self.p_log_K())
        kl_B = kl_divergence(self.q_log_B(), self.p_log_B())
        return kl_K + kl_B

    @staticmethod
    def compute_tau(K: Tensor, B: Tensor, s_sq: Tensor) -> Tensor:
        """Wilson prior rate: tau = (1/G) * exp(2B*s^2)."""
        return (1.0 / K) * torch.exp(2.0 * B * s_sq)

    #  Diagnostics
    def posterior_means(self) -> dict[str, float]:
        """Approximate posterior means of G and B for logging.

        Since q(log G) is Normal(mu, sigma), G = exp(log G) is LogNormal:
            E[G] = exp(mu + sigma^2/2)
        """
        s_K = F.softplus(self.q_log_K_log_scale)
        s_B = F.softplus(self.q_log_B_log_scale)
        out = {
            "K_mean": (self.q_log_K_loc + 0.5 * s_K**2).exp().item(),
            "B_mean": (self.q_log_B_loc + 0.5 * s_B**2).exp().item(),
            "K_std": self._lognormal_std(self.q_log_K_loc, s_K),
            "B_std": self._lognormal_std(self.q_log_B_loc, s_B),
        }
        if self.learn_concentration:
            alphas = F.softplus(self.log_alpha_per_group).detach()
            out["alpha_mean"] = alphas.mean().item()
            out["alpha_min"] = alphas.min().item()
            out["alpha_max"] = alphas.max().item()
        return out

    @staticmethod
    def _lognormal_std(mu: Tensor, sigma: Tensor) -> float:
        """Standard deviation of exp(Normal(mu, sigma)), i.e. LogNormal."""
        var = (torch.exp(sigma**2) - 1) * torch.exp(2 * mu + sigma**2)
        return var.sqrt().item()

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

        # Profile KL — global N(0, profile_sigma_prior²) prior on h.
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

        # Intensity KL: E_{q(G,B)}[ KL(q(I) || Gamma(alpha_k, alpha_k*tau_i)) ]
        # s^2 computed per-reflection from metadata["d"]
        metadata = kwargs.get("metadata")
        if metadata is None or "d" not in metadata:
            raise ValueError(
                "WilsonLoss requires metadata['d'] (per-reflection resolution) to compute s^2."
            )
        d = metadata["d"].to(device)
        s_sq = 1.0 / (4.0 * d.pow(2))  # (B,)

        if self.learn_concentration:
            alpha_i = F.softplus(self.log_alpha_per_group[groups])  # (B,)
        else:
            alpha_i = None

        for _ in range(self.n_wilson_samples):
            log_K = self.q_log_K().rsample()  # scalar
            log_B = self.q_log_B().rsample()  # scalar
            K = torch.exp(log_K)
            B = torch.exp(log_B)
            tau = self.compute_tau(K, B, s_sq)  # (B,)
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

        # Hyperprior KL (amortized over dataset)
        kl_hyper = self.kl_hyperparams() / self.dataset_size

        # Poisson NLL
        ll = Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))
        ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
        neg_ll = (-ll_mean).sum(1)

        # Total loss
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
