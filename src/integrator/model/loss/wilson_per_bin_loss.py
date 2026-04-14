"""Per-bin loss with learned Wilson intensity prior.

Wilson model: tau_k = (1/K) * exp(2B * s_k^2), where K and B are
global hyperparameters inferred via variational inference.

Intensity prior: I_k ~ Gamma(alpha_k, alpha_k * tau_k)
  alpha_k = 1 gives Exponential; learnable alpha_k when pi_cfg.name = "gamma".
Background prior: bg_k ~ Gamma(alpha_bg, alpha_bg * lambda_k) (fixed, per-bin).
Profile prior: Dirichlet (per-bin) or latent Normal (global).
Hyperprior: KL(q(log K) || p(log K)) + KL(q(log B) || p(log B)), amortized over N.
"""

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

from integrator.model.distributions.logistic_normal import ProfilePosterior
from integrator.model.loss.kl_helpers import compute_bg_kl, compute_profile_kl
from integrator.model.loss.loss import _kl
from integrator.model.loss.per_bin_loss import _load_buffer


class WilsonPerBinLoss(nn.Module):
    """ELBO loss with learned Wilson intensity prior and empirical Bayes bg/profile.

    The Wilson distribution models expected intensity in bin k as:
        Sigma_k = K * exp(-2B * s_k^2)
    where s_k^2 = 1/(4d_k^2) is precomputed per bin. The prior rate for
    bin k is tau_k = 1/Sigma_k = (1/K) * exp(2B * s_k^2).

    K and B are two global hyperparameters inferred via variational inference.
    Background and profile priors remain per-bin empirical Bayes.

    Args:
        mc_samples: Monte Carlo samples for KL estimation.
        eps: Numerical stability constant for Poisson rate.
        s_squared_per_group: 1/(4d^2) per resolution bin.
        bg_rate_per_group: Exponential rates for background prior, one per group.
        concentration_per_group: Path to .pt file with Dirichlet concentrations
            (n_groups, n_pixels). Only used when the profile surrogate returns a
            Distribution (Dirichlet). Ignored when the surrogate returns a
            ProfilePosterior (latent decoder), which uses a global
            N(0, sigma_prior^2 I) prior instead.
        tau_per_group: Empirical per-bin rates. Used to auto-initialize q(log K)
            and q(log B) via linear regression of log(tau) on s^2 when
            `init_from_tau=True`.
        init_from_tau: Whether to initialize K, B from `tau_per_group` via
            linear regression. When False, uses `init_log_K` / `init_log_B`
            directly. Default True.
        init_log_K: Initial mean for q(log K). Used when init_from_tau=False.
        init_log_B: Initial mean for q(log B). Used when init_from_tau=False.
        hp_log_K_loc, hp_log_K_scale: Hyperprior p(log K) ~ Normal(loc, scale).
        hp_log_B_loc, hp_log_B_scale: Hyperprior p(log B) ~ Normal(loc, scale).
        n_wilson_samples: MC samples over (K, B) for the intensity KL outer
            expectation.
        bg_concentration: Shape parameter for the background Gamma prior.
            Default 1.0 gives Exp(lambda_k). Higher values give a tighter prior
            around mean = 1/lambda_k, with CV = 1/sqrt(bg_concentration).
        learn_concentration: If True, learn per-bin alpha_k for Gamma intensity
            prior.
        Auto-set True when pi_cfg.name = "gamma". Default False (Exponential).
    init_alpha : float
        Initial value for alpha_k when learn_concentration=True.
    pi_cfg : PriorConfig or None
        Intensity prior config. name selects "exponential" or "gamma".
    pbg_cfg : PriorConfig or None
        Background prior config. name selects "exponential" or "gamma".
    pprf_cfg : PriorConfig or None
        Profile prior config. `pprf_cfg.weight` scales the profile KL term.
    pprf_weight, pbg_weight, pi_weight : float
        Scalar fallback weights when *_cfg is not provided.
    dataset_size : int
        Total dataset size for amortizing the hyperprior KL.
    """

    def __init__(
        self,
        *,
        mc_samples: int = 4,
        eps: float = 1e-6,
        # Resolution data
        s_squared_per_group: list[float] | str,
        # Empirical Bayes priors (per-bin)
        bg_rate_per_group: list[float] | str,
        concentration_per_group: str | float | None = None,
        pprf_n_pixels: int | None = None,
        pprf_quantile: float | None = None,
        pprf_conc_factor: float = 40.0,
        bg_concentration: float = 1.0,
        # Wilson initialization
        tau_per_group: list[float] | str | None = None,
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
        # Prior configs (from YAML pi_cfg / pbg_cfg / pprf_cfg)
        pi_cfg=None,
        pbg_cfg=None,
        pprf_cfg=None,
        # Scalar fallback weights
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
        self.pprf_weight = (
            pprf_cfg.weight if pprf_cfg is not None else pprf_weight
        )
        self.pbg_weight = pbg_cfg.weight if pbg_cfg is not None else pbg_weight
        self.pi_weight = pi_cfg.weight if pi_cfg is not None else pi_weight
        self.n_wilson_samples = n_wilson_samples

        # pi_cfg.name == "gamma" -> learn per-bin alpha
        if (
            pi_cfg is not None
            and hasattr(pi_cfg, "name")
            and pi_cfg.name == "gamma"
        ):
            learn_concentration = True
        self.learn_concentration = learn_concentration

        # Fixed buffers (per-bin)
        self.register_buffer(
            "s_squared_per_group", _load_buffer(s_squared_per_group)
        )
        self.register_buffer(
            "bg_rate_per_group", _load_buffer(bg_rate_per_group)
        )
        n_bins = int(self.s_squared_per_group.shape[0])
        if concentration_per_group is not None:
            if isinstance(concentration_per_group, (int, float)):
                if pprf_n_pixels is None:
                    raise ValueError(
                        "pprf_n_pixels is required when"
                        " concentration_per_group is a scalar"
                    )
                conc = torch.full(
                    (n_bins, pprf_n_pixels),
                    float(concentration_per_group),
                )
            else:
                conc = _load_buffer(concentration_per_group)
                if pprf_quantile is not None:
                    threshold = torch.quantile(conc.float(), pprf_quantile)
                    conc[conc > threshold] *= pprf_conc_factor
                    conc = conc / conc.max()
                if conc.dim() == 1:
                    conc = conc.unsqueeze(0).expand(n_bins, -1).contiguous()
            self.register_buffer(
                "concentration_per_group",
                conc.clamp(min=1e-6),
            )
        else:
            self.concentration_per_group = None
        if bg_concentration_per_group is not None:
            self.register_buffer(
                "bg_concentration_per_group",
                _load_buffer(bg_concentration_per_group).clamp(min=0.1),
            )
        else:
            self.bg_concentration_per_group = None

        # Auto-initialize K, B from empirical tau via linear regression
        if init_from_tau and tau_per_group is not None:
            tau = _load_buffer(tau_per_group)
            init_log_K, init_log_B = self._fit_wilson_init(
                tau, self.s_squared_per_group
            )

        # Learnable variational parameters for q(log K), q(log B)
        self.q_log_K_loc = nn.Parameter(torch.tensor(float(init_log_K)))
        self.q_log_K_log_scale = nn.Parameter(torch.tensor(-2.0))
        self.q_log_B_loc = nn.Parameter(torch.tensor(float(init_log_B)))
        self.q_log_B_log_scale = nn.Parameter(torch.tensor(-2.0))

        # Per-bin learnable concentration (Gamma shape)
        if self.learn_concentration:
            n_groups = len(self.s_squared_per_group)
            if i_concentration_per_group is not None:
                # Initialize from Gamma MLE alpha values
                alpha_init = _load_buffer(i_concentration_per_group).clamp(
                    min=0.1
                )
                init_raw = torch.log(torch.expm1(alpha_init))
            else:
                init_raw = torch.full(
                    (n_groups,), math.log(math.expm1(init_alpha))
                )
            self.log_alpha_per_group = nn.Parameter(init_raw)

        # Fixed hyperprior parameters
        self.register_buffer("hp_log_K_loc", torch.tensor(hp_log_K_loc))
        self.register_buffer("hp_log_K_scale", torch.tensor(hp_log_K_scale))
        self.register_buffer("hp_log_B_loc", torch.tensor(hp_log_B_loc))
        self.register_buffer("hp_log_B_scale", torch.tensor(hp_log_B_scale))

    @staticmethod
    def _fit_wilson_init(tau: Tensor, s_sq: Tensor) -> tuple[float, float]:
        """Fit Wilson model to empirical tau via linear regression.

        log(tau_k) = -log(K) + 2B*s_k^2  ->  y = a + b*x
        where y = log(tau), x = s^2.  Then K = exp(-a), B = b/2.

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
        """Variational posterior q(log K)."""
        return Normal(self.q_log_K_loc, F.softplus(self.q_log_K_log_scale))

    def q_log_B(self) -> Normal:
        """Variational posterior q(log B)."""
        return Normal(self.q_log_B_loc, F.softplus(self.q_log_B_log_scale))

    # Hyperprior distributions

    def p_log_K(self) -> Normal:
        """Hyperprior p(log K)."""
        return Normal(self.hp_log_K_loc, self.hp_log_K_scale)

    def p_log_B(self) -> Normal:
        """Hyperprior p(log B)."""
        return Normal(self.hp_log_B_loc, self.hp_log_B_scale)

    #  KL terms

    def kl_hyperparams(self) -> Tensor:
        """KL(q(log K) || p(log K)) + KL(q(log B) || p(log B)).

        Closed-form Normal-Normal KL. Returns a scalar.
        """
        kl_K = kl_divergence(self.q_log_K(), self.p_log_K())
        kl_B = kl_divergence(self.q_log_B(), self.p_log_B())
        return kl_K + kl_B

    @staticmethod
    def compute_tau(K: Tensor, B: Tensor, s_sq: Tensor) -> Tensor:
        """Wilson prior rate: tau = (1/K) * exp(2B*s^2)."""
        return (1.0 / K) * torch.exp(2.0 * B * s_sq)

    #  Diagnostics

    def posterior_means(self) -> dict[str, float]:
        """Approximate posterior means of K and B for logging.

        Since q(log K) is Normal(mu, sigma), K = exp(log K) is LogNormal:
            E[K] = exp(mu + sigma^2/2)
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

    def extra_repr(self) -> str:
        means = self.posterior_means()
        s = (
            f"E[K]={means['K_mean']:.4f} +/- {means['K_std']:.4f}, "
            f"E[B]={means['B_mean']:.4f} +/- {means['B_std']:.4f}"
        )
        if self.learn_concentration:
            s += (
                f", alpha: mean={means['alpha_mean']:.3f} "
                f"[{means['alpha_min']:.3f}, {means['alpha_max']:.3f}]"
            )
        return s

    #  Forward

    def forward(
        self,
        rate: Tensor,
        counts: Tensor,
        qp: Distribution | ProfilePosterior,
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
            self.concentration_per_group,
            self.pprf_weight,
            self.mc_samples,
            self.eps,
            device,
            metadata=kwargs.get("metadata"),
        )
        kl = kl + kl_prf

        # Intensity KL: E_{q(K,B)}[ KL(q(I) || Gamma(alpha_k, alpha_k*tau_k)) ]
        # When learn_concentration=False, alpha_k = 1 (Exponential).
        # When True, alpha_k is a learnable per-bin parameter.
        s_sq = self.s_squared_per_group[groups]  # (B,)

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
                # Gamma(alpha_k, alpha_k*tau_k): mean = 1/tau_k, var = 1/(alpha_k*tau_k^2)
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
