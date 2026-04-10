"""Per-bin loss with fully Bayesian Wilson intensity prior.

Intensity prior — Wilson model (learned, per-bin):
    log K ~ Normal(μ_K, σ_K)         variational posterior q(log K)
    log B ~ Normal(μ_B, σ_B)         variational posterior q(log B)
    τ_k   = (1/K) · exp(2B · s_k²)  per-bin rate from Wilson formula
    Σ_k   = 1/τ_k = K · exp(-2B · s_k²)  per-bin expected intensity

    When learn_concentration=False (default):
        I_k  ~ Gamma(1, τ_k) = Exp(τ_k)

    When learn_concentration=True:
        I_k  ~ Gamma(α_k, α_k · τ_k)
        where α_k is a learnable per-bin concentration parameter.
        This keeps E[I_k] = Σ_k (Wilson mean) but allows the prior
        variance to vary per bin:  Var[I_k] = Σ_k² / α_k.
        When α_k = 1, this reduces to the standard Exponential.

Background prior — empirical Bayes (fixed, per-bin):
    bg_k  ~ Exp(λ_k)               fixed per-group rate

Profile prior — depends on surrogate type:
    Dirichlet surrogate:  prf_k ~ Dir(α_k)          fixed per-group concentration
    Latent decoder (ProfilePosterior):  h ~ N(0, σ²I)  global prior

ELBO:
    L = E_q[ log p(x | I, prf, bg) ]
      - KL(q(prf) || p(prf))                              — profile (see above)
      - E_{q(K,B)}[ KL(q(I) || Gamma(α_k, α_k·τ_k)) ]   — intensity (Wilson)
      - KL(q(bg)  || Exp(λ_k))                            — background
      - KL(q(log K) || p(log K)) / N                      — hyperprior
      - KL(q(log B) || p(log B)) / N                      — hyperprior
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import (
    Dirichlet,
    Distribution,
    Gamma,
    Normal,
    Poisson,
    kl_divergence,
)

from integrator.model.distributions.logistic_normal import ProfilePosterior
from integrator.model.loss.loss import _kl
from integrator.model.loss.per_bin_loss import _load_buffer


class WilsonPerBinLoss(nn.Module):
    """ELBO loss with learned Wilson intensity prior and empirical Bayes bg/profile.

    The Wilson distribution models expected intensity in bin k as:
        Σ_k = K · exp(-2B · s_k²)
    where s_k² = 1/(4d_k²) is precomputed per bin. The prior rate for
    bin k is τ_k = 1/Σ_k = (1/K) · exp(2B · s_k²).

    K and B are two global hyperparameters inferred via variational inference.
    Background and profile priors remain per-bin empirical Bayes.

    Parameters
    ----------
    mc_samples : int
        Monte Carlo samples for KL estimation.
    eps : float
        Numerical stability constant for Poisson rate.
    s_squared_per_group : list[float] or str
        1/(4d²) per resolution bin.
    bg_rate_per_group : list[float] or str
        Exponential rates for background prior, one per group.
    concentration_per_group : str
        Path to .pt file with Dirichlet concentrations (n_groups, n_pixels).
        Only used when the profile surrogate returns a Distribution (Dirichlet).
        Ignored when the surrogate returns a ProfilePosterior (latent decoder),
        which uses a global N(0, sigma_prior²I) prior instead.
    tau_per_group : list[float] or str or None
        If provided, auto-initializes q(log K) and q(log B) via linear
        regression of log(τ) on s².
    init_log_K : float
        Initial μ for q(log K). Ignored if tau_per_group is provided.
    init_log_B : float
        Initial μ for q(log B). Ignored if tau_per_group is provided.
    hp_log_K_loc, hp_log_K_scale : float
        Hyperprior p(log K) ~ Normal(loc, scale).
    hp_log_B_loc, hp_log_B_scale : float
        Hyperprior p(log B) ~ Normal(loc, scale).
    n_wilson_samples : int
        MC samples over (K, B) for the intensity KL outer expectation.
    bg_concentration : float
        Shape parameter for the background Gamma prior. Default 1.0
        gives Exp(lambda_k). Higher values give a tighter prior around
        mean = 1/lambda_k, with CV = 1/sqrt(bg_concentration).
    learn_concentration : bool
        If True, learn a per-bin concentration parameter α_k for the
        intensity Gamma prior: I_k ~ Gamma(α_k, α_k·τ_k).
        The Wilson mean is preserved (E[I_k] = Σ_k) but the variance
        adapts per bin.  When False (default), α_k = 1 (Exponential).
    init_alpha : float
        Initial value for α_k when learn_concentration=True.
    pprf_weight, pbg_weight, pi_weight : float
        Scaling factors for each KL term (set all to 1.0 for a proper ELBO).
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
        concentration_per_group: str,
        bg_concentration: float = 1.0,
        # Optional auto-init from empirical tau
        tau_per_group: list[float] | str | None = None,
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
        self.pprf_weight = pprf_cfg.weight if pprf_cfg is not None else pprf_weight
        self.pbg_weight = pbg_cfg.weight if pbg_cfg is not None else pbg_weight
        self.pi_weight = pi_cfg.weight if pi_cfg is not None else pi_weight
        self.n_wilson_samples = n_wilson_samples
        self.learn_concentration = learn_concentration

        # -- Fixed buffers (empirical Bayes, per-bin) -------------------------
        self.register_buffer(
            "s_squared_per_group", _load_buffer(s_squared_per_group)
        )
        self.register_buffer(
            "bg_rate_per_group", _load_buffer(bg_rate_per_group)
        )
        self.register_buffer(
            "concentration_per_group",
            _load_buffer(concentration_per_group).clamp(min=1e-6),
        )
        if bg_concentration_per_group is not None:
            self.register_buffer(
                "bg_concentration_per_group",
                _load_buffer(bg_concentration_per_group).clamp(min=0.1),
            )
        else:
            self.bg_concentration_per_group = None

        # -- Auto-initialize from empirical tau if provided -------------------
        if tau_per_group is not None:
            tau = _load_buffer(tau_per_group)
            init_log_K, init_log_B = self._fit_wilson_init(
                tau, self.s_squared_per_group
            )

        # -- Learnable variational parameters for q(log K), q(log B) ---------
        self.q_log_K_loc = nn.Parameter(torch.tensor(float(init_log_K)))
        self.q_log_K_log_scale = nn.Parameter(torch.tensor(-2.0))
        self.q_log_B_loc = nn.Parameter(torch.tensor(float(init_log_B)))
        self.q_log_B_log_scale = nn.Parameter(torch.tensor(-2.0))

        # -- Per-bin learnable concentration (Gamma shape) ----------------------
        if self.learn_concentration:
            n_groups = len(self.s_squared_per_group)
            if i_concentration_per_group is not None:
                # Initialize from Gamma MLE alpha values
                alpha_init = _load_buffer(i_concentration_per_group).clamp(min=0.1)
                init_raw = torch.log(torch.expm1(alpha_init))
            else:
                init_raw = torch.full(
                    (n_groups,), math.log(math.expm1(init_alpha))
                )
            self.log_alpha_per_group = nn.Parameter(init_raw)

        # -- Fixed hyperprior parameters
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

        # Profile KL: per-bin Dirichlet or global Normal (latent decoder)
        if isinstance(qp, ProfilePosterior):
            kl_prf = qp.kl_divergence() * self.pprf_weight
        else:
            alpha = self.concentration_per_group[groups]  # (B, n_pixels)
            p_prf = Dirichlet(alpha)
            kl_prf = (
                _kl(qp, p_prf, self.mc_samples, eps=self.eps)
                * self.pprf_weight
            )
        kl = kl + kl_prf

        # Intensity KL: E_{q(K,B)}[ KL(q(I) || Gamma(α_k, α_k·τ_k)) ]
        # When learn_concentration=False, α_k = 1 (Exponential).
        # When True, α_k is a learnable per-bin parameter.
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
                # Gamma(α_k, α_k·τ_k): mean = 1/τ_k (Wilson), var = 1/(α_k·τ_k²)
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

        # Background KL: KL(q(bg) || Gamma(alpha_bg, alpha_bg * lambda_k))
        bg_rate_per_refl = self.bg_rate_per_group[groups]  # (B,)
        if self.bg_concentration_per_group is not None:
            alpha_bg = self.bg_concentration_per_group[groups]  # (B,)
        else:
            alpha_bg = torch.full_like(bg_rate_per_refl, self.bg_concentration)
        p_bg = Gamma(
            concentration=alpha_bg,
            rate=alpha_bg * bg_rate_per_refl,
        )
        kl_bg = _kl(qbg, p_bg, self.mc_samples, eps=self.eps) * self.pbg_weight
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
