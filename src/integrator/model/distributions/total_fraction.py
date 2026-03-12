"""Total-fraction reparameterization for the joint (I, bg) posterior.

Instead of directly parameterizing (I, bg), the surrogate predicts a bivariate
distribution in the (log T, logit f) space, where:

    T = I + n_pixels * bg   (total count rate across the shoebox)
    f = I / T               (signal fraction)

Recovery:
    I  = f * T
    bg = (1 - f) * T / n_pixels

Motivation: if the Poisson likelihood drives a negative I-bg correlation, T and f
may be more nearly independent in the posterior — T is constrained by total counts
while f is constrained by the spatial profile.

The induced prior on (T, f) from independent Gamma(I) and Gamma(bg) priors is:
    log p(T, f) = log p_I(fT) + log p_bg((1-f)T/n) + log T - log n
which is handled by _joint_prior_kl_tf in loss.py (MC estimator).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, LogNormal, MultivariateNormal, constraints


class TotalFractionPosterior(Distribution):
    """Joint posterior over (T, f) in original space.

    Internally, (log T, logit f) ~ BivariateNormal(loc, L L^T).

    Args:
        loc:        (B, 2) — [mu_logT, mu_logitf]
        scale_tril: (B, 2, 2) — lower-triangular Cholesky, diagonal > 0
        n_pixels:   Shoebox pixel count (default 441 = 21×21). Used for bg = (1-f)*T/n.
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale_tril": constraints.lower_cholesky,
    }
    has_rsample = True

    def __init__(self, loc, scale_tril, n_pixels=441, validate_args=None):
        self.loc = loc
        self.scale_tril = scale_tril
        self.n_pixels = n_pixels
        self._mvn = MultivariateNormal(loc=loc, scale_tril=scale_tril)
        batch_shape = self._mvn.batch_shape
        event_shape = torch.Size([2])
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size([])):
        """Reparameterized samples of (T, f). Returns (*sample_shape, *batch, 2)."""
        z = self._mvn.rsample(sample_shape)      # (..., 2)
        T = z[..., 0].exp()                       # > 0
        f = torch.sigmoid(z[..., 1])              # in (0, 1)
        return torch.stack([T, f], dim=-1)

    def log_prob(self, x):
        """Log density in (T, f) space.

        log q(T, f) = log N(log T, logit f | loc, Σ)  +  log|J|
        where log|J| = -log T - log f - log(1-f)  (change-of-variables Jacobian).
        """
        T = x[..., 0].clamp(min=1e-10)
        f = x[..., 1].clamp(min=1e-10, max=1.0 - 1e-10)
        log_T   = T.log()
        logit_f = f.log() - (1.0 - f).log()
        z       = torch.stack([log_T, logit_f], dim=-1)
        log_jac = -log_T - f.log() - (1.0 - f).log()
        return self._mvn.log_prob(z) + log_jac

    def to_intensity_bg(self, samples):
        """Convert (T, f) samples → (I, bg). samples: (..., 2)."""
        T  = samples[..., 0]
        f  = samples[..., 1]
        I  = f * T
        bg = (1.0 - f) * T / self.n_pixels
        return I, bg

    @property
    def mean(self):
        """Point estimate at posterior mode → (I_est, bg_est), shape (..., 2)."""
        T_mode = self.loc[..., 0].exp()
        f_mode = torch.sigmoid(self.loc[..., 1])
        return torch.stack(
            [f_mode * T_mode, (1.0 - f_mode) * T_mode / self.n_pixels],
            dim=-1,
        )

    def marginal_i(self):
        """Approximate LogNormal for I via delta-method moment matching.

        log I = log T + log f
        E[log I]  ≈ mu_T + log sigmoid(mu_logitf)        (delta method)
        Var[log I] ≈ sigma_T² + (1 - f_mode)² * sigma_f²
        """
        mu_T        = self.loc[..., 0]
        mu_logit_f  = self.loc[..., 1]
        f_mode      = torch.sigmoid(mu_logit_f)
        sigma_T_sq  = self.scale_tril[..., 0, 0] ** 2
        sigma_f_sq  = self.scale_tril[..., 1, 0] ** 2 + self.scale_tril[..., 1, 1] ** 2
        mu_log_I    = mu_T + F.logsigmoid(mu_logit_f)
        var_log_I   = sigma_T_sq + (1.0 - f_mode) ** 2 * sigma_f_sq
        return LogNormal(loc=mu_log_I, scale=var_log_I.clamp(min=1e-10).sqrt())

    def marginal_bg(self):
        """Approximate LogNormal for bg via delta-method moment matching.

        log bg = log T + log(1-f) - log(n_pixels)
        E[log bg]  ≈ mu_T + log(1 - sigmoid(mu_logitf)) - log(n_pixels)
        Var[log bg] ≈ sigma_T² + f_mode² * sigma_f²
        """
        mu_T        = self.loc[..., 0]
        mu_logit_f  = self.loc[..., 1]
        f_mode      = torch.sigmoid(mu_logit_f)
        sigma_T_sq  = self.scale_tril[..., 0, 0] ** 2
        sigma_f_sq  = self.scale_tril[..., 1, 0] ** 2 + self.scale_tril[..., 1, 1] ** 2
        mu_log_bg   = mu_T + F.logsigmoid(-mu_logit_f) - math.log(self.n_pixels)
        var_log_bg  = sigma_T_sq + f_mode ** 2 * sigma_f_sq
        return LogNormal(loc=mu_log_bg, scale=var_log_bg.clamp(min=1e-10).sqrt())

    @property
    def log_space_correlation(self):
        """Pearson correlation of (log T, logit f) in [-1, 1]."""
        L_11   = self.scale_tril[..., 0, 0]
        L_21   = self.scale_tril[..., 1, 0]
        L_22   = self.scale_tril[..., 1, 1]
        std_T  = L_11
        std_f  = (L_21 ** 2 + L_22 ** 2).sqrt()
        return (L_11 * L_21) / (std_T * std_f + 1e-8)


class TotalFractionSurrogate(nn.Module):
    """Inference network outputting a TotalFractionPosterior over (T, f).

    Maps encoder features to (mu_logT, mu_logitf) and a lower-triangular
    Cholesky factor for the covariance of (log T, logit f).

    Args:
        in_features: Encoder output dimension.
        n_pixels:    Shoebox pixel count (H*W or D*H*W). Default 441 = 21×21.
        eps:         Stability constant added to Cholesky diagonal.
        diagonal:    If True, L_21 is fixed to zero (independent log T, logit f).
                     Uses 4 output params instead of 5.  Equivalent to independent
                     LogNormal(T) × LogisticNormal(f) with no cross-correlation.
    """

    def __init__(
        self,
        in_features: int,
        n_pixels: int = 441,
        eps: float = 1e-3,
        diagonal: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.n_params = 4 if diagonal else 5
        self.fc       = nn.Linear(in_features, self.n_params)
        self.n_pixels = n_pixels
        self.eps      = eps
        self.diagonal = diagonal

    def forward(self, x, x_=None):
        params = self.fc(x)  # (B, 4) or (B, 5)
        if self.diagonal:
            mu_T, mu_f, raw_L11, raw_L22 = params.unbind(dim=-1)
            L_21 = torch.zeros_like(mu_T)
        else:
            mu_T, mu_f, raw_L11, L_21, raw_L22 = params.unbind(dim=-1)
        L_11 = F.softplus(raw_L11) + self.eps
        L_22 = F.softplus(raw_L22) + self.eps
        loc  = torch.stack([mu_T, mu_f], dim=-1)  # (B, 2)
        zeros      = torch.zeros_like(L_11)
        scale_tril = torch.stack(
            [
                torch.stack([L_11, zeros], dim=-1),
                torch.stack([L_21, L_22],  dim=-1),
            ],
            dim=-2,
        )  # (B, 2, 2)
        return TotalFractionPosterior(loc=loc, scale_tril=scale_tril, n_pixels=self.n_pixels)
