import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MLPScale(nn.Module):
    """MLP scale that replaces s/lp with a single learned correction."""

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        frame_min: float = 0.0,
        frame_max: float = 1000.0,
        beam_center: list[float] | None = None,
        r_max: float = 1500.0,
        d_min: float = 1.0,
        d_max: float = 60.0,
        head_init_std: float = 0.0,
        n_extra: int = 0,
        extra_loc: list[float] | None = None,
        extra_scale: list[float] | None = None,
        friedel_safe: bool = False,
    ):
        super().__init__()
        self.friedel_safe = bool(friedel_safe)

        frame_mid = (frame_min + frame_max) / 2.0
        frame_half = max((frame_max - frame_min) / 2.0, 1.0)
        self.register_buffer("frame_mid", torch.tensor(frame_mid))
        self.register_buffer("frame_half", torch.tensor(frame_half))

        cx, cy = beam_center or [0.0, 0.0]
        self.register_buffer("beam_cx", torch.tensor(cx))
        self.register_buffer("beam_cy", torch.tensor(cy))
        self.register_buffer("r_max", torch.tensor(max(r_max, 1.0)))

        self.register_buffer("d_min", torch.tensor(d_min))
        self.register_buffer("d_max", torch.tensor(max(d_max, d_min + 1.0)))

        # Extra metadata features
        # standardized by loc, scale): feat = (raw - loc) / scale.
        self.n_extra = int(n_extra)
        if self.n_extra > 0:
            loc = extra_loc if extra_loc is not None else [0.0] * self.n_extra
            scl = (
                extra_scale
                if extra_scale is not None
                else [1.0] * self.n_extra
            )
            if len(loc) != self.n_extra or len(scl) != self.n_extra:
                raise ValueError(
                    f"extra_loc/extra_scale must have length n_extra={self.n_extra}"
                )
            self.register_buffer("extra_loc", torch.tensor(loc).float())
            self.register_buffer(
                "extra_scale",
                torch.tensor(scl).float().clamp(min=1e-8),
            )

        n_input = (2 if self.friedel_safe else 4) + self.n_extra
        layers = []
        in_dim = n_input
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        nn.init.zeros_(self.net[-1].bias)
        if head_init_std > 0.0:
            nn.init.normal_(self.net[-1].weight, std=head_init_std)
        else:
            nn.init.zeros_(self.net[-1].weight)

    def forward(
        self,
        frame: Tensor,
        x: Tensor,
        y: Tensor,
        lp: Tensor,
        d: Tensor,
        extra: Tensor | None = None,
    ) -> Tensor:
        frame_norm = (frame - self.frame_mid) / self.frame_half
        d_norm = (d - self.d_min) / (self.d_max - self.d_min)

        if self.friedel_safe:
            features = torch.stack([frame_norm, d_norm], dim=-1)
        else:
            r = torch.sqrt(
                (x - self.beam_cx).pow(2) + (y - self.beam_cy).pow(2)
            )
            r_norm = r / self.r_max
            features = torch.stack([frame_norm, r_norm, d_norm, lp], dim=-1)
        if self.n_extra > 0:
            if extra is None:
                raise ValueError(
                    "MLPScale was built with extra metadata features but "
                    "`extra` was not provided; check scale_extra_features."
                )
            if extra.shape[-1] != self.n_extra:
                raise ValueError(
                    f"extra has {extra.shape[-1]} features, expected "
                    f"{self.n_extra} (scale_extra_features mismatch)."
                )
            extra = (extra - self.extra_loc) / self.extra_scale
            features = torch.cat([features, extra], dim=-1)
        return F.softplus(self.net(features).squeeze(-1))


def _chebyshev(x: Tensor, degree: int) -> Tensor:
    """Chebyshev-T basis `T_0..T_degree` evaluated at `x` in `[-1, 1]`.

    Returns shape `(..., degree + 1)`.
    """
    x = x.clamp(-1.0, 1.0)
    terms = [torch.ones_like(x), x]
    for _ in range(2, degree + 1):
        terms.append(2.0 * x * terms[-1] - terms[-2])
    return torch.stack(terms[: degree + 1], dim=-1)


class ChebyshevScale(nn.Module):
    """Smooth positive scale as a Chebyshev polynomial of the rotation frame."""

    def __init__(
        self,
        degree: int = 5,
        frame_min: float = 0.0,
        frame_max: float = 1000.0,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.degree = degree
        frame_mid = (frame_min + frame_max) / 2.0
        frame_half = max((frame_max - frame_min) / 2.0, 1.0)
        self.register_buffer("frame_mid", torch.tensor(frame_mid))
        self.register_buffer("frame_half", torch.tensor(frame_half))
        c = torch.zeros(degree + 1)
        # T_0 coefficient seeds a flat scale at `init_scale` (softplus output).
        c[0] = math.log(math.exp(init_scale) - 1.0)
        self.coeffs = nn.Parameter(c)

    def forward(self, frame: Tensor) -> Tensor:
        x = (frame - self.frame_mid) / self.frame_half
        basis = _chebyshev(x, self.degree)
        return F.softplus(basis @ self.coeffs)


class CoarseScale(nn.Module):
    """K(phi) * exp(2 B(phi) s^2); LP applied outside."""

    def __init__(
        self,
        frame_min: float = 0.0,
        frame_max: float = 1000.0,
        k_degree: int = 5,
        decay_degree: int = 0,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.k_degree = k_degree
        self.decay_degree = decay_degree
        frame_mid = (frame_min + frame_max) / 2.0
        frame_half = max((frame_max - frame_min) / 2.0, 1.0)
        self.register_buffer("frame_mid", torch.tensor(frame_mid))
        self.register_buffer("frame_half", torch.tensor(frame_half))

        # log K(phi): T_0 seeds log(init_scale) so K starts at init_scale.
        k = torch.zeros(k_degree + 1)
        k[0] = math.log(init_scale)
        self.k_coeffs = nn.Parameter(k)
        # B(phi): starts at zero (no decay at init); degree 0 -> one global B.
        self.b_coeffs = nn.Parameter(torch.zeros(decay_degree + 1))

    def forward(self, frame: Tensor, s_sq: Tensor) -> Tensor:
        x = ((frame - self.frame_mid) / self.frame_half).clamp(-1.0, 1.0)
        log_k = _chebyshev(x, self.k_degree) @ self.k_coeffs
        b = _chebyshev(x, self.decay_degree) @ self.b_coeffs
        log_scale = (log_k + 2.0 * b * s_sq).clamp(-8.0, 8.0)
        return torch.exp(log_scale)


class SolvedScale(nn.Module):
    """Scale coefficients are solved by weighted least squares (EM)."""

    def __init__(
        self,
        frame_min: float = 0.0,
        frame_max: float = 1000.0,
        k_degree: int = 5,
        decay_degree: int = 0,
        ridge: float = 1e-3,
    ):
        super().__init__()
        self.k_degree = k_degree
        self.decay_degree = decay_degree
        self.ridge = ridge
        frame_mid = (frame_min + frame_max) / 2.0
        frame_half = max((frame_max - frame_min) / 2.0, 1.0)
        self.register_buffer("frame_mid", torch.tensor(frame_mid))
        self.register_buffer("frame_half", torch.tensor(frame_half))

        n = (k_degree + 1) + (decay_degree + 1)
        self.n_basis = n
        self.register_buffer("theta", torch.zeros(n))  # solved coefficients
        self.register_buffer("_AtA", torch.zeros(n, n))
        self.register_buffer("_Atb", torch.zeros(n))
        self.register_buffer("_sum_wphi", torch.zeros(n))
        self.register_buffer("_sum_w", torch.zeros(()))
        self.residual: nn.Module | None = None

    def _phi(self, frame: Tensor, s_sq: Tensor) -> Tensor:
        x = ((frame - self.frame_mid) / self.frame_half).clamp(-1.0, 1.0)
        tk = _chebyshev(
            x, self.k_degree
        )  # (B, k+1) -> K(phi); col 0 is T_0 = 1
        tb = _chebyshev(x, self.decay_degree) * s_sq.unsqueeze(-1)  # (B, m+1)
        return torch.cat([tk, tb], dim=-1)  # (B, n_basis)

    def forward(self, frame: Tensor, s_sq: Tensor) -> Tensor:
        log_scale = self._phi(frame, s_sq) @ self.theta
        if self.residual is not None:
            log_scale = log_scale + self.residual(frame, s_sq)
        return torch.exp(log_scale.clamp(-8.0, 8.0))

    @torch.no_grad()
    def accumulate(
        self, frame: Tensor, s_sq: Tensor, log_target: Tensor, weight: Tensor
    ) -> None:
        """Add a batch to the normal equations for `log scale ~ log(J/I_h)`."""
        phi = self._phi(frame, s_sq)
        w = weight.clamp(min=0.0)
        wphi = phi * w.unsqueeze(-1)
        self._AtA += wphi.transpose(0, 1) @ phi
        self._Atb += wphi.transpose(0, 1) @ log_target
        self._sum_wphi += wphi.sum(0)
        self._sum_w += w.sum()

    @torch.no_grad()
    def solve(self) -> None:
        """Ridge-regularized weighted LS, gauge-fixed to geom-mean scale = 1."""
        n = self.n_basis
        eye = torch.eye(n, device=self.theta.device, dtype=self.theta.dtype)
        theta = torch.linalg.solve(self._AtA + self.ridge * eye, self._Atb)

        if float(self._sum_w) > 0:
            mu = (self._sum_wphi / self._sum_w) @ theta
            theta = theta.clone()
            theta[0] = theta[0] - mu
        self.theta.copy_(theta)
        self.reset_accumulators()

    @torch.no_grad()
    def reset_accumulators(self) -> None:
        self._AtA.zero_()
        self._Atb.zero_()
        self._sum_wphi.zero_()
        self._sum_w.zero_()
