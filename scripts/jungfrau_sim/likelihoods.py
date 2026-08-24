"""Pixel likelihoods on a ladder from naive to exact.

Every rung scores the same simulated shoebox against the same rate `lam = I*prof + B`.
They differ only in what they assume the detector did to `lam` on the way out:

  normal_free     Normal(x; lam, s^2), s fitted.   Discards the mean-variance coupling
                  entirely -- the default when someone reaches for an MSE loss.
  normal_coupled  Normal(x; lam, lam + sigma^2).   The *correct* Gaussian: Poisson
                  variance plus read noise. Should track `exact` closely.
  poisson_counts  Poisson(round(clamp(x, 0)); lam). The integer route. Implicitly
                  asserts sigma = 0, and the clamp censors the negative background.
  exact           sum_n Poisson(n; lam) * Normal(x; n, sigma). The true generative
                  law of the simulator, hence the ceiling every other rung is
                  measured against.
  gat             Generalized Anscombe: variance-stabilize, then unit Normal. The
                  cheap approximation to `exact`, and the transform already wired
                  into the integrator's data module.

All rungs work in photon-equivalent units, so `sigma` is read noise in photons
(0.024 for a real JUNGFRAU G0 pixel) and gain has already been divided out.
"""

from __future__ import annotations

import math

import torch

LOG_2PI = math.log(2.0 * math.pi)


def normal_free(x: torch.Tensor, lam: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    var = s.clamp_min(1e-6) ** 2
    return -0.5 * ((x - lam) ** 2 / var + torch.log(var) + LOG_2PI)


def normal_coupled(
    x: torch.Tensor, lam: torch.Tensor, sigma: torch.Tensor | float
) -> torch.Tensor:
    var = (lam + _as_tensor(sigma, lam) ** 2).clamp_min(1e-6)
    return -0.5 * ((x - lam) ** 2 / var + torch.log(var) + LOG_2PI)


def poisson_counts(n: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    lam = lam.clamp_min(1e-10)
    return n * torch.log(lam) - lam - torch.lgamma(n + 1.0)


def gat(
    x: torch.Tensor, lam: torch.Tensor, sigma: torch.Tensor | float
) -> torch.Tensor:
    """Generalized Anscombe transform, then a unit-variance Normal.

    The forward transform 2*sqrt(x + 3/8 + sigma^2) maps Poisson+Gaussian noise to
    approximately Normal(2*sqrt(lam + 3/8 + sigma^2), 1). The approximation degrades
    below lam ~ 1, which is exactly the background regime.
    """
    off = 0.375 + _as_tensor(sigma, lam) ** 2
    tx = 2.0 * torch.sqrt((x + off).clamp_min(0.0))
    mu = 2.0 * torch.sqrt((lam + off).clamp_min(1e-10))
    return -0.5 * ((tx - mu) ** 2 + LOG_2PI)


def _as_tensor(sigma: torch.Tensor | float, like: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(sigma):
        return sigma
    return torch.full_like(like, float(sigma))


def exact(
    x: torch.Tensor,
    lam: torch.Tensor,
    sigma: torch.Tensor | float,
    n_max: int | None = None,
) -> torch.Tensor:
    """log sum_n Poisson(n; lam) * Normal(x; n, sigma), the simulator's own law.

    `sigma` may be a scalar or a per-pixel tensor. Per-pixel is the physical case: the
    gain stage is chosen per pixel per shot and each stage has its own read noise, and
    crucially the stage is *observed* (the top 2 bits of the readout word), so
    conditioning on it is free information rather than an assumption.

    The sum is truncated at `n_max`, covering the Poisson bulk plus the read-noise
    spread. Cost is O(n_max) per pixel, so this is the expensive rung.
    """
    sig = _as_tensor(sigma, lam)
    if bool((sig <= 0).any()):
        raise ValueError("exact likelihood needs sigma > 0; use poisson_counts instead")

    lam_d = lam.detach()
    sig_d = sig.detach()

    # The summand Poisson(n;lam) * Normal(x;n,sigma) is concentrated where both factors
    # live, i.e. within a few sd of lam (and of x, which is centred on lam). Summing from
    # n=0 every time costs O(lam) per pixel and is hopeless at lam ~ 7000; instead window
    # each pixel around its own centre. Width is set by the *global* max sd so the tensor
    # stays rectangular, but the offset is per-pixel.
    sd = (lam_d + 1.0).sqrt()
    half = float(
        math.ceil(8.0 * float(sd.max()) + 8.0 * float(sig_d.max()) + 5.0)
    )
    if n_max is not None:
        half = min(half, float(n_max))
    width = int(2 * half + 1)

    centre = torch.round(0.5 * (x.detach() + lam_d)).clamp_min(0.0)
    lo = (centre - half).clamp_min(0.0)

    offsets = torch.arange(width, dtype=x.dtype, device=x.device)
    n = lo.unsqueeze(-1) + offsets  # (..., width), per-pixel window

    lam_e = lam.unsqueeze(-1).clamp_min(1e-10)
    x_e = x.unsqueeze(-1)
    sig_e = sig.unsqueeze(-1)

    log_pois = n * torch.log(lam_e) - lam_e - torch.lgamma(n + 1.0)
    log_norm = -0.5 * (((x_e - n) / sig_e) ** 2 + LOG_2PI) - torch.log(sig_e)
    return torch.logsumexp(log_pois + log_norm, dim=-1)


def hybrid(
    x: torch.Tensor,
    n: torch.Tensor,
    lam: torch.Tensor,
    sigma: torch.Tensor,
    deadband_ratio: float = 0.25,
) -> torch.Tensor:
    """Per-pixel choice between the two cheap rungs, by which regime the pixel is in.

    Rounding is lossless in two *opposite* limits, for opposite reasons:

      sigma << 0.5   the noise is far inside the rounding deadband, so round(x) == N
                     exactly and Poisson on the rounded count IS the exact likelihood.
      sigma >> 1     quantization (variance 1/12) is negligible against the noise, and
                     lam is large enough that Poisson is itself near-Gaussian, so
                     Normal(lam, lam + sigma^2) is near-exact.

    The failure is the *middle*, sigma ~ 0.5, where rounding neither recovers the count
    nor is negligible. On a JUNGFRAU that middle is G1 (sigma = 0.72), so this rung
    routes G0 to Poisson and everything else to the coupled Normal.

    VERDICT: not worth it. `study.py --only gain_stages` finds this ties `poisson_counts`
    and `exact` to within 0.01% bias / 0.01% RMSE at every intensity that populates G1 or
    G2, because by then the reflection is bright enough that the likelihood choice stops
    mattering at all. Kept as the null result -- it is the obvious idea, and it is
    measurably unnecessary. Prefer plain `poisson_counts`.
    """
    use_poisson = sigma < deadband_ratio
    return torch.where(
        use_poisson,
        poisson_counts(n, lam),
        normal_coupled(x, lam, sigma),
    )


LADDER = ("normal_free", "normal_coupled", "poisson_counts", "gat", "hybrid", "exact")

#: Rungs that consume rounded integer counts rather than real-valued photon equivalents.
NEEDS_COUNTS = frozenset({"poisson_counts", "hybrid"})


def log_prob(
    name: str,
    x: torch.Tensor,
    n: torch.Tensor,
    lam: torch.Tensor,
    sigma: torch.Tensor | float,
    s: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch to a rung. `x` is photon-equivalent, `n` its rounded/clamped counts."""
    if name == "normal_free":
        assert s is not None, "normal_free needs its fitted scale"
        return normal_free(x, lam, s)
    if name == "normal_coupled":
        return normal_coupled(x, lam, sigma)
    if name == "poisson_counts":
        return poisson_counts(n, lam)
    if name == "gat":
        return gat(x, lam, sigma)
    if name == "hybrid":
        return hybrid(x, n, lam, _as_tensor(sigma, lam))
    if name == "exact":
        return exact(x, lam, sigma)
    raise ValueError(f"unknown likelihood {name!r}; expected one of {LADDER}")
