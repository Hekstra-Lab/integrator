"""Gaussian smoother used by the DIALS smooth scale components.

A port of `dials::refinement::GaussianSmoother` (1D case) and the
`GaussianSmootherFirstFixed` subclass used by scaling. The smoothed
value at a point is a Gaussian-weighted mean of a few neighbouring
parameters, so it is linear in the parameters and trivially
differentiable; the weights depend only on the sample positions and are
built once per dataset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .types import DTYPE, IDTYPE

__all__ = [
    "GaussianSmoother1D",
    "SmootherWeights",
    "build_smoother",
    "nparam_to_val",
]


def nparam_to_val(n_params: int) -> int:
    """Number of sample intervals implied by a parameter count.

    Inverse of the smoother's own `n_params` rule, matching
    `SmoothMixin.nparam_to_val`.

    Args:
        n_params: number of smoother parameters, at least 2.

    Returns:
        The `num_intervals` to construct the smoother with.

    Raises:
        ValueError: if `n_params` is less than 2.
    """
    if n_params < 2:
        raise ValueError("a smooth scale factor needs at least 2 parameters")
    if n_params in (2, 3):
        return n_params - 1
    return n_params - 2


def _iround(x: torch.Tensor) -> torch.Tensor:
    """Round half away from zero, as `boost::math::iround` does.

    `torch.round` rounds halves to even, which shifts the index range by
    one whenever the normalised coordinate lands exactly on a half.
    """
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


@dataclass(frozen=True)
class SmootherWeights:
    """Gathered smoother weights for a fixed set of points.

    Attributes:
        index: `(N, n_average)` parameter indices. Padded entries point
            at a valid parameter and carry zero weight.
        weight: `(N, n_average)` Gaussian weights.
    """

    index: torch.Tensor
    weight: torch.Tensor

    def apply(self, params: torch.Tensor) -> torch.Tensor:
        """Interpolate `params` at the stored points.

        Args:
            params: `(n_values,)` smoother parameters.

        Returns:
            `(N,)` weighted mean, zero where every weight vanishes.
        """
        total = self.weight.sum(-1)
        safe = torch.where(total > 0.0, total, torch.ones_like(total))
        value = (self.weight * params[self.index]).sum(-1) / safe
        return torch.where(total > 0.0, value, torch.zeros_like(value))


class GaussianSmoother1D:
    """One-dimensional Gaussian smoother over evenly spaced parameters.

    Args:
        x_range: `(lo, hi)` of the raw unnormalised coordinate.
        num_intervals: number of sample intervals.

    Raises:
        ValueError: if `num_intervals` is not positive or `x_range` is
            not increasing.
    """

    def __init__(
        self, x_range: tuple[float, float], num_intervals: int
    ) -> None:
        if num_intervals < 1:
            raise ValueError("num_intervals must be positive")
        lo, hi = float(x_range[0]), float(x_range[1])
        if hi <= lo:
            raise ValueError("x_range must be increasing")

        self._x0 = lo
        self._n_sample = int(num_intervals)
        if self._n_sample == 1:
            self._n_values = 2
        elif self._n_sample == 2:
            self._n_values = 3
        else:
            self._n_values = self._n_sample + 2

        self._spacing = (hi - lo) / self._n_sample

        offset = 0.0 if self._n_values < 4 else -0.5
        self._positions = torch.arange(self._n_values, dtype=DTYPE) + offset

        self.set_smoothing(3, -1.0)

    def set_smoothing(self, num_average: int, sigma: float) -> None:
        """Set the averaging width, as `GaussianSmoother::set_smoothing`.

        Args:
            num_average: points included in each interpolation, capped
                at `n_values` and then required to lie in `[1, 5]`.
            sigma: kernel width; negative picks the default for
                `num_average`.

        Raises:
            ValueError: if the capped `num_average` is outside `[1, 5]`
                or `sigma` is zero.
        """
        n_average = min(num_average, self._n_values)
        if not 1 <= n_average <= 5:
            raise ValueError("num_average must be between 1 and 5")
        if sigma == 0.0:
            raise ValueError("sigma cannot be zero")

        self._n_average = n_average
        self._half_n_average = n_average / 2.0
        # Defaults 0.65, 0.7, 0.75, 0.8 for n_average 2, 3, 4, 5.
        self._sigma = 0.65 + 0.05 * (n_average - 2) if sigma < 0.0 else sigma

    @property
    def n_values(self) -> int:
        """Number of smoother parameters."""
        return self._n_values

    @property
    def n_samples(self) -> int:
        """Number of sample intervals."""
        return self._n_sample

    @property
    def n_average(self) -> int:
        """Number of parameters averaged per point."""
        return self._n_average

    @property
    def sigma(self) -> float:
        """Width of the smoothing kernel in normalised coordinates."""
        return self._sigma

    @property
    def spacing(self) -> float:
        """Raw-coordinate spacing between adjacent parameters."""
        return self._spacing

    @property
    def positions(self) -> torch.Tensor:
        """`(n_values,)` parameter positions in normalised coordinates."""
        return self._positions

    def _idx_range(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Half-open `[i1, i2)` parameter range contributing at `z`."""
        if self._n_values <= 3:
            i1 = torch.zeros_like(z, dtype=IDTYPE)
            return i1, torch.full_like(i1, self._n_values)

        i1 = _iround(z - self._half_n_average).to(IDTYPE) + 1
        i2 = i1 + self._n_average

        below = i1 < 0
        i2 = torch.where(below, torch.clamp(i2, min=2), i2)
        i1 = torch.where(below, torch.zeros_like(i1), i1)

        above = i2 > self._n_values
        i1 = torch.where(above, torch.clamp(i1, max=self._n_values - 2), i1)
        i2 = torch.where(above, torch.full_like(i2, self._n_values), i2)
        return i1, i2

    def weights(self, x: torch.Tensor) -> SmootherWeights:
        """Build the gather indices and weights for a set of points.

        Args:
            x: `(N,)` raw unnormalised coordinates.

        Returns:
            The `(N, n_average)` index and weight tensors.
        """
        z = (x.to(DTYPE) - self._x0) / self._spacing
        i1, i2 = self._idx_range(z)

        offsets = torch.arange(self._n_average, dtype=IDTYPE)
        column = i1.unsqueeze(-1) + offsets
        inside = column < i2.unsqueeze(-1)
        index = torch.clamp(column, 0, self._n_values - 1)

        ds = (z.unsqueeze(-1) - self._positions[index]) / self._sigma
        # DIALS weights with exp(-ds^2), not exp(-ds^2/2); sigma here is
        # not a standard-normal width.
        weight = torch.exp(-ds * ds) * inside
        return SmootherWeights(index=index, weight=weight)

    def value(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Interpolate `params` at `x` in one step.

        Args:
            x: `(N,)` raw unnormalised coordinates.
            params: `(n_values,)` smoother parameters.

        Returns:
            `(N,)` smoothed values.
        """
        return self.weights(x).apply(params)


def build_smoother(
    x: torch.Tensor, n_params: int
) -> tuple[GaussianSmoother1D, torch.Tensor]:
    """Construct the smoother a scale component would build for `x`.

    Mirrors `SmoothScaleComponent1D.update_reflection_data`: the
    coordinates are shifted so the smallest is zero, and the range is
    then widened to whole units.

    Args:
        x: `(N,)` normalised coordinate, e.g. `phi * s_norm_fac`.
        n_params: number of parameters the component holds.

    Returns:
        The smoother and the zeroed coordinates to evaluate it at.
    """
    zeroed = x.to(DTYPE) - x.min()
    lo = math.floor(round(float(zeroed.min()), 10))
    hi = max(math.ceil(round(float(zeroed.max()), 10)), 1)
    smoother = GaussianSmoother1D((lo, hi), nparam_to_val(n_params))
    if smoother.n_values != n_params:
        raise ValueError(
            f"smoother holds {smoother.n_values} values but the component "
            f"has {n_params} parameters"
        )
    return smoother, zeroed
