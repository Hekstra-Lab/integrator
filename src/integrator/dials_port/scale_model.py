"""The DIALS physical scaling model as a differentiable torch module.

The inverse scale factor is the product of three components, all
functions of the rotation coordinate `phi` (the fractional image index,
not an angle):

    scale       S = smoother(phi * s_norm_fac)
    decay       S = exp(B / (2 d^2)),  B = smoother(phi * d_norm_fac)
    absorption  S = 1 + sum_lm C_lm Y_lm

Everything that does not depend on the parameters -- smoother weights
and the spherical harmonic basis -- is precomputed into `ScaleFeatures`,
leaving `forward` as a handful of gathers and products. Gradients come
from autograd; no derivative is written out by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .harmonics import harmonic_table, n_abs_params
from .io import Experiment
from .smoother import SmootherWeights, build_smoother
from .types import DTYPE, Reflections

__all__ = ["PhysicalScalingModel", "ScaleFeatures", "build_features"]

DEFAULT_DECAY_RESTRAINT = 1e-1


@dataclass
class ScaleFeatures:
    """Per-reflection inputs to `PhysicalScalingModel`.

    Attributes:
        scale: smoother weights on the scale coordinate.
        decay: smoother weights on the decay coordinate.
        d: `(N,)` resolution in Angstrom.
        harmonics: `(N, lmax * (lmax + 2))` absorption basis.
    """

    scale: SmootherWeights
    decay: SmootherWeights
    d: torch.Tensor
    harmonics: torch.Tensor

    def __len__(self) -> int:
        return int(self.d.shape[0])


def build_features(
    refl: Reflections,
    expt: Experiment,
    lmax: int = 4,
    points_per_degree: int | None = None,
    sqrt2: float = float(np.sqrt(2.0)),
) -> ScaleFeatures:
    """Precompute the parameter-independent part of the scaling model.

    Args:
        refl: observations to evaluate the model on.
        expt: experiment carrying a refined `scaling_model`, which
            supplies the normalisation factors and parameter counts.
        lmax: maximum spherical harmonic degree.
        points_per_degree: if given, quantize the absorption angles onto
            the DIALS lookup grid. Needed only to reproduce a DIALS
            `inverse_scale_factor` for a dataset over 100k reflections.
        sqrt2: harmonic normalisation constant; pass
            `harmonics.SQRT2_DIALS` for DIALS' truncated literal.

    Returns:
        The precomputed features.

    Raises:
        ValueError: if `expt` carries no refined scaling model.
    """
    model = expt.scaling_model
    if model is None:
        raise ValueError("experiment has no scaling model")

    config = model.config
    phi = refl.phi.to(DTYPE)

    scale_smoother, scale_x = build_smoother(
        phi * config["s_norm_fac"], len(model.components["scale"])
    )
    decay_smoother, decay_x = build_smoother(
        phi * config["d_norm_fac"], len(model.components["decay"])
    )

    return ScaleFeatures(
        scale=scale_smoother.weights(scale_x),
        decay=decay_smoother.weights(decay_x),
        d=refl.d.to(DTYPE),
        harmonics=harmonic_table(
            refl,
            expt,
            lmax=lmax,
            points_per_degree=points_per_degree,
            sqrt2=sqrt2,
        ),
    )


class PhysicalScalingModel(torch.nn.Module):
    """Scale, decay and absorption corrections with refinable parameters.

    Args:
        scale: `(n_scale,)` initial smoother parameters.
        decay: `(n_decay,)` initial B-factor smoother parameters.
        absorption: `(lmax * (lmax + 2),)` initial surface coefficients.
        abs_surface_weight: L2 restraint weight on the surface.
        decay_restraint: L2 restraint weight on the B-factors.
    """

    def __init__(
        self,
        scale: torch.Tensor,
        decay: torch.Tensor,
        absorption: torch.Tensor,
        abs_surface_weight: float = 5e5,
        decay_restraint: float = DEFAULT_DECAY_RESTRAINT,
    ) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(scale.to(DTYPE).clone())
        self.decay = torch.nn.Parameter(decay.to(DTYPE).clone())
        self.absorption = torch.nn.Parameter(absorption.to(DTYPE).clone())
        self.abs_surface_weight = float(abs_surface_weight)
        self.decay_restraint = float(decay_restraint)

    @classmethod
    def from_dials(cls, expt: Experiment) -> PhysicalScalingModel:
        """Build a model from DIALS' own refined parameters.

        Args:
            expt: experiment read from a scaled `.expt`.

        Returns:
            A model holding the refined parameters, ready to
            forward-evaluate.

        Raises:
            ValueError: if `expt` carries no physical scaling model, or
                the absorption vector does not match `lmax`.
        """
        model = expt.scaling_model
        if model is None or model.model_id != "physical":
            raise ValueError("experiment has no physical scaling model")

        config = model.config
        lmax = int(config["lmax"])
        absorption = torch.as_tensor(
            model.components["absorption"], dtype=DTYPE
        )
        if absorption.numel() != n_abs_params(lmax):
            raise ValueError(
                f"absorption has {absorption.numel()} parameters, expected "
                f"{n_abs_params(lmax)} for lmax={lmax}"
            )

        return cls(
            scale=torch.as_tensor(model.components["scale"], dtype=DTYPE),
            decay=torch.as_tensor(model.components["decay"], dtype=DTYPE),
            absorption=absorption,
            abs_surface_weight=float(config["abs_surface_weight"]),
        )

    def scale_component(self, feats: ScaleFeatures) -> torch.Tensor:
        """`(N,)` smoothly varying overall scale."""
        return feats.scale.apply(self.scale)

    def decay_component(self, feats: ScaleFeatures) -> torch.Tensor:
        """`(N,)` resolution-dependent decay correction."""
        b = feats.decay.apply(self.decay)
        return torch.exp(b / (2.0 * feats.d * feats.d))

    def absorption_component(self, feats: ScaleFeatures) -> torch.Tensor:
        """`(N,)` absorption surface correction."""
        return 1.0 + feats.harmonics @ self.absorption

    def forward(self, feats: ScaleFeatures) -> torch.Tensor:
        """Inverse scale factor for every reflection.

        Args:
            feats: precomputed features from `build_features`.

        Returns:
            `(N,)` inverse scale factor.
        """
        return (
            self.scale_component(feats)
            * self.decay_component(feats)
            * self.absorption_component(feats)
        )

    def restraint(self) -> torch.Tensor:
        """Scalar L2 penalty on the absorption and decay parameters."""
        return (
            self.abs_surface_weight * (self.absorption * self.absorption).sum()
            + self.decay_restraint * (self.decay * self.decay).sum()
        )
