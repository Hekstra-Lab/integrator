from dataclasses import dataclass
from typing import Literal


@dataclass
class IntegratorCfg:
    """Architecture and inference hyperparameters for an integrator `LightningModule`.

    Attributes:
        data_dim: Shoebox dimensionality, either `2d` or `3d`.
        d: Shoebox depth in pixels (number of z-slices); ignored for `2d` data but still validated positive.
        h: Shoebox height in pixels; must be positive.
        w: Shoebox width in pixels; must be positive.
        encoder_out: Width of the encoder embedding consumed by the surrogates.
        mc_samples: Monte Carlo samples drawn from each surrogate per forward pass; must be `>= 1`.
        predict_keys: Output columns to emit at predict time, or `default` for the built-in set.
        profile_support: Voxels the profile softmax is normalized over. `valid`
            spreads the profile across every valid voxel (default, unchanged
            behavior); `foreground` restricts it to the DIALS foreground bit,
            the learned-model analog of foreground summation. Ragged integrators
            only; falls back to `valid` per reflection when a shoebox has no
            foreground voxel.
        rb_gate_half_signal: Signal photons at which the profile-uncertainty term of
            the Rao-Blackwellized sigma is half-weighted. `None` applies it in full.
            Only consulted when an `i_rb`/`var_rb` column is requested; the term is
            unreliable where the data cannot constrain the profile, so a few hundred
            photons is a reasonable guard on weak data.
    """

    data_dim: Literal["2d", "3d"]
    d: int
    h: int
    w: int
    encoder_out: int = 64
    mc_samples: int = 4
    predict_keys: Literal["default"] | list[str] = "default"
    profile_support: Literal["valid", "foreground"] = "valid"
    rb_gate_half_signal: float | None = None
    profile_coord_channels: bool = False
    """Give the profile encoder two extra channels holding each pixel's offset
    in the beam-centred radial/tangential frame. A conv encoder is translation
    equivariant and so cannot otherwise know the azimuth that sets the streak
    direction. Requires `xyzcal.px.0/1` in the metadata and a beam centre."""
    beam_center: list[float] | None = None
    """Detector beam centre in pixels; filled from dataset.yaml when unset."""

    def __post_init__(self):
        if self.data_dim not in ("2d", "3d"):
            raise ValueError(
                f"data_dim must be '2d' or '3d', got {self.data_dim!r}"
            )

        for name in ("d", "h", "w"):
            v = getattr(self, name)
            if v <= 0:
                raise ValueError(f"{name} must be positive, got {v}")

        if self.mc_samples < 1:
            raise ValueError(f"mc_samples must be >= 1, got {self.mc_samples}")
