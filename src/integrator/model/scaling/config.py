from dataclasses import dataclass
from typing import Literal


@dataclass
class MergingIntegratorCfg:
    """Architecture and inference hyperparameters for the amortized merger.

    The scaling sibling of `IntegratorCfg`: it carries the same shoebox/encoder
    fields plus the merge- and scale-specific knobs the merger needs. The
    factory selects this dataclass via the integrator's `CFG_CLASS` attribute,
    so `main`'s `IntegratorCfg` stays free of scaling-only fields.

    Attributes:
        data_dim: Shoebox dimensionality, `2d` or `3d`.
        d: Shoebox depth in pixels (z-slices).
        h: Shoebox height in pixels.
        w: Shoebox width in pixels.
        encoder_out: Width of the encoder embedding consumed by the heads.
        mc_samples: Monte Carlo samples drawn per forward pass; must be `>= 1`.
        predict_keys: Output columns to emit at predict time, or `default`.
        n_hkl: Number of unique reflection (asu) ids; sizes the merge buffers.
        wilson_alpha: Wilson prior shape (`1.0` -> acentric exponential).
        merge_kl_weight: Weight on the per-HKL Wilson intensity KL.
        scaling_lr: Decoupled learning rate for the scale field; `None` -> `lr`.
        consistency_weight: Weight on the data-only scaling-consistency loss.
        consistency_pool_friedel: Pool the consistency target over Friedel mates.
        centric_anchor_weight: Weight on the centric zero-anomalous scale anchor.
        double_wilson_weight: Weight on the double-Wilson Friedel coupling.
        wilson_centric_prior: Give centric reflections the chi^2_1 Wilson prior.
        dmin: Minimum d-spacing (used to normalize the scale's d input).
        scale_mlp: Use the MLP scale; otherwise a frame-only Chebyshev scale.
        scale_degree: Chebyshev degree (frame-only fallback scale).
        scale_mlp_hidden: MLP scale hidden width.
        scale_mlp_layers: MLP scale depth.
        scale_head_init_std: Std for the MLP scale output-head init (0 = flat).
        scale_frame_min: Minimum rotation frame (for frame normalization).
        scale_frame_max: Maximum rotation frame.
        scale_beam_center: Detector beam center `[cx, cy]` in pixels.
        scale_r_max: Maximum detector radius (for radius normalization).
        scale_sh_lmax: Max spherical-harmonic order for crystal-frame absorption.
        scale_mlp_absorption: Feed crystal-frame SH absorption to the MLP scale.
        scale_mlp_absorption_even_only: Keep only even-l (Friedel-safe) harmonics.
    """

    data_dim: Literal["2d", "3d"]
    d: int
    h: int
    w: int
    # Number of unique merge reflections; sizes the merge buffers. Auto-filled
    # from dataset.yaml's `n_hkl` block by the factory when omitted.
    n_hkl: int | None = None
    encoder_out: int = 64
    mc_samples: int = 4
    predict_keys: Literal["default"] | list[str] = "default"

    # Merge / Wilson prior
    wilson_alpha: float = 1.0
    merge_kl_weight: float = 1.0
    wilson_centric_prior: bool = False
    # Anomalous: merge on the Friedel-SEPARATE id (`miller_idx_unfriedelized`)
    # so I(+) != I(-). If false, merge on the pooled `miller_idx_friedelized`.
    anomalous: bool = True

    # Optimization
    scaling_lr: float | None = None

    # Anomalous-preserving auxiliary losses (all default off)
    consistency_weight: float = 0.0
    consistency_pool_friedel: bool = False
    centric_anchor_weight: float = 0.0
    double_wilson_weight: float = 0.0

    # Scale field
    dmin: float = 1.0
    scale_mlp: bool = True
    scale_degree: int = 5
    scale_mlp_hidden: int = 64
    scale_mlp_layers: int = 2
    scale_head_init_std: float = 0.0
    scale_frame_min: float = 0.0
    scale_frame_max: float = 1000.0
    scale_beam_center: list[float] | None = None
    scale_r_max: float = 1500.0
    scale_sh_lmax: int = 4
    scale_mlp_absorption: bool = False
    scale_mlp_absorption_even_only: bool = True

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
        if self.n_hkl is not None and self.n_hkl <= 0:
            raise ValueError(f"n_hkl must be positive, got {self.n_hkl}")
