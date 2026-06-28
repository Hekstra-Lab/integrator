from dataclasses import dataclass
from typing import Literal


@dataclass
class MergingIntegratorCfg:
    """Architecture and inference hyperparameters for the amortized merger.

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
        wilson_centric_prior: Give centric reflections the chi^2_1 Wilson prior.
        d_min: Minimum d-spacing (used to normalize the scale's d input).
        scale_mlp: Use the MLP scale; otherwise a frame-only Chebyshev scale.
        scale_degree: Chebyshev degree (frame-only fallback scale).
        scale_mlp_hidden: MLP scale hidden width.
        scale_mlp_layers: MLP scale depth.
        scale_head_init_std: Std for the MLP scale output-head init (0 = flat).
        scale_frame_min: Minimum rotation frame (for frame normalization).
        scale_frame_max: Maximum rotation frame.
        scale_beam_center: Detector beam center `[cx, cy]` in pixels.
        scale_r_max: Maximum detector radius (for radius normalization).
        scale_extra_features: Extra metadata.npy columns to feed the MLP scale.
            A bare name is one scalar; a nested list is a vector standardized
            together with one shared scale, e.g. `[[s1.0, s1.1, s1.2], d]`.
        scale_standardize: Auto-compute loc/scale from the dataset at build
            time (per feature, shared within a group). Off passes columns raw.
        scale_extra_loc: Manual standardization offsets; overrides auto.
        scale_extra_scale: Manual standardization scales; overrides auto.
    """

    data_dim: Literal["2d", "3d"]
    d: int
    h: int
    w: int
    n_hkl: int | None = None
    encoder_out: int = 64
    mc_samples: int = 4
    predict_keys: Literal["default"] | list[str] = "default"

    # Merge / Wilson prior
    wilson_alpha: float = 1.0
    merge_kl_weight: float = 1.0
    wilson_centric_prior: bool = False
    anomalous: bool = True

    signal_probability_gate_init: float = -2.0

    # delta (anomalous fraction): closed-form empirical-Bayes
    sigma_delta_init: float = 0.05
    sigma_delta_ema: float = 0.99

    log_merging_stats: bool = False
    merging_stats_bins: int = 10

    # Optimization
    scaling_lr: float | None = None

    # Scale field
    d_min: float = 1.0
    scale_mlp: bool = True

    scale_mode: Literal["mlp", "chebyshev", "coarse", "solved"] | None = None
    scale_decay_degree: int = 0  # B(phi) Chebyshev degree (0 = global B)
    scale_ridge: float = 1.0e-3  # ridge for the solved-scale least squares
    scale_solve_warmup: int = 2  # epochs before the first EM scale solve
    scale_degree: int = 5
    scale_mlp_hidden: int = 64
    scale_mlp_layers: int = 2
    scale_head_init_std: float = 0.0
    scale_frame_min: float = 0.0
    scale_frame_max: float = 1000.0
    scale_beam_center: list[float] | None = None
    scale_r_max: float = 1500.0

    # Extra metadata.npy columns
    scale_extra_features: list[str] | None = None
    scale_standardize: bool = True
    scale_extra_loc: list[float] | None = None
    scale_extra_scale: list[float] | None = None

    # Model-vs-DIALS epoch scatter logging
    log_intensity_scatter: bool = False
    log_background_scatter: bool = False

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
