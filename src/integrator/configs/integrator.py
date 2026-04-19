from dataclasses import dataclass
from typing import Literal


@dataclass
class IntegratorCfg:
    data_dim: Literal["2d", "3d"]
    d: int
    h: int
    w: int
    lr: float = 0.001
    encoder_out: int = 64
    weight_decay: float = 0.0
    decoder_weight_decay: float | None = None
    qp_smoothness_weight: float | None = None
    qp_orthogonality_weight: float | None = None
    qp_sparsity_weight: float | None = None
    # Penalties that act on the profile tensor itself (softmax output of
    # qp), not on the decoder weights. Used when qp has no basis decoder
    # (e.g. DirichletDistribution) or in addition to the basis penalties.
    qp_profile_tv_weight: float | None = None
    qp_profile_entropy_weight: float | None = None
    # Gaussian prior on the translation head's predicted shift. Penalty
    # is `shift_prior_weight * sum((shift / shift_prior_sigma) ** 2)`,
    # mean-reduced over the batch. Equivalent to MAP under a diagonal
    # N(0, diag(σ²)) prior. shift_prior_sigma may be a scalar (isotropic)
    # or a list matching the shift dimensionality (anisotropic per-axis).
    # Only active when the surrogate has a shift head; otherwise ignored.
    shift_prior_weight: float | None = None
    shift_prior_sigma: float | list[float] = 0.5
    lr_schedule: Literal["cosine_warmup"] | None = None
    warmup_epochs: int = 5
    lr_min: float = 1.0e-5
    mc_samples: int = 4
    renyi_scale: float = 0.0
    predict_keys: Literal["default"] | list[str] = "default"
    group_hidden_dim: int = 64

    def __post_init__(self):
        if self.data_dim not in ("2d", "3d"):
            raise ValueError(
                f"data_dim must be '2d' or '3d', got {self.data_dim!r}"
            )

        for name in ("d", "h", "w"):
            v = getattr(self, name)
            if v <= 0:
                raise ValueError(f"{name} must be positive, got {v}")

        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")

        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be non-negative, got {self.weight_decay}"
            )

        if (
            self.decoder_weight_decay is not None
            and self.decoder_weight_decay < 0
        ):
            raise ValueError(
                "decoder_weight_decay must be non-negative, got "
                f"{self.decoder_weight_decay}"
            )

        for name in (
            "qp_smoothness_weight",
            "qp_orthogonality_weight",
            "qp_sparsity_weight",
            "qp_profile_tv_weight",
            "qp_profile_entropy_weight",
            "shift_prior_weight",
        ):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise ValueError(f"{name} must be non-negative, got {v}")

        if isinstance(self.shift_prior_sigma, list):
            if any(s <= 0 for s in self.shift_prior_sigma):
                raise ValueError(
                    f"shift_prior_sigma entries must be > 0, got "
                    f"{self.shift_prior_sigma}"
                )
        elif self.shift_prior_sigma <= 0:
            raise ValueError(
                f"shift_prior_sigma must be > 0, got {self.shift_prior_sigma}"
            )

        if self.warmup_epochs < 0:
            raise ValueError(
                f"warmup_epochs must be non-negative, got {self.warmup_epochs}"
            )
        if self.lr_min < 0:
            raise ValueError(f"lr_min must be non-negative, got {self.lr_min}")
        if self.lr_min > self.lr:
            raise ValueError(
                f"lr_min ({self.lr_min}) must be <= lr ({self.lr})"
            )

        if self.mc_samples < 1:
            raise ValueError(f"mc_samples must be >= 1, got {self.mc_samples}")


@dataclass
class IntegratorConfig:
    name: str
    args: IntegratorCfg

    def __post_init__(self):
        from integrator.registry import REGISTRY

        valid = REGISTRY["integrator"].keys()
        if self.name not in valid:
            raise ValueError(
                f"Unknown integrator '{self.name}'. "
                f"Available integrators: {sorted(valid)}"
            )
