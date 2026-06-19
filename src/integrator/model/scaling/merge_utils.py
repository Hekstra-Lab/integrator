"""Small self-contained helpers for the scaling/merging models."""

from typing import Literal

import torch
from torch import Tensor

from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)
from integrator.model.integrators.integrator_utils import (
    IntegratorBaseOutputs,
    _assemble_outputs,
)

__all__ = [
    "IntegratorBaseOutputs",
    "_assemble_outputs",
    "_scatter_sum_compact",
    "_sample_profile",
    "_log_loss",
]


def _scatter_sum_compact(
    src: Tensor, index: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Scatter-sum `src` over the unique values of `index`.

    Returns:
        out: `(n_unique,)` sum of `src` grouped by `index`.
        inverse: `(B,)` mapping each row of `src` to its position in `out`.
        unique_idx: `(n_unique,)` the unique values of `index`.
    """
    unique_idx, inverse = torch.unique(index, return_inverse=True)
    out = torch.zeros(len(unique_idx), device=src.device, dtype=src.dtype)
    out.scatter_add_(0, inverse, src)
    return out, inverse, unique_idx


def _sample_profile(qp, mc_samples: int) -> Tensor:
    """Profile samples in `(B, S, K)` order from a profile surrogate output."""
    if isinstance(qp, ProfileSurrogateOutput):
        return qp.zp.permute(1, 0, 2)
    return qp.rsample([mc_samples]).permute(1, 0, 2)


def _log_loss(
    self,
    kl,
    nll,
    total_loss,
    step: Literal["train", "val"],
    kl_components: dict[str, Tensor] | None = None,
):
    """Log the ELBO and its components (mirrors the base integrator's logger)."""
    if step == "train":
        self.log(
            "train elbo",
            total_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
    else:
        self.log(
            "val elbo", total_loss, on_step=False, on_epoch=True, prog_bar=True
        )
    self.log(f"{step} kl", kl, on_step=False, on_epoch=True)
    self.log(f"{step} nll", nll, on_step=False, on_epoch=True)
    self.log("epoch", float(self.current_epoch), on_step=False, on_epoch=True)
    if kl_components is not None:
        for name, value in kl_components.items():
            self.log(f"{step} kl_{name}", value, on_step=False, on_epoch=True)
