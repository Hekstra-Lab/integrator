from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from integrator.configs.integrator import IntegratorCfg
from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)


@dataclass
class IntegratorBaseOutputs:
    rates: Tensor
    counts: Tensor
    mask: Tensor
    qbg: Any
    qp: Any
    qi: Any
    zp: Tensor
    zbg: Tensor
    metadata: dict[str, torch.Tensor]


@dataclass
class IntegratorModelArgs:
    cfg: IntegratorCfg
    loss: nn.Module
    surrogates: dict[str, nn.Module]
    encoders: dict[str, nn.Module]


def _assemble_outputs(
    out: IntegratorBaseOutputs,
) -> dict[str, Any]:
    is_profile_output = isinstance(out.qp, ProfileSurrogateOutput)

    qp_mean = out.qp.mean_profile if is_profile_output else out.qp.mean

    base = {
        "rates": out.rates,
        "counts": out.counts,
        "mask": out.mask,
        "zp": out.zp,
        "qbg_mean": out.qbg.mean,
        "qbg_var": out.qbg.variance,
        "qp_mean": qp_mean,
        "qi_mean": out.qi.mean,
        "qi_var": out.qi.variance,
        "profile": qp_mean,
    }

    if is_profile_output:
        base["qp_loc"] = out.qp.loc
        base["qp_scale"] = out.qp.scale

    if out.metadata is None:
        return base

    # Storing the surrogate distribution parameters
    distribution_params = {
        "qbg_params": {
            name: getattr(out.qbg, name) for name in out.qbg.arg_constraints
        },
        "qi_params": {
            name: getattr(out.qi, name) for name in out.qi.arg_constraints
        },
    }

    # Update base dictionary
    base.update(out.metadata)
    base.update(distribution_params)

    return base


class IntensityScatterMixin:
    """End-of-epoch model `scale*I_h` vs DIALS `intensity.sum.value` scatter.

    A merge-quality readout shared by the merging integrators: each train batch
    stashes a subsample of (`scale*I_h`, DIALS intensity), and at epoch end a
    log-log scatter (plus the `intensity_logCC`) is logged to wandb and saved to
    `<run>/intensity_scatter/`. Host integrators call
    `self._init_intensity_scatter(cfg)` in `__init__` and, in the train branch of
    `_step`, `self._collect_intensity_scatter(outputs, metadata)`. The
    `on_train_epoch_end` here chains to `super()`, so list the mixin BEFORE the
    integrator base in the class bases.
    """

    def _init_intensity_scatter(self, cfg) -> None:
        self.log_intensity_scatter = bool(
            getattr(cfg, "log_intensity_scatter", False)
        )
        self._iscatter_model: list[Tensor] = []
        self._iscatter_dials: list[Tensor] = []

    def _collect_intensity_scatter(
        self, outputs: dict, metadata: dict, max_per_batch: int = 64
    ) -> None:
        """Stash a per-batch subsample of (model scale*I_h, DIALS intensity)."""
        with torch.no_grad():
            model_i = (
                outputs["scale"] * outputs["qi"].mean
            ).detach().float().cpu()
            dials_i = metadata["intensity.sum.value"].detach().float().cpu()
            n = model_i.shape[0]
            k = min(max_per_batch, n)
            if k <= 0:
                return
            idx = torch.randperm(n)[:k]
            self._iscatter_model.append(model_i[idx])
            self._iscatter_dials.append(dials_i[idx])

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        if not (self.log_intensity_scatter and self._iscatter_model):
            return
        import numpy as np

        model_i = torch.cat(self._iscatter_model).numpy()
        dials_i = torch.cat(self._iscatter_dials).numpy()
        self._iscatter_model.clear()
        self._iscatter_dials.clear()
        if len(model_i) > 5000:
            sel = np.random.choice(len(model_i), 5000, replace=False)
            model_i, dials_i = model_i[sel], dials_i[sel]
        self._plot_intensity_scatter(model_i, dials_i)

    def _plot_intensity_scatter(self, model_i, dials_i) -> None:
        """Log-log scatter of model scale*I_h vs DIALS intensity.sum.value."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception:
            return

        keep = (
            (model_i > 0)
            & (dials_i > 0)
            & np.isfinite(model_i)
            & np.isfinite(dials_i)
        )
        mi, di = model_i[keep], dials_i[keep]
        if len(mi) < 10:
            return
        log_cc = float(np.corrcoef(np.log(mi), np.log(di))[0, 1])

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(di, mi, s=4, alpha=0.3, edgecolors="none")
        lo, hi = float(min(di.min(), mi.min())), float(max(di.max(), mi.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("DIALS intensity.sum.value")
        ax.set_ylabel(r"model  scale $\cdot$ $I_h$")
        ax.set_title(
            f"epoch {self.current_epoch}  log-CC={log_cc:.3f}  n={len(mi)}"
        )
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()

        try:
            import wandb

            if self.logger is not None and hasattr(self.logger, "experiment"):
                self.logger.experiment.log(
                    {
                        "intensity_vs_dials": wandb.Image(fig),
                        "intensity_logCC": log_cc,
                        "epoch": self.current_epoch,
                    }
                )
        except Exception:
            pass
        try:
            from pathlib import Path

            save_dir = getattr(self.logger, "save_dir", None)
            out = Path(save_dir or ".") / "intensity_scatter"
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"epoch{self.current_epoch:04d}.png", dpi=110)
        except Exception:
            pass
        plt.close(fig)
