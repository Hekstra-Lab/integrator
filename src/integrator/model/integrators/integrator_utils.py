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


class ScatterLoggerMixin:
    """End-of-epoch log-log scatters of model vs DIALS quantities (merge-quality
    readouts), reusable by any integrator.

    Each train batch stashes a subsample of (model, DIALS) pairs for every enabled
    scatter; at epoch end each is plotted (with its log-CC), logged to wandb as
    `<name>_vs_dials` / `<name>_logCC`, and saved to `<run>/<name>_scatter/`.
    Built-in scatters (each gated on its `log_*_scatter` cfg flag AND the needed
    keys being present, so they no-op safely):

      - "intensity":  model `scale*I_h` (or `qi.mean` when there is no scale) vs
                      DIALS `intensity.sum.value`.
      - "background": model `qbg.mean * n_valid_pixels` vs DIALS
                      `background.sum.value`.

    Host integrators call `self._init_scatter_logger(cfg)` in `__init__` and, in
    the train branch of `_step`, `self._collect_scatters(outputs, metadata, mask)`.
    `on_train_epoch_end` chains to `super()`, so list the mixin BEFORE the
    integrator base. Add custom scatters with `self._stash_scatter(name, model,
    ref, xlabel, ylabel)`.
    """

    def _init_scatter_logger(self, cfg) -> None:
        self.log_intensity_scatter = bool(
            getattr(cfg, "log_intensity_scatter", False)
        )
        self.log_background_scatter = bool(
            getattr(cfg, "log_background_scatter", False)
        )
        # name -> [model_chunks, ref_chunks, xlabel, ylabel]
        self._scatter_buffers: dict[str, list] = {}

    def _stash_scatter(
        self,
        name: str,
        model_vals: Tensor,
        ref_vals: Tensor,
        xlabel: str,
        ylabel: str,
        max_per_batch: int = 64,
    ) -> None:
        """Stash a per-batch subsample of (model, reference) for scatter `name`."""
        with torch.no_grad():
            mi = model_vals.detach().float().reshape(-1).cpu()
            ri = ref_vals.detach().float().reshape(-1).cpu()
            n = min(mi.shape[0], ri.shape[0])
            k = min(max_per_batch, n)
            if k <= 0:
                return
            idx = torch.randperm(n)[:k]
            buf = self._scatter_buffers.setdefault(
                name, [[], [], xlabel, ylabel]
            )
            buf[0].append(mi[:n][idx])
            buf[1].append(ri[:n][idx])

    def _collect_intensity_scatter(
        self, outputs: dict, metadata: dict, max_per_batch: int = 64
    ) -> None:
        """Stash (model scale*I_h, DIALS intensity). Uses qi.mean if no scale."""
        if "intensity.sum.value" not in metadata or "qi" not in outputs:
            return
        model_i = outputs["qi"].mean
        scale = outputs.get("scale")
        if scale is not None:
            model_i = scale * model_i
        self._stash_scatter(
            "intensity",
            model_i,
            metadata["intensity.sum.value"],
            "DIALS intensity.sum.value",
            r"model  scale $\cdot$ $I_h$",
            max_per_batch,
        )

    def _collect_background_scatter(
        self, outputs: dict, metadata: dict, mask: Tensor, max_per_batch: int = 64
    ) -> None:
        """Stash (model total background, DIALS background.sum.value).

        The model background is the per-pixel rate `qbg.mean` times the number of
        valid pixels, to match the DIALS integrated `background.sum.value`.
        """
        if "background.sum.value" not in metadata or "qbg" not in outputs:
            return
        bg = outputs["qbg"].mean
        npix = mask.reshape(mask.shape[0], -1).float().sum(-1).to(bg.device)
        self._stash_scatter(
            "background",
            bg * npix,
            metadata["background.sum.value"],
            "DIALS background.sum.value",
            "model  bg total",
            max_per_batch,
        )

    def _collect_scatters(
        self, outputs: dict, metadata: dict, mask: Tensor | None = None
    ) -> None:
        """Collect every enabled+available scatter. Call in the train `_step`."""
        if getattr(self, "log_intensity_scatter", False):
            self._collect_intensity_scatter(outputs, metadata)
        if getattr(self, "log_background_scatter", False) and mask is not None:
            self._collect_background_scatter(outputs, metadata, mask)

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        import numpy as np

        for name, buf in list(getattr(self, "_scatter_buffers", {}).items()):
            model_list, ref_list, xlabel, ylabel = buf
            if not model_list:
                continue
            model_i = torch.cat(model_list).numpy()
            ref_i = torch.cat(ref_list).numpy()
            if len(model_i) > 5000:
                sel = np.random.choice(len(model_i), 5000, replace=False)
                model_i, ref_i = model_i[sel], ref_i[sel]
            self._plot_scatter(name, model_i, ref_i, xlabel, ylabel)
        if hasattr(self, "_scatter_buffers"):
            self._scatter_buffers.clear()

    def _plot_scatter(
        self, name: str, model_i, ref_i, xlabel: str, ylabel: str
    ) -> None:
        """Log-log scatter of a model quantity vs its DIALS reference."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception:
            return

        keep = (
            (model_i > 0)
            & (ref_i > 0)
            & np.isfinite(model_i)
            & np.isfinite(ref_i)
        )
        mi, di = model_i[keep], ref_i[keep]
        if len(mi) < 10:
            return
        log_cc = float(np.corrcoef(np.log(mi), np.log(di))[0, 1])

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(di, mi, s=4, alpha=0.3, edgecolors="none")
        lo, hi = float(min(di.min(), mi.min())), float(max(di.max(), mi.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
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
                        f"{name}_vs_dials": wandb.Image(fig),
                        f"{name}_logCC": log_cc,
                        "epoch": self.current_epoch,
                    }
                )
        except Exception:
            pass
        try:
            from pathlib import Path

            save_dir = getattr(self.logger, "save_dir", None)
            out = Path(save_dir or ".") / f"{name}_scatter"
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"epoch{self.current_epoch:04d}.png", dpi=110)
        except Exception:
            pass
        plt.close(fig)
