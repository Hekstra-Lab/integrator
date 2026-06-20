"""End-of-epoch model-vs-DIALS scatter logging for the scaling/merging models.

A self-contained `LightningModule` mixin (no `BaseIntegrator` dependency). Each
train batch stashes a subsample of (model, DIALS) pairs for every enabled
scatter; at epoch end each is plotted with its log-CC, logged to W&B as
`<name>_vs_dials` / `<name>_logCC`, and saved under `<run>/<name>_scatter/`.

Built-in scatters (each gated on its `log_*_scatter` cfg flag AND the needed
keys being present, so they no-op safely):

  - "intensity":  model `scale * qi.mean` vs DIALS `intensity.sum.value`.
  - "intensity_prf": same model x vs DIALS `intensity.prf.value`.
  - "intensity_var": model `scale^2 * Var[I]` vs DIALS `intensity.prf.variance`.
  - "background": model `qbg.mean` (per-pixel) vs the per-shoebox masked median.

Host modules call `self._init_scatter_logger(cfg)` in `__init__` and, in the
train branch of `_step`, `self._collect_scatters(outputs, metadata, mask,
counts)`. `on_train_epoch_end` chains to `super()`, so list the mixin BEFORE the
LightningModule base.
"""

import torch
from torch import Tensor


class ScatterLoggerMixin:
    """Log-log scatters of model vs DIALS quantities (merge-quality readouts)."""

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

    def _scaled_model_intensity(self, outputs: dict) -> Tensor:
        """Model observed intensity = scale * qi.mean (qi.mean if no scale)."""
        model_i = outputs["qi"].mean
        scale = outputs.get("scale")
        return scale * model_i if scale is not None else model_i

    def _collect_intensity_scatter(
        self, outputs: dict, metadata: dict, max_per_batch: int = 64
    ) -> None:
        """Stash (model scale*I, DIALS intensity.sum.value). Model on x."""
        if "intensity.sum.value" not in metadata or "qi" not in outputs:
            return
        self._stash_scatter(
            "intensity",
            self._scaled_model_intensity(outputs),
            metadata["intensity.sum.value"],
            r"model  scale $\cdot$ $I$",
            "DIALS intensity.sum.value",
            max_per_batch,
        )

    def _collect_intensity_prf_scatter(
        self, outputs: dict, metadata: dict, max_per_batch: int = 64
    ) -> None:
        """Stash (model scale*I, DIALS intensity.prf.value), model on x."""
        if "intensity.prf.value" not in metadata or "qi" not in outputs:
            return
        self._stash_scatter(
            "intensity_prf",
            self._scaled_model_intensity(outputs),
            metadata["intensity.prf.value"],
            r"model  scale $\cdot$ $I$",
            "DIALS intensity.prf.value",
            max_per_batch,
        )

    def _collect_variance_scatter(
        self, outputs: dict, metadata: dict, max_per_batch: int = 64
    ) -> None:
        """Stash (model scale^2 * Var[I], DIALS intensity.prf.variance) -- a
        sigma(I) calibration check (Var[s*I] = s^2 Var[I]), model on x."""
        if "intensity.prf.variance" not in metadata or "qi" not in outputs:
            return
        model_v = outputs["qi"].variance
        scale = outputs.get("scale")
        if scale is not None:
            model_v = scale.pow(2) * model_v
        self._stash_scatter(
            "intensity_var",
            model_v,
            metadata["intensity.prf.variance"],
            r"model  $s^2\,\mathrm{Var}[I]$",
            "DIALS intensity.prf.variance",
            max_per_batch,
        )

    def _collect_background_scatter(
        self,
        outputs: dict,
        metadata: dict,
        mask: Tensor,
        counts: Tensor,
        max_per_batch: int = 64,
    ) -> None:
        """Stash (model per-pixel bg, data per-pixel bg) -- a per-pixel check.

        The model `qbg.mean` is a per-pixel background rate, so the reference is
        a per-pixel background from the data: the robust per-shoebox median of
        the masked counts (the small Bragg peak does not move the median in a
        mostly-background box).
        """
        if "qbg" not in outputs or counts is None:
            return
        bg = outputs["qbg"].mean.reshape(-1)
        cm = counts.reshape(counts.shape[0], -1).clamp(min=0).float()
        mk = mask.reshape(mask.shape[0], -1).float().to(cm.device)
        valid = mk > 0
        filled = torch.where(valid, cm, torch.full_like(cm, float("inf")))
        n_valid = valid.sum(-1).clamp(min=1).long()
        sorted_c, _ = filled.sort(dim=-1)
        med_idx = ((n_valid - 1) // 2).unsqueeze(1)
        data_bg = sorted_c.gather(1, med_idx).squeeze(1)
        self._stash_scatter(
            "background",
            bg,
            data_bg,
            "model bg/pixel (qbg.mean)",
            "data bg/pixel (shoebox median)",
            max_per_batch,
        )

    def _collect_scatters(
        self,
        outputs: dict,
        metadata: dict,
        mask: Tensor | None = None,
        counts: Tensor | None = None,
    ) -> None:
        """Collect every enabled+available scatter. Call in the train `_step`."""
        if getattr(self, "log_intensity_scatter", False):
            self._collect_intensity_scatter(outputs, metadata)
            self._collect_intensity_prf_scatter(outputs, metadata)
            self._collect_variance_scatter(outputs, metadata)
        if (
            getattr(self, "log_background_scatter", False)
            and mask is not None
            and counts is not None
        ):
            self._collect_background_scatter(outputs, metadata, mask, counts)

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
        """Log-log scatter of a model quantity (x) vs its DIALS reference (y)."""
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
        ax.scatter(mi, di, s=4, alpha=0.3, edgecolors="none")  # model x, DIALS y
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
