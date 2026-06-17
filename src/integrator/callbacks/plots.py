from collections import defaultdict

from pytorch_lightning.callbacks import Callback

from .run_logger import get_run_logger

# Per-reflection loss components emitted by the integrator's _step.
_TERMS = ("loss", "nll", "kl", "kl_prf", "kl_i", "kl_bg")
_SPLITS = ("train", "val")


class LossCurveLogger(Callback):
    """Plot train/val total loss and split-ELBO curves each epoch."""

    def __init__(self):
        super().__init__()
        self._hist: list[dict] = []
        self._acc: dict[str, dict] = {}
        self._reset_epoch()

    def _reset_epoch(self):
        self._acc = {s: defaultdict(list) for s in _SPLITS}

    def _collect(self, split, outputs):
        lc = (
            outputs.get("loss_components")
            if isinstance(outputs, dict)
            else None
        )
        if not lc:
            return
        for term in _TERMS:
            v = lc.get(term)
            if v is not None:
                self._acc[split][term].append(float(v))

    def on_train_epoch_start(self, trainer, pl_module):
        self._reset_epoch()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if not trainer.sanity_checking:
            self._collect("train", outputs)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if not trainer.sanity_checking:
            self._collect("val", outputs)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        row: dict = {"epoch": int(trainer.current_epoch)}
        for split in _SPLITS:
            for term in _TERMS:
                vals = self._acc[split].get(term)
                if vals:
                    row[f"{split}_{term}"] = sum(vals) / len(vals)
        if len(row) == 1:
            return
        self._hist.append(row)
        self._plot(trainer)

    def _plot(self, trainer):
        import matplotlib.pyplot as plt
        import polars as pl

        rl = get_run_logger(self, trainer)
        epochs = [r["epoch"] for r in self._hist]

        def series(key):
            return [r.get(key) for r in self._hist]

        fig, ax = plt.subplots(figsize=(4, 3), dpi=90)
        for split, style in (("train", "-"), ("val", "--")):
            ys = series(f"{split}_loss")
            if any(y is not None for y in ys):
                ax.plot(epochs, ys, style, label=split)
        ax.set_xlabel("epoch")
        ax.set_ylabel("total loss (ELBO)")
        ax.set_title("loss")
        ax.legend()
        rl.log_figure("loss_total", fig)

        panel = ("nll", "kl_prf", "kl_i", "kl_bg")
        fig2, axes = plt.subplots(2, 2, figsize=(7, 5), dpi=90)
        for ax_, term in zip(axes.ravel(), panel, strict=False):
            for split, style in (("train", "-"), ("val", "--")):
                ys = series(f"{split}_{term}")
                if any(y is not None for y in ys):
                    ax_.plot(epochs, ys, style, label=split)
            ax_.set_title(term)
            ax_.set_xlabel("epoch")
            ax_.legend(fontsize=7)
        fig2.tight_layout()
        rl.log_figure("loss_terms", fig2)

        rl.log_table("loss_history", pl.DataFrame(self._hist))


class PredictionScatterLogger(Callback):
    """Scatter model vs DIALS intensity/background for a subset of val reflections.

    Off by default (opt-in).

    y-axis = DIALS, x-axis = model
    """

    def __init__(self, max_points: int = 2000, every_n_epochs: int = 1):
        super().__init__()
        self.max_points = max_points
        self.every_n_epochs = every_n_epochs
        self._buf: dict[str, list] = defaultdict(list)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._buf = defaultdict(list)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if trainer.sanity_checking or not isinstance(outputs, dict):
            return
        out = outputs.get("forward_out")
        if out is None or "qi_mean" not in out or "qbg_mean" not in out:
            return
        if not isinstance(batch, (tuple, list)) or len(batch) < 4:
            return
        metadata = batch[3]
        di = metadata.get("intensity.prf.value")
        if di is None:
            di = metadata.get("intensity.sum.value")
        dbg = metadata.get("background.mean")
        if dbg is None:
            dbg = metadata.get("background.sum.value")
        if di is None or dbg is None:
            return

        self._buf["qi_mean"].append(out["qi_mean"].detach().flatten().cpu())
        self._buf["qbg_mean"].append(out["qbg_mean"].detach().flatten().cpu())
        self._buf["dials_intensity"].append(di.detach().flatten().cpu())
        self._buf["dials_background"].append(dbg.detach().flatten().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not self._buf.get("qi_mean"):
            return
        epoch = int(trainer.current_epoch)
        if self.every_n_epochs and epoch % self.every_n_epochs != 0:
            return

        import polars as pl
        import torch

        cols = {k: torch.cat(v).numpy() for k, v in self._buf.items()}
        n = len(cols["qi_mean"])
        if n > self.max_points:
            idx = torch.randperm(n)[: self.max_points].numpy()
            cols = {k: v[idx] for k, v in cols.items()}
        df = pl.DataFrame(cols)

        rl = get_run_logger(self, trainer)
        rl.log_scatter(
            "val_intensity_model_vs_dials",
            df,
            x="qi_mean",
            y="dials_intensity",
            step=epoch,
            loglog=True,
        )
        rl.log_scatter(
            "val_background_model_vs_dials",
            df,
            x="qbg_mean",
            y="dials_background",
            step=epoch,
            loglog=False,
        )


class WilsonParamLogger(Callback):
    """Log the learned Wilson prior parameters each epoch.

    Tracks the B-factor (always) and the scale: a scalar G for the
    monochromatic loss, or the full G(lambda) Chebyshev spectrum for the
    polychromatic loss. No-op when the loss is not a Wilson loss.
    """

    def __init__(self, n_lambda: int = 100):
        super().__init__()
        self.n_lambda = n_lambda
        self._hist: list[dict] = []
        self._spectra: list[tuple] = []  # (epoch, lambdas, G)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        loss = getattr(pl_module, "loss", None)
        if loss is None or not hasattr(loss, "get_B"):
            return  # not a Wilson loss

        import torch
        import torch.nn.functional as F

        epoch = int(trainer.current_epoch)
        row: dict = {"epoch": epoch}
        with torch.no_grad():
            row["B"] = float(loss.get_B())

            spectrum = getattr(loss, "spectrum", None)
            if spectrum is not None:
                lam_lo = float(spectrum.lam_mid - spectrum.lam_scale)
                lam_hi = float(spectrum.lam_mid + spectrum.lam_scale)
                lam = torch.linspace(lam_lo, lam_hi, self.n_lambda)
                g = spectrum.get_log_G(lam).exp()
                self._spectra.append(
                    (epoch, lam.cpu().numpy(), g.cpu().numpy())
                )
                row["G_mean"] = float(g.mean())
            elif hasattr(loss, "get_G"):
                row["G"] = float(loss.get_G())

            if getattr(loss, "learn_concentration", False) and hasattr(
                loss, "log_alpha_per_group"
            ):
                row["alpha_mean"] = float(
                    F.softplus(loss.log_alpha_per_group).mean()
                )

        self._hist.append(row)
        self._plot(trainer)

    def _plot(self, trainer):
        import matplotlib.pyplot as plt
        import polars as pl

        rl = get_run_logger(self, trainer)
        epochs = [r["epoch"] for r in self._hist]

        rl.log_scalars(
            {f"wilson/{k}": v for k, v in self._hist[-1].items() if k != "epoch"},
            step=self._hist[-1]["epoch"],
        )

        def series(key):
            return [r.get(key) for r in self._hist]

        g_key = "G" if any("G" in r for r in self._hist) else "G_mean"
        fig, axes = plt.subplots(1, 2, figsize=(7, 3), dpi=90)
        axes[0].plot(epochs, series("B"), "-", marker=".")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("B (Å²)")
        axes[0].set_title("Wilson B-factor")
        if any(g_key in r for r in self._hist):
            axes[1].plot(epochs, series(g_key), "-", marker=".")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel(g_key)
        axes[1].set_title("Wilson scale")
        fig.tight_layout()
        rl.log_figure("wilson_params", fig)

        # G(lambda) spectrum and its evolution (polychromatic only)
        if self._spectra:
            n = len(self._spectra)
            fig2, ax = plt.subplots(figsize=(4, 3), dpi=90)
            for i, (ep, lam, g) in enumerate(self._spectra):
                label = f"ep {ep}" if i in (0, n - 1) else None
                ax.plot(
                    lam, g, color=plt.cm.viridis(i / max(n - 1, 1)), label=label
                )
            ax.set_xlabel("wavelength λ (Å)")
            ax.set_ylabel("G(λ)")
            ax.set_title("Learned spectrum G(λ)")
            ax.legend(fontsize=8)
            rl.log_figure("wilson_spectrum", fig2)

            ep, lam, g = self._spectra[-1]
            rl.log_table(
                "wilson_spectrum", pl.DataFrame({"wavelength": lam, "G": g})
            )

        rl.log_table("wilson_params", pl.DataFrame(self._hist))
