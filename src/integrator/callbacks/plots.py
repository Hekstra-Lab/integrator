from collections import defaultdict
from pathlib import Path

from pytorch_lightning.callbacks import Callback

from .run_logger import get_run_logger

# Per-reflection loss components emitted by the integrator's _step.
_TERMS = ("loss", "nll", "kl", "kl_prf", "kl_i", "kl_bg")
_SPLITS = ("train", "val")


class LossCurveLogger(Callback):
    """Plot train/val total loss and split-ELBO curves each epoch."""

    def __init__(self, out_dir=None):
        super().__init__()
        self.out_dir = Path(out_dir) if out_dir is not None else None
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
            value = lc.get(term)
            if value is not None:
                self._acc[split][term].append(float(value))

    def on_train_epoch_start(self, trainer, pl_module):
        self._reset_epoch()

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        if not trainer.sanity_checking:
            self._collect("train", outputs)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if not trainer.sanity_checking:
            self._collect("val", outputs)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        row: dict = {"epoch": int(trainer.current_epoch)}

        for split in _SPLITS:
            for term in _TERMS:
                values = self._acc[split].get(term)

                if values:
                    row[f"{split}_{term}"] = sum(values) / len(values)

        if len(row) == 1:
            return

        self._hist.append(row)
        self._plot(trainer)

    def _plot(self, trainer):
        import matplotlib.pyplot as plt
        import polars as pl

        run_logger = get_run_logger(self, trainer)
        epochs = [row["epoch"] for row in self._hist]

        def series(key):
            return [row.get(key) for row in self._hist]

        fig, ax = plt.subplots(figsize=(4, 3), dpi=90)

        for split, style in (("train", "-"), ("val", "--")):
            values = series(f"{split}_loss")

            if any(value is not None for value in values):
                ax.plot(
                    epochs,
                    values,
                    style,
                    label=split,
                )

        ax.set_xlabel("epoch")
        ax.set_ylabel("total loss (ELBO)")
        ax.set_title("loss")
        ax.legend()

        run_logger.log_figure("loss_total", fig)

        panel = ("nll", "kl_prf", "kl_i", "kl_bg")

        fig2, axes = plt.subplots(
            2,
            2,
            figsize=(7, 5),
            dpi=90,
        )

        for axis, term in zip(
            axes.ravel(),
            panel,
            strict=False,
        ):
            for split, style in (("train", "-"), ("val", "--")):
                values = series(f"{split}_{term}")

                if any(value is not None for value in values):
                    axis.plot(
                        epochs,
                        values,
                        style,
                        label=split,
                    )

            axis.set_title(term)
            axis.set_xlabel("epoch")
            axis.legend(fontsize=7)

        fig2.tight_layout()
        run_logger.log_figure("loss_terms", fig2)

        run_logger.log_table(
            "loss_history",
            pl.DataFrame(self._hist),
        )


class PredictionScatterLogger(Callback):
    """Scatter model vs DIALS intensity/background for a subset of val reflections.

    Off by default (opt-in).

    y-axis = DIALS, x-axis = model
    """

    def __init__(
        self,
        out_dir=None,
        max_points: int = 2000,
        every_n_epochs: int = 1,
    ):
        super().__init__()

        self.out_dir = (
            Path(out_dir)
            if out_dir is not None
            else None
        )
        self.max_points = max_points
        self.every_n_epochs = every_n_epochs
        self._buf: dict[str, list] = defaultdict(list)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._buf = defaultdict(list)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if trainer.sanity_checking:
            return

        if not isinstance(outputs, dict):
            return

        output = outputs.get("forward_out")

        if output is None:
            return

        if "qi_mean" not in output or "qbg_mean" not in output:
            return

        if not isinstance(batch, (tuple, list)) or len(batch) < 4:
            return

        metadata = batch[3]

        dials_intensity = metadata.get("intensity.prf.value")

        if dials_intensity is None:
            dials_intensity = metadata.get("intensity.sum.value")

        dials_background = metadata.get("background.mean")

        if dials_background is None:
            dials_background = metadata.get("background.sum.value")

        if dials_intensity is None or dials_background is None:
            return

        self._buf["qi_mean"].append(
            output["qi_mean"].detach().flatten().cpu()
        )
        self._buf["qbg_mean"].append(
            output["qbg_mean"].detach().flatten().cpu()
        )
        self._buf["dials_intensity"].append(
            dials_intensity.detach().flatten().cpu()
        )
        self._buf["dials_background"].append(
            dials_background.detach().flatten().cpu()
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        if not self._buf.get("qi_mean"):
            return

        epoch = int(trainer.current_epoch)

        if (
            self.every_n_epochs
            and epoch % self.every_n_epochs != 0
        ):
            return

        import polars as pl
        import torch

        columns = {
            key: torch.cat(values).numpy()
            for key, values in self._buf.items()
        }

        n_points = len(columns["qi_mean"])

        if n_points > self.max_points:
            indices = torch.randperm(n_points)[
                : self.max_points
            ].numpy()

            columns = {
                key: values[indices]
                for key, values in columns.items()
            }

        dataframe = pl.DataFrame(columns)

        run_logger = get_run_logger(self, trainer)

        run_logger.log_scatter(
            "val_intensity_model_vs_dials",
            dataframe,
            x="qi_mean",
            y="dials_intensity",
            step=epoch,
            loglog=True,
        )

        run_logger.log_scatter(
            "val_background_model_vs_dials",
            dataframe,
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

    def __init__(
        self,
        out_dir=None,
        n_lambda: int = 100,
    ):
        super().__init__()

        self.out_dir = (
            Path(out_dir)
            if out_dir is not None
            else None
        )
        self.n_lambda = n_lambda
        self._hist: list[dict] = []
        self._spectra: list[tuple] = []

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

            # Per-image B/G: mean, std, min, max

            if hasattr(loss, "get_wilson_stats"):
                stats = loss.get_wilson_stats()

                for key, value in stats.items():
                    row[key] = float(value)

            # Existing global B/G behavior
            else:
                row["B"] = float(loss.get_B())

            spectrum = getattr(loss, "spectrum", None)

            if spectrum is not None:
                lam_lo = float(
                    spectrum.lam_mid - spectrum.lam_scale
                )
                lam_hi = float(
                    spectrum.lam_mid + spectrum.lam_scale
                )

                lam = torch.linspace(
                    lam_lo,
                    lam_hi,
                    self.n_lambda,
                )

                g = spectrum.get_log_G(lam).exp()

                self._spectra.append(
                    (
                        epoch,
                        lam.cpu().numpy(),
                        g.cpu().numpy(),
                    )
                )

                row["G_mean"] = float(g.mean())

            elif (
                not hasattr(loss, "get_wilson_stats")
                and hasattr(loss, "get_G")
            ):
                row["G"] = float(loss.get_G())

            if (
                getattr(loss, "learn_concentration", False)
                and hasattr(loss, "log_alpha_per_group")
            ):
                row["alpha_mean"] = float(
                    F.softplus(
                        loss.log_alpha_per_group
                    ).mean()
                )

        self._hist.append(row)
        self._plot(trainer)

    def _plot(self, trainer):
        import matplotlib.pyplot as plt
        import polars as pl

        run_logger = get_run_logger(self, trainer)

        epochs = [
            row["epoch"]
            for row in self._hist
        ]

        run_logger.log_scalars(
            {
                f"wilson/{key}": value
                for key, value in self._hist[-1].items()
                if key != "epoch"
            },
            step=self._hist[-1]["epoch"],
        )

        def series(key):
            return [
                row.get(key)
                for row in self._hist
            ]

        # PLOTTING
        """
        solid line: mean
        shaded band: mean ± standard deviation
        dashed lines: min and max
        """

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(7, 3),
            dpi=90,
        )

        # Per-image B/G statistics
        if any(
            "B_mean" in row
            for row in self._hist
        ):
            b_mean = series("B_mean")
            b_std = series("B_std")
            b_min = series("B_min")
            b_max = series("B_max")

            g_mean = series("G_mean")
            g_std = series("G_std")
            g_min = series("G_min")
            g_max = series("G_max")

            axes[0].plot(
                epochs,
                b_mean,
                "-",
                marker=".",
                label="mean",
            )

            axes[0].fill_between(
                epochs,
                [
                    mean - std
                    for mean, std in zip(
                        b_mean,
                        b_std,
                    )
                ],  # lower y
                [
                    mean + std
                    for mean, std in zip(
                        b_mean,
                        b_std,
                    )
                ],  # upper y
                alpha=0.2,
                label="mean ± std",
            )

            axes[0].plot(
                epochs,
                b_min,
                "--",
                label="min",
            )

            axes[0].plot(
                epochs,
                b_max,
                "--",
                label="max",
            )

            axes[1].plot(
                epochs,
                g_mean,
                "-",
                marker=".",
                label="mean",
            )

            axes[1].fill_between(
                epochs,
                [
                    mean - std
                    for mean, std in zip(
                        g_mean,
                        g_std,
                    )
                ],  # lower y
                [
                    mean + std
                    for mean, std in zip(
                        g_mean,
                        g_std,
                    )
                ],  # upper y
                alpha=0.2,
                label="mean ± std",
            )

            axes[1].plot(
                epochs,
                g_min,
                "--",
                label="min",
            )

            axes[1].plot(
                epochs,
                g_max,
                "--",
                label="max",
            )

            axes[0].legend(fontsize=7)
            axes[1].legend(fontsize=7)

        else:
            # Existing global B/G behavior

            g_key = (
                "G"
                if any(
                    "G" in row
                    for row in self._hist
                )
                else "G_mean"
            )

            axes[0].plot(
                epochs,
                series("B"),
                "-",
                marker=".",
            )

            if any(
                g_key in row
                for row in self._hist
            ):
                axes[1].plot(
                    epochs,
                    series(g_key),
                    "-",
                    marker=".",
                )

        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("B (Å²)")
        axes[0].set_title("Wilson B-factor")

        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("G")
        axes[1].set_title("Wilson scale")

        fig.tight_layout()

        run_logger.log_figure(
            "wilson_params",
            fig,
        )

        # G(lambda) spectrum and its evolution (polychromatic only)
        if self._spectra:
            n_spectra = len(self._spectra)

            fig2, ax = plt.subplots(
                figsize=(4, 3),
                dpi=90,
            )

            for index, (
                spectrum_epoch,
                wavelengths,
                g_values,
            ) in enumerate(self._spectra):
                label = (
                    f"ep {spectrum_epoch}"
                    if index in (0, n_spectra - 1)
                    else None
                )

                ax.plot(
                    wavelengths,
                    g_values,
                    color=plt.cm.viridis(
                        index / max(n_spectra - 1, 1)
                    ),
                    label=label,
                )

            ax.set_xlabel("wavelength λ (Å)")
            ax.set_ylabel("G(λ)")
            ax.set_title("Learned spectrum G(λ)")
            ax.legend(fontsize=8)

            run_logger.log_figure(
                "wilson_spectrum",
                fig2,
            )

            _, wavelengths, g_values = self._spectra[-1]

            run_logger.log_table(
                "wilson_spectrum",
                pl.DataFrame(
                    {
                        "wavelength": wavelengths,
                        "G": g_values,
                    }
                ),
            )

        run_logger.log_table(
            "wilson_params",
            pl.DataFrame(self._hist),
        )