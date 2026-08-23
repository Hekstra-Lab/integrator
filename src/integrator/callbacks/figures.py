"""Callbacks that collect the per-epoch state behind the training figures.

Each callback owns one figure family and writes into a single `figures/`
directory using the layout in `integrator.reporting.figure_data`, so the
post-hoc checkpoint replay in `scripts/make_training_figures.py` can
produce or extend the same dumps.

All three are opt-in: training never depends on them, and a failure
inside a plotting step is logged rather than raised.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from torch.utils.data import Subset
from torch.utils.data._utils.collate import default_collate

from integrator.reporting.figure_data import (
    BasisRecorder,
    LatentRecorder,
    TrackedRecorder,
    dials_snr,
    select_tracked,
)

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _frozen_rng(seed: int = 0):
    """Run with a fixed RNG and restore the training RNG state afterwards.

    The tracked-shoebox pass draws Monte Carlo samples. Reusing the same
    draw every epoch keeps the movie's frame-to-frame change attributable
    to the model rather than to sampling noise, and restoring the state
    keeps the pass from perturbing training.
    """
    cpu_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        torch.manual_seed(seed)
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def _val_dataset(trainer):
    """Return `(base_dataset, indices)` for the validation split."""
    loaders = trainer.val_dataloaders
    if isinstance(loaders, list | tuple):
        loaders = loaders[0]
    dataset = loaders.dataset
    if isinstance(dataset, Subset):
        return dataset.dataset, np.asarray(dataset.indices, dtype=int)
    return dataset, np.arange(len(dataset), dtype=int)


def _chunked_counts_snr(base, indices, chunk: int = 8192):
    """SNR proxy from raw counts, read in chunks to avoid copying the split."""
    from integrator.reporting.figure_data import _counts_snr

    parts = []
    for start in range(0, len(indices), chunk):
        sel = indices[start : start + chunk]
        counts = np.asarray(base.counts[sel], dtype=np.float32)
        masks = np.asarray(base.masks[sel], dtype=np.float32)
        parts.append(_counts_snr(counts, masks))
    return np.concatenate(parts) if parts else np.zeros(0)


def _profile_decoder(pl_module):
    """The learned-basis profile decoder, or None for other profile models."""
    surrogates = getattr(pl_module, "surrogates", None)
    if surrogates is None or "qp" not in surrogates:
        return None
    return getattr(surrogates["qp"], "decoder", None)


class TrackedShoeboxLogger(Callback):
    """Replay a fixed set of shoeboxes each epoch and record the predictions.

    The tracked set is chosen once, at the first validation epoch, and
    spans the weak, medium, and strong I/sigma regimes so the figure
    shows how the posterior behaves where the data is informative and
    where it is not.

    Args:
        out_dir: Directory for the figure dumps.
        n_per_regime: Shoeboxes tracked per regime.
        every_n_epochs: Record every n-th epoch.
        plot: Render the figures at the end of training.
        animate: Also write the training movie as a GIF.
    """

    def __init__(
        self,
        out_dir,
        n_per_regime: int = 4,
        every_n_epochs: int = 1,
        plot: bool = True,
        animate: bool = True,
    ):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.n_per_regime = n_per_regime
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.plot = plot
        self.animate = animate
        self._recorder: TrackedRecorder | None = None
        self._batch = None

    def _build(self, trainer, pl_module) -> None:
        base, indices = _val_dataset(trainer)
        reference = {
            k: np.asarray(v)[indices] for k, v in base.reference.items()
        }
        counts = None
        if dials_snr(reference) is None:
            snr = _chunked_counts_snr(base, indices)
            reference["intensity.sum.value"] = snr
            reference["intensity.sum.variance"] = np.ones_like(snr)
            logger.info("no DIALS columns; tracking on a counts-based SNR")
        selection = select_tracked(
            reference, counts=counts, n_per_regime=self.n_per_regime
        )
        picked = indices[selection["index"]]
        self._batch = default_collate([base[int(i)] for i in picked])
        selection["index"] = picked
        tracked_counts = self._batch[0].float().numpy()
        tracked_masks = self._batch[2].float().numpy()
        self._recorder = TrackedRecorder(
            selection,
            tracked_counts,
            tracked_masks,
            pl_module.shoebox_shape,
        )
        logger.info(
            "tracking %d shoeboxes: %s",
            len(picked),
            ", ".join(
                f"{r}:{int(i)}"
                for r, i in zip(
                    selection["regime"], selection["refl_id"], strict=True
                )
            ),
        )

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Record one epoch of predictions for the tracked shoeboxes."""
        if trainer.sanity_checking:
            return
        epoch = int(trainer.current_epoch)
        if epoch % self.every_n_epochs:
            return
        if self._recorder is None:
            try:
                self._build(trainer, pl_module)
            except Exception as exc:  # noqa: BLE001 - never break a run
                logger.warning("tracked-shoebox selection failed: %s", exc)
                self.every_n_epochs = 10**9
                return

        was_training = pl_module.training
        pl_module.eval()
        try:
            with torch.no_grad(), _frozen_rng():
                batch = _to_device(self._batch, pl_module.device)
                out = pl_module(*batch)["forward_out"]
                self._recorder.record(epoch, out)
        finally:
            pl_module.train(was_training)
        self._recorder.save(self.out_dir)

    def on_fit_end(self, trainer, pl_module) -> None:
        """Write the dumps and render the tracked-shoebox figures."""
        if self._recorder is None or not self._recorder.epochs:
            return
        self._recorder.save(self.out_dir)
        if not self.plot:
            return
        try:
            from integrator.reporting.plot_jobs import (
                render_explain,
                render_tracked,
            )

            render_tracked(self.out_dir, animate=self.animate)
            render_explain(self.out_dir)
        except Exception as exc:  # noqa: BLE001 - figures are not the run
            logger.warning("tracked-shoebox figures failed: %s", exc)


class ProfileBasisLogger(Callback):
    """Snapshot the learned profile decoder once per epoch.

    No-op when the profile surrogate has no linear decoder, which is the
    case for the Dirichlet profile.
    """

    def __init__(
        self,
        out_dir,
        every_n_epochs: int = 1,
        plot: bool = True,
        animate: bool = True,
    ):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.plot = plot
        self.animate = animate
        self._recorder: BasisRecorder | None = None

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Store this epoch's decoder weight and bias."""
        if trainer.sanity_checking:
            return
        epoch = int(trainer.current_epoch)
        if epoch % self.every_n_epochs:
            return
        decoder = _profile_decoder(pl_module)
        if decoder is None:
            return
        if self._recorder is None:
            self._recorder = BasisRecorder(pl_module.shoebox_shape)
        self._recorder.record(
            epoch, decoder.weight.detach(), decoder.bias.detach()
        )
        self._recorder.save(self.out_dir)

    def on_fit_end(self, trainer, pl_module) -> None:
        """Write the snapshots and render the basis figures."""
        if self._recorder is None or not self._recorder.epochs:
            return
        self._recorder.save(self.out_dir)
        if not self.plot:
            return
        try:
            from integrator.reporting.plot_jobs import render_basis

            render_basis(self.out_dir, animate=self.animate)
        except Exception as exc:  # noqa: BLE001 - figures are not the run
            logger.warning("basis figures failed: %s", exc)


class LatentSpaceLogger(Callback):
    """Collect profile latents and covariates over the validation split.

    Args:
        out_dir: Directory for the figure dumps.
        every_n_epochs: Collect every n-th epoch.
        max_points: Cap on reflections kept per epoch.
        n_clusters: k for the k-means panel.
        plot: Render the figures at the end of training.
    """

    def __init__(
        self,
        out_dir,
        every_n_epochs: int = 5,
        max_points: int = 20000,
        n_clusters: int = 6,
        plot: bool = True,
    ):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.max_points = max_points
        self.n_clusters = n_clusters
        self.plot = plot
        self._recorder = LatentRecorder(max_points=max_points)
        self._active = False

    def _collecting(self, trainer) -> bool:
        epoch = int(trainer.current_epoch)
        last = epoch == max(trainer.max_epochs - 1, 0)
        return last or epoch % self.every_n_epochs == 0

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        """Decide whether this epoch is collected, and clear the buffer."""
        self._active = not trainer.sanity_checking and self._collecting(
            trainer
        )
        if self._active:
            self._recorder.reset()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        """Buffer one batch of latents plus its metadata."""
        if not self._active or not isinstance(outputs, dict):
            return
        forward_out = outputs.get("forward_out")
        if forward_out is None:
            return
        metadata = batch[3] if len(batch) > 3 else None
        self._recorder.add(forward_out, metadata)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Write this epoch's latents."""
        if not self._active:
            return
        path = self._recorder.save(self.out_dir, int(trainer.current_epoch))
        if path is not None:
            logger.info(
                "latents: %d points -> %s", self._recorder.n_points, path.name
            )
        self._active = False

    def on_fit_end(self, trainer, pl_module) -> None:
        """Render the latent-space figures from the last collected epoch."""
        if not self.plot:
            return
        decoder = _profile_decoder(pl_module)
        weight = bias = None
        if decoder is not None:
            weight = decoder.weight.detach().cpu().numpy()
            bias = decoder.bias.detach().cpu().numpy()
        try:
            from integrator.reporting.plot_jobs import render_latent

            render_latent(
                self.out_dir,
                weight=weight,
                bias=bias,
                shape=pl_module.shoebox_shape,
                n_clusters=self.n_clusters,
            )
        except Exception as exc:  # noqa: BLE001 - figures are not the run
            logger.warning("latent figures failed: %s", exc)
