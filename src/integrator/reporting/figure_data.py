"""Collect and persist the per-epoch state behind the training figures.

The recorders here are the single source of truth for the on-disk layout,
so the training callbacks in `integrator.callbacks.figures` and the
post-hoc checkpoint replay in `scripts/make_training_figures.py` produce
interchangeable dumps:

    figures/
      tracked_selection.json      tracked reflections and their regimes
      tracked_scalars.parquet     per (epoch, shoebox) posterior summaries
      tracked_arrays.npz          per-epoch profile/rate images
      basis_snapshots.npz         decoder W and b per epoch
      basis_diagnostics.parquet   spectrum, effective rank, step size
      latents_epoch_XXXX.parquet  profile latents joined to covariates

Nothing here imports matplotlib: collection runs inside training, where
plotting is optional.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

REGIMES = ("weak", "medium", "strong")

# Covariates carried alongside the latents; missing ones are skipped.
DEFAULT_COVARIATES = (
    "intensity.prf.value",
    "intensity.prf.variance",
    "intensity.sum.value",
    "background.mean",
    "d",
    "xyzcal.px.0",
    "xyzcal.px.1",
    "xyzcal.px.2",
    "panel",
    "profile.correlation",
    "partiality",
    "refl_ids",
)


def _np(x) -> np.ndarray:
    """Detach any tensor-like to a 1-D float numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _refl_id_key(reference: dict) -> str | None:
    for key in ("refl_ids", "refl_id"):
        if key in reference:
            return key
    return None


def dials_snr(reference: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (intensity, sigma) from the DIALS columns, prf before sum."""
    for val, var in (
        ("intensity.prf.value", "intensity.prf.variance"),
        ("intensity.sum.value", "intensity.sum.variance"),
    ):
        if val in reference and var in reference:
            i = _np(reference[val]).astype(float)
            sigma = np.sqrt(np.clip(_np(reference[var]).astype(float), 0, None))
            return i, sigma
    return None


def _counts_snr(counts: np.ndarray, masks: np.ndarray | None) -> np.ndarray:
    """Crude per-shoebox SNR from raw counts, used when DIALS columns are absent."""
    counts = np.asarray(counts, dtype=float)
    if masks is not None:
        counts = counts * np.asarray(masks, dtype=float)
    total = counts.sum(-1)
    bg = np.median(counts, axis=-1) * counts.shape[-1]
    return (total - bg) / np.sqrt(np.clip(total, 1.0, None))


def select_tracked(
    reference: dict,
    counts: np.ndarray | None = None,
    masks: np.ndarray | None = None,
    n_per_regime: int = 4,
    weak_max: float = 3.0,
    strong_min: float = 20.0,
) -> dict:
    """Choose a fixed set of shoeboxes spanning the weak/medium/strong regimes.

    Selection is deterministic: within each regime the reflections sitting
    at evenly spaced SNR quantiles are taken, so the panel spans the
    regime instead of showing three copies of its extreme.

    Args:
        reference: Metadata dict of per-reflection arrays or tensors.
        counts: Optional raw counts `(N, K)`, used only as an SNR fallback.
        masks: Optional masks `(N, K)` matching `counts`.
        n_per_regime: Shoeboxes to track per regime.
        weak_max: Upper I/sigma bound of the weak regime.
        strong_min: Lower I/sigma bound of the strong regime.

    Returns:
        Dict with `index`, `refl_id`, `regime`, `snr`, `intensity`, `sigma`.
    """
    pair = dials_snr(reference)
    if pair is not None:
        intensity, sigma = pair
        snr = intensity / np.where(sigma > 0, sigma, np.nan)
    elif counts is not None:
        intensity = np.asarray(counts, dtype=float).sum(-1)
        sigma = np.sqrt(np.clip(intensity, 1.0, None))
        snr = _counts_snr(counts, masks)
    else:
        raise ValueError("select_tracked needs DIALS columns or counts")

    finite = np.isfinite(snr)
    buckets = {
        "weak": finite & (snr < weak_max),
        "medium": finite & (snr >= weak_max) & (snr <= strong_min),
        "strong": finite & (snr > strong_min),
    }
    # Absolute thresholds can leave a bucket empty on unusual data; fall
    # back to global SNR terciles so the panel is always populated.
    if any(mask.sum() < n_per_regime for mask in buckets.values()):
        valid = np.flatnonzero(finite)
        order = valid[np.argsort(snr[valid])]
        thirds = np.array_split(order, 3)
        buckets = {}
        for name, part in zip(REGIMES, thirds, strict=False):
            mask = np.zeros_like(finite)
            mask[part] = True
            buckets[name] = mask
        logger.info("regime thresholds fell back to global SNR terciles")

    index: list[int] = []
    regime: list[str] = []
    for name in REGIMES:
        candidates = np.flatnonzero(buckets[name])
        if candidates.size == 0:
            continue
        ordered = candidates[np.argsort(snr[candidates])]
        take = min(n_per_regime, ordered.size)
        positions = np.linspace(0.15, 0.85, take) * (ordered.size - 1)
        picked = np.unique(np.round(positions).astype(int))
        for p in picked:
            index.append(int(ordered[p]))
            regime.append(name)

    idx = np.asarray(index, dtype=int)
    key = _refl_id_key(reference)
    refl_id = (
        _np(reference[key])[idx].astype(int)
        if key
        else idx.astype(int)
    )
    return {
        "index": idx,
        "refl_id": refl_id,
        "regime": regime,
        "snr": snr[idx],
        "intensity": intensity[idx],
        "sigma": sigma[idx],
    }


class TrackedRecorder:
    """Accumulate posterior summaries and images for the tracked shoeboxes."""

    def __init__(
        self,
        selection: dict,
        counts: np.ndarray,
        masks: np.ndarray,
        shape: tuple[int, ...],
    ):
        self.selection = selection
        self.counts = np.asarray(counts, dtype=np.float32)
        self.masks = np.asarray(masks, dtype=np.float32)
        self.shape = tuple(int(s) for s in shape)
        self.epochs: list[int] = []
        self._profiles: list[np.ndarray] = []
        self._rates: list[np.ndarray] = []
        self._rows: list[dict] = []

    def record(self, epoch: int, forward_out: dict) -> None:
        """Store one epoch of predictions for the tracked mini-batch."""
        profile = _np(forward_out["qp_mean"]).astype(np.float32)
        rates = _np(forward_out["rates"]).astype(np.float32)
        if rates.ndim == 3:  # (B, mc_samples, K)
            rates = rates.mean(1)
        qi = _np(forward_out["qi_mean"]).ravel()
        qi_var = _np(forward_out["qi_var"]).ravel()
        qbg = _np(forward_out["qbg_mean"]).ravel()
        qbg_var = _np(forward_out["qbg_var"]).ravel()

        self.epochs.append(int(epoch))
        self._profiles.append(profile)
        self._rates.append(rates)

        mask = self.masks
        resid = (self.counts - rates) * mask
        denom = np.clip(mask.sum(-1), 1.0, None)
        resid_rms = np.sqrt((resid**2).sum(-1) / denom)
        pearson = np.sqrt(np.clip(rates, 1e-6, None))
        z_rms = np.sqrt((((self.counts - rates) / pearson) ** 2 * mask).sum(-1) / denom)

        sel = self.selection
        for slot in range(len(qi)):
            self._rows.append(
                {
                    "epoch": int(epoch),
                    "slot": slot,
                    "refl_id": int(sel["refl_id"][slot]),
                    "regime": sel["regime"][slot],
                    "qi_mean": float(qi[slot]),
                    "qi_sd": float(np.sqrt(max(qi_var[slot], 0.0))),
                    "qbg_mean": float(qbg[slot]),
                    "qbg_sd": float(np.sqrt(max(qbg_var[slot], 0.0))),
                    "dials_i": float(sel["intensity"][slot]),
                    "dials_sigma": float(sel["sigma"][slot]),
                    "dials_snr": float(sel["snr"][slot]),
                    "resid_rms": float(resid_rms[slot]),
                    "z_rms": float(z_rms[slot]),
                    "profile_peak": float(profile[slot].max()),
                }
            )

    def save(self, out_dir: str | Path) -> None:
        """Write the selection, tidy scalars, and stacked image arrays."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self.epochs:
            logger.warning("TrackedRecorder has no epochs to save")
            return

        sel = {
            "index": [int(i) for i in self.selection["index"]],
            "refl_id": [int(i) for i in self.selection["refl_id"]],
            "regime": list(self.selection["regime"]),
            "snr": [float(v) for v in self.selection["snr"]],
            "intensity": [float(v) for v in self.selection["intensity"]],
            "sigma": [float(v) for v in self.selection["sigma"]],
            "shape": list(self.shape),
        }
        (out_dir / "tracked_selection.json").write_text(json.dumps(sel, indent=2))
        pl.DataFrame(self._rows).write_parquet(out_dir / "tracked_scalars.parquet")
        np.savez_compressed(
            out_dir / "tracked_arrays.npz",
            epochs=np.asarray(self.epochs, dtype=np.int32),
            counts=self.counts,
            masks=self.masks,
            profiles=np.stack(self._profiles),
            rates=np.stack(self._rates),
            shape=np.asarray(self.shape, dtype=np.int32),
        )


def effective_rank(singular_values: np.ndarray) -> float:
    """Entropy-based effective rank of a spectrum (Roy & Vetterli)."""
    sv = np.asarray(singular_values, dtype=float)
    sv = sv[sv > 0]
    if sv.size == 0:
        return 0.0
    p = sv / sv.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def subspace_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Largest principal angle in degrees between the column spans of a and b."""
    qa, _ = np.linalg.qr(np.asarray(a, dtype=float))
    qb, _ = np.linalg.qr(np.asarray(b, dtype=float))
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s.min(), -1.0, 1.0))))


def column_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Absolute cosine similarity between matching columns of a and b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    num = np.abs((a * b).sum(0))
    den = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    return num / np.where(den > 0, den, np.nan)


class BasisRecorder:
    """Snapshot the learned profile decoder and its convergence diagnostics."""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = tuple(int(s) for s in shape)
        self.epochs: list[int] = []
        self._weights: list[np.ndarray] = []
        self._biases: list[np.ndarray] = []

    def record(self, epoch: int, weight: np.ndarray, bias: np.ndarray) -> None:
        """Store the decoder weight `(K, d)` and bias `(K,)` for one epoch."""
        self.epochs.append(int(epoch))
        self._weights.append(_np(weight).astype(np.float32))
        self._biases.append(_np(bias).astype(np.float32))

    def diagnostics(self) -> pl.DataFrame:
        """Per-epoch spectrum, effective rank, step size, and distance to final."""
        rows = []
        final = self._weights[-1]
        for i, (epoch, w) in enumerate(zip(self.epochs, self._weights, strict=True)):
            sv = np.linalg.svd(w, compute_uv=False)
            row = {
                "epoch": epoch,
                "eff_rank": effective_rank(sv),
                "frob_norm": float(np.linalg.norm(w)),
                "angle_to_final_deg": subspace_angle_deg(w, final),
                "mean_cos_to_final": float(
                    np.nanmean(column_cosines(w, final))
                ),
            }
            for j, value in enumerate(sv):
                row[f"sv_{j}"] = float(value)
            if i > 0:
                prev = self._weights[i - 1]
                denom = float(np.linalg.norm(prev)) or 1.0
                row["rel_step"] = float(np.linalg.norm(w - prev) / denom)
            else:
                row["rel_step"] = float("nan")
            rows.append(row)
        return pl.DataFrame(rows)

    def save(self, out_dir: str | Path) -> None:
        """Write the snapshots and the diagnostics table."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self.epochs:
            logger.warning("BasisRecorder has no epochs to save")
            return
        np.savez_compressed(
            out_dir / "basis_snapshots.npz",
            epochs=np.asarray(self.epochs, dtype=np.int32),
            weights=np.stack(self._weights),
            biases=np.stack(self._biases),
            shape=np.asarray(self.shape, dtype=np.int32),
        )
        self.diagnostics().write_parquet(out_dir / "basis_diagnostics.parquet")


class LatentRecorder:
    """Accumulate profile latents and their covariates over one epoch."""

    def __init__(
        self,
        max_points: int = 20000,
        covariates: tuple[str, ...] = DEFAULT_COVARIATES,
    ):
        self.max_points = max_points
        self.covariates = covariates
        self._loc: list[np.ndarray] = []
        self._scale: list[np.ndarray] = []
        self._extra: dict[str, list[np.ndarray]] = {}
        self._n = 0

    @property
    def n_points(self) -> int:
        """Number of reflections accumulated so far this epoch."""
        return self._n

    def reset(self) -> None:
        """Drop everything accumulated for the current epoch."""
        self._loc.clear()
        self._scale.clear()
        self._extra.clear()
        self._n = 0

    def add(self, forward_out: dict, metadata: dict | None = None) -> None:
        """Add one batch of latents plus whichever covariates are present."""
        if self._n >= self.max_points or "qp_loc" not in forward_out:
            return
        loc = _np(forward_out["qp_loc"])
        room = self.max_points - self._n
        take = min(room, loc.shape[0])
        self._loc.append(loc[:take])
        if "qp_scale" in forward_out:
            self._scale.append(_np(forward_out["qp_scale"])[:take])

        source = dict(metadata or {})
        for key in ("qi_mean", "qi_var", "qbg_mean"):
            if key in forward_out:
                source[key] = forward_out[key]
        for key in (*self.covariates, "qi_mean", "qi_var", "qbg_mean"):
            if key not in source:
                continue
            values = _np(source[key]).reshape(-1)[:take]
            self._extra.setdefault(key, []).append(values)
        self._n += take

    def frame(self) -> pl.DataFrame | None:
        """Tidy frame with one row per reflection: `h0..hd`, scales, covariates."""
        if not self._loc:
            return None
        loc = np.concatenate(self._loc)
        cols = {f"h{i}": loc[:, i] for i in range(loc.shape[1])}
        if self._scale:
            scale = np.concatenate(self._scale)
            for i in range(scale.shape[1]):
                cols[f"s{i}"] = scale[:, i]
        for key, chunks in self._extra.items():
            values = np.concatenate(chunks)
            if values.shape[0] == loc.shape[0]:
                cols[key.replace(".", "_")] = values
        return pl.DataFrame(cols)

    def save(self, out_dir: str | Path, epoch: int) -> Path | None:
        """Write this epoch's latents; returns the path, or None if empty."""
        frame = self.frame()
        if frame is None:
            return None
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"latents_epoch_{int(epoch):04d}.parquet"
        frame.write_parquet(path)
        return path


def load_tracked(fig_dir: str | Path) -> tuple[dict, pl.DataFrame, dict]:
    """Read back `tracked_selection.json`, the scalars, and the arrays."""
    fig_dir = Path(fig_dir)
    selection = json.loads((fig_dir / "tracked_selection.json").read_text())
    scalars = pl.read_parquet(fig_dir / "tracked_scalars.parquet")
    arrays = dict(np.load(fig_dir / "tracked_arrays.npz"))
    return selection, scalars, arrays


def load_basis(fig_dir: str | Path) -> tuple[dict, pl.DataFrame]:
    """Read back the basis snapshots and their diagnostics."""
    fig_dir = Path(fig_dir)
    snapshots = dict(np.load(fig_dir / "basis_snapshots.npz"))
    diagnostics = pl.read_parquet(fig_dir / "basis_diagnostics.parquet")
    return snapshots, diagnostics


def load_latents(fig_dir: str | Path) -> dict[int, pl.DataFrame]:
    """Read every `latents_epoch_XXXX.parquet` keyed by epoch."""
    frames = {}
    for path in sorted(Path(fig_dir).glob("latents_epoch_*.parquet")):
        epoch = int(path.stem.split("_")[-1])
        frames[epoch] = pl.read_parquet(path)
    return frames
