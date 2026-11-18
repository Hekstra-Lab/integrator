"""The profile each model draws, overall and by detector region.

Two questions from one pass over the data, because both need the same
expensive thing -- the model's forward on real shoeboxes.

**Which profile does this model draw?** Most profile models learn no basis
at all: `shifted_basis_profile` warps a fixed grid, `steerable_profile`
and `mixture_gaussian_profile` evaluate an analytic family, and
`quasiconformal_profile` learns a warp of eleven parameters. Only
`learned_basis_profile` and `geometric_basis_profile` keep a decoder matrix
that `dump_basis.py` can read, and `tucker_basis_profile` keeps one
factorised. So a basis figure cannot compare the families; the profile they
produce can, and it is what the model actually puts on the detector.

**Does the profile depend on where the reflection landed?** The detector is
split into a 3x3 grid on `xyzcal.px`, and the mean profile is accumulated
per cell. A model that tracks position shows a different shape in the
corners than at the centre; one that has stopped reading the shoebox -- the
dead-encoder failure `dump_encoder_health.py` measures -- shows the same
shape in all nine.

The grid is cut on the OBSERVED range of each dataset's own predicted
positions rather than a nominal detector size, so a panel is never empty
because the beam centre was offset.

Writes one npz with, per arm: `<arm>_mean` (h, w), `<arm>_cells`
(3, 3, h, w), `<arm>_counts` (3, 3), `<arm>_extent` (4,), and a long CSV
beside it with the per-cell `n_eff = 1 / sum(p^2)` and the fraction of mass
outside the central 5x5, which are the two numbers that say whether a
profile is peaked or flat.

Usage:
    DATASET=dhfr_sim python dump_profile_maps.py <out.npz> <run-dir>...
    N_BATCHES=8 ...   # how much data to average over, default 4
"""

from __future__ import annotations

import csv
import glob
import os
import re
import sys

DATASET = os.environ.get("DATASET", "")
N_BATCHES = int(os.environ.get("N_BATCHES", "4"))
GRID = 3


def last_checkpoint(run_dir: str) -> str | None:
    found = sorted(
        p for p in glob.glob(f"{run_dir}/files/checkpoints/epoch=*.ckpt")
        if re.search(r"epoch=(\d+)", os.path.basename(p))
    )
    return found[-1] if found else None


def profile_of(outputs):
    """The mean profile, (B, P), from whatever the forward returned."""
    qp = outputs.get("qp") if isinstance(outputs, dict) else None
    for attr in ("mean_profile", "mean"):
        got = getattr(qp, attr, None)
        if got is not None:
            return got
    return None


def gate_of(outputs):
    """The per-pixel signal probability, (B, P), where the model has one.

    The SVAE integrators put it in `forward_out["signal_prob"]`: with
    `gate: learned` it is `sigmoid(gate_head(x))`, and with
    `gate: responsibility` it is `lam_s / (lam_s + b0)`, computed rather
    than learned. The hierarchical integrators have no gate and return
    nothing, which is why this returns None rather than raising -- the
    figure then draws the profile panels alone.
    """
    if not isinstance(outputs, dict):
        return None
    for key in ("signal_prob",):
        if key in outputs:
            return outputs[key]
    inner = outputs.get("forward_out")
    if isinstance(inner, dict):
        return inner.get("signal_prob")
    return None


def accumulate(model, loader, device, n_batches: int):
    """Mean profile overall and per detector cell, in one pass."""
    import numpy as np
    import torch

    total = None
    cells: dict[tuple[int, int], tuple] = {}
    xs, ys = [], []
    kept, gates = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            counts, shoebox, mask, metadata = batch
            counts = counts.to(device)
            shoebox = shoebox.to(device)
            mask = mask.to(device)
            metadata = {k: (v.to(device) if hasattr(v, "to") else v)
                        for k, v in metadata.items()}
            out = model(counts, shoebox, mask, metadata)
            prf = profile_of(out)
            if prf is None:
                return None
            p = prf.detach().float().cpu().numpy()
            kept.append(p)
            g = gate_of(out)
            gates.append(g.detach().float().cpu().numpy()
                         if g is not None else None)
            xs.append(metadata["xyzcal.px.0"].detach().float().cpu().numpy())
            ys.append(metadata["xyzcal.px.1"].detach().float().cpu().numpy())
    if not kept:
        return None
    p = np.concatenate(kept, axis=0)
    gate = (np.concatenate(gates, axis=0)
            if gates and all(g is not None for g in gates) else None)
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    total = p.mean(axis=0)

    # the grid is cut on the OBSERVED range, so no cell is empty merely
    # because the beam centre is off the geometric middle of the panel
    x0, x1 = float(np.min(x)), float(np.max(x))
    y0, y1 = float(np.min(y)), float(np.max(y))
    ix = np.clip(((x - x0) / max(x1 - x0, 1e-9) * GRID).astype(int), 0, GRID - 1)
    iy = np.clip(((y - y0) / max(y1 - y0, 1e-9) * GRID).astype(int), 0, GRID - 1)
    for r in range(GRID):
        for c in range(GRID):
            sel = (iy == r) & (ix == c)
            cells[(r, c)] = (
                p[sel].mean(axis=0) if sel.any() else None,
                int(sel.sum()),
                gate[sel].mean(axis=0) if (gate is not None and sel.any())
                else None,
            )
    gate_total = gate.mean(axis=0) if gate is not None else None
    return total, cells, (x0, x1, y0, y1), gate_total


def profile_stats(p, h: int, w: int) -> tuple[float, float]:
    """`n_eff = 1 / sum(p^2)` and the mass outside the central 5x5."""
    s = float((p ** 2).sum())
    n_eff = 1.0 / s if s > 0 else float("nan")
    if h and w and p.size == h * w:
        box = p.reshape(h, w)
        r0, c0 = max(h // 2 - 2, 0), max(w // 2 - 2, 0)
        core = float(box[r0:r0 + 5, c0:c0 + 5].sum())
    else:
        core = float("nan")
    return n_eff, 1.0 - core


def shape_of(cfg) -> tuple[int, int]:
    args = (cfg.get("integrator", {}) or {}).get("args", {}) or {}
    return int(args.get("h", 0) or 0), int(args.get("w", 0) or 0)


def main() -> int:
    if not DATASET:
        raise SystemExit("set DATASET to the dataset these runs trained on")
    if len(sys.argv) < 3:
        raise SystemExit("usage: dump_profile_maps.py <out.npz> <run-dir>...")

    import numpy as np
    import torch
    import yaml

    from integrator.utils.factory_utils import (
        construct_data_loader,
        construct_integrator,
    )

    out_path, runs = sys.argv[1], sys.argv[2:]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blobs: dict[str, np.ndarray] = {}
    rows = []
    for run in runs:
        arm = os.path.basename(run.rstrip("/"))
        cfg_path = os.path.join(run, "config_log.yaml")
        ckpt = last_checkpoint(run)
        if not os.path.isfile(cfg_path) or not ckpt:
            print(f"  {arm}: no config_log.yaml or no checkpoint")
            continue
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh) or {}
        qp = ((cfg.get("surrogates", {}) or {}).get("qp", {}) or {}).get("name", "?")
        h, w = shape_of(cfg)
        try:
            model = construct_integrator(cfg).to(device).eval()
            state = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state.get("state_dict", state), strict=False)
            dm = construct_data_loader(cfg)
            dm.setup("fit")
            got = accumulate(model, dm.train_dataloader(), device, N_BATCHES)
        except Exception as exc:
            print(f"  {arm}: {type(exc).__name__}: {str(exc)[:110]}")
            continue
        if got is None:
            print(f"  {arm}: the forward returned no profile")
            continue
        total, cells, extent, gate_total = got
        grid = np.zeros((GRID, GRID, total.size), dtype=np.float32)
        counts = np.zeros((GRID, GRID), dtype=np.int32)
        gate_grid = np.zeros_like(grid)
        has_gate = gate_total is not None
        for (r, c), (mean, n, gmean) in cells.items():
            counts[r, c] = n
            if mean is not None:
                grid[r, c] = mean
            if gmean is not None:
                gate_grid[r, c] = gmean
        blobs[f"{arm}_mean"] = total.astype(np.float32)
        blobs[f"{arm}_cells"] = grid
        blobs[f"{arm}_counts"] = counts
        blobs[f"{arm}_extent"] = np.asarray(extent, dtype=np.float32)
        blobs[f"{arm}_shape"] = np.asarray([h, w], dtype=np.int32)
        if has_gate:
            blobs[f"{arm}_gate_mean"] = gate_total.astype(np.float32)
            blobs[f"{arm}_gate_cells"] = gate_grid

        n_eff, outside = profile_stats(total, h, w)
        rows.append({"method": arm, "dataset": DATASET, "profile_model": qp,
                     "cell": "all", "row": "", "col": "",
                     "n": int(counts.sum()), "n_eff": round(n_eff, 4),
                     "mass_outside_5x5": round(outside, 4)})
        for r in range(GRID):
            for c in range(GRID):
                if counts[r, c] == 0:
                    continue
                n_eff, outside = profile_stats(grid[r, c], h, w)
                rows.append({"method": arm, "dataset": DATASET,
                             "profile_model": qp, "cell": f"{r}{c}",
                             "row": r, "col": c, "n": int(counts[r, c]),
                             "n_eff": round(n_eff, 4),
                             "mass_outside_5x5": round(outside, 4)})
        print(f"  {arm}: {qp}, shoebox {h}x{w}, {int(counts.sum()):,} "
              f"reflections, n_eff {rows[-1]['n_eff']:.1f}")

    if not blobs:
        raise SystemExit("nothing measured")
    np.savez_compressed(out_path, **blobs)
    csv_path = os.path.splitext(out_path)[0] + ".csv"
    with open(csv_path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"{out_path}: {len({k.split('_')[0] for k in blobs})} arms")
    print(f"{csv_path}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
