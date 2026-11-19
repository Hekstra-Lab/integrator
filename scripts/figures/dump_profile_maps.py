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
# REFL_IDS names a file of reflection ids, one per line or a .npy, and the
# reducer then collects THOSE reflections from every arm however many
# batches they are spread over. Naming them is what makes two arms'
# gates comparable; see `accumulate`.
_ids_file = os.environ.get("REFL_IDS", "")
WANT_IDS = None
if _ids_file:
    import numpy as _np
    WANT_IDS = (_np.load(_ids_file) if _ids_file.endswith(".npy")
                else _np.loadtxt(_ids_file, dtype="int64")).ravel()
    print(f"collecting {WANT_IDS.size} named reflections from {_ids_file}")
N_EXAMPLES = int(os.environ.get("N_EXAMPLES", "6"))
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


def accumulate(model, loader, device, n_batches: int, want_ids=None):
    """Mean profile overall and per detector cell, in one pass.

    `want_ids` is a set of reflection ids to collect for the batch-level
    arrays. The reflections are IDENTIFIED rather than taken positionally,
    because no loader guarantees two runs see the same rows: the train
    loader shuffles per run and the train/val SPLIT is drawn per run, so
    two arms' first batches shared none of their reflections -- measured,
    zero of 2048, twice. The data carries `refl_ids`, so the reflections
    to compare can simply be named, and then it does not matter what order
    a loader produces them in or how many batches they are spread over.
    """
    import numpy as np
    import torch

    total = None
    cells: dict[tuple[int, int], tuple] = {}
    xs, ys = [], []
    kept, gates, raws = [], [], []
    examples: dict = {}
    # rows matching want_ids, gathered across however many batches they
    # are spread over
    want = None if want_ids is None else set(int(v) for v in want_ids)
    picked: dict[str, list] = {k: [] for k in
                              ("refl_id", "gate", "profile", "counts",
                               "mask", "d")}
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches and not want:
                break
            counts, shoebox, mask, metadata = batch
            counts = counts.to(device)
            shoebox = shoebox.to(device)
            mask = mask.to(device)
            metadata = {k: (v.to(device) if hasattr(v, "to") else v)
                        for k, v in metadata.items()}
            metadata, note = ensure_group_label(metadata, device)
            if note and i == 0:
                print(f"    {note}")
            out = model(counts, shoebox, mask, metadata)
            prf = profile_of(out)
            if prf is None:
                return None
            p = prf.detach().float().cpu().numpy()
            g = gate_of(out)
            # Every per-reflection array has to be appended under the SAME
            # condition, because they are concatenated and then indexed by
            # one boolean mask. `want_ids` runs the loop past `n_batches`
            # to find the named reflections, and appending the gate, the
            # counts and the positions on those extra batches while the
            # profile was not left `p` with 512 rows against a mask of
            # 2,652,160 -- `IndexError: boolean index did not match`, after
            # 55 minutes of forwards.
            if i < n_batches:
                kept.append(p)
                gates.append(g.detach().float().cpu().numpy()
                             if g is not None else None)
                raws.append(counts.detach().float().cpu().numpy())
                xs.append(
                    metadata["xyzcal.px.0"].detach().float().cpu().numpy())
                ys.append(
                    metadata["xyzcal.px.1"].detach().float().cpu().numpy())

            if want:
                rid = metadata.get("refl_ids")
                if rid is not None:
                    rid_np = rid.detach().cpu().numpy().astype("int64")
                    sel = np.flatnonzero(np.isin(rid_np, list(want)))
                    if sel.size:
                        gg = g
                        dd = metadata.get("d")
                        picked["refl_id"].append(rid_np[sel])
                        picked["profile"].append(p[sel])
                        picked["counts"].append(
                            counts.detach().float().cpu().numpy()[sel])
                        picked["mask"].append(
                            mask.detach().float().cpu().numpy()[sel])
                        if gg is not None:
                            picked["gate"].append(
                                gg.detach().float().cpu().numpy()[sel])
                        if dd is not None:
                            picked["d"].append(
                                dd.detach().float().cpu().numpy()[sel])
                        want -= set(rid_np[sel].tolist())
                        if not want:
                            print(f"    collected every named reflection "
                                  f"by batch {i}")
            if i == 0:
                # a few whole reflections, kept as they are: the observed
                # counts, the profile drawn for them, and the gate. A mean
                # over thousands hides whether any single shoebox is fitted
                # at all, which is the thing the eye settles fastest.
                examples["counts"] = counts.detach().float().cpu().numpy()
                examples["profile"] = p
                if g is not None:
                    examples["gate"] = g.detach().float().cpu().numpy()
                d = metadata.get("d")
                if d is not None:
                    examples["d"] = d.detach().float().cpu().numpy()
                # The reflection ids for the WHOLE first batch, with its
                # gate, so two arms' gates can be compared on the same
                # reflections. Without an id the arrays are two different
                # batches in two different orders and any agreement
                # measured between them is noise.
                rid = metadata.get("refl_ids")
                if rid is not None:
                    examples["batch_refl_id"] = (
                        rid.detach().cpu().numpy().astype("int64"))
                if g is not None:
                    examples["batch_gate"] = (
                        g.detach().float().cpu().numpy())
                examples["batch_profile"] = p
                # The counts and the mask for the same batch. Without the
                # counts a gate can only be compared with another gate,
                # and the question that matters -- is the gate where the
                # signal actually is -- cannot be asked at all, because
                # the only record of where the signal is is the shoebox.
                examples["batch_counts"] = (
                    counts.detach().float().cpu().numpy())
                examples["batch_mask"] = (
                    mask.detach().float().cpu().numpy())
                if d is not None:
                    examples["batch_d"] = d.detach().float().cpu().numpy()
    if not kept:
        return None
    p = np.concatenate(kept, axis=0)
    gate = (np.concatenate(gates, axis=0)
            if gates and all(g is not None for g in gates) else None)
    raw = np.concatenate(raws, axis=0)
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
    # the reflections named by want_ids, concatenated across batches, into
    # `examples` under `named_*` so one npz carries both the six drawn
    # examples and the identified set the arms are compared on
    for key, chunks in picked.items():
        if chunks:
            examples[f"named_{key}"] = np.concatenate(chunks, axis=0)
    if want:
        print(f"    {len(want)} named reflections were not in the loader")

    return total, cells, (x0, x1, y0, y1), gate_total, examples, p, gate, raw


def ensure_group_label(metadata, device):
    """Supply `group_label` when the installed data module did not.

    The repository's `POLY_DS_COLS` lists it, but the environment's
    installed package is older and the PDZ2 metadata carries `radial_bin`
    instead, so the forward died on `KeyError: 'group_label'`. It selects
    the per-group prior, so a wrong value would move the PRIOR -- not the
    encoder, the profile shape or the gate, which is what is being
    measured here. `radial_bin` is used where present and zeros
    otherwise, and the substitution is printed rather than done quietly.
    """
    import torch

    if "group_label" in metadata:
        return metadata, ""
    ref = next((v for v in metadata.values() if hasattr(v, "shape")), None)
    if ref is None:
        return metadata, ""
    if "radial_bin" in metadata:
        metadata["group_label"] = metadata["radial_bin"].long()
        return metadata, "group_label <- radial_bin"
    metadata["group_label"] = torch.zeros(
        ref.shape[0], dtype=torch.long, device=device)
    return metadata, "group_label <- zeros (prior groups collapsed)"


def gate_fidelity(prf, gate, counts):
    """How far the gate is from being a function of the profile.

    `r_i = I p_i / (I p_i + B)` is monotone increasing in `p_i`, so a
    correctly constructed gate ranks pixels exactly as the profile does and
    the rank correlation is 1 BY CONSTRUCTION, not as an empirical finding.
    Anything below 1 is a defect with no threshold to argue about.

    Two more, because ordering is necessary and not sufficient:

    - `pearson_weak` -- in the weak limit `I p << B` the sigmoid is in its
      linear region and `r ~ (I/B) p`, so the gate should be the profile
      scaled down, pixel for pixel. Measured only on the reflections that
      are in that regime.
    - `mae_vs_exact` -- the mean absolute gap to the responsibility built
      from a joint (I, B) fit on the same counts, which catches a gate
      that orders pixels correctly at the wrong level.
    """
    import numpy as np
    from scipy.stats import rankdata

    n_pix = prf.shape[1]
    spp = (prf * prf).sum(1)
    sp = prf.sum(1)
    spc = (prf * counts).sum(1)
    sc = counts.sum(1)
    det = spp * n_pix - sp * sp
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    i_fit = (n_pix * spc - sp * sc) / det
    b_fit = (spp * sc - sp * spc) / det

    rho, pear, mae, weak = [], [], [], []
    for k in range(prf.shape[0]):
        pk, gk = prf[k], gate[k]
        if np.ptp(pk) <= 0 or np.ptp(gk) <= 0:
            continue
        rho.append(float(np.corrcoef(rankdata(pk), rankdata(gk))[0, 1]))
        if i_fit[k] <= 0 or b_fit[k] <= 0:
            continue
        exact = i_fit[k] * pk / (i_fit[k] * pk + b_fit[k])
        mae.append(float(np.abs(gk - exact).mean()))
        if i_fit[k] * pk.max() / b_fit[k] < 3.0:   # linear regime
            weak.append(float(np.corrcoef(pk, gk)[0, 1]))
    out = {
        "gate_profile_rho": float(np.median(rho)) if rho else float("nan"),
        "gate_pearson_weak": float(np.median(weak)) if weak else float("nan"),
        "gate_mae_vs_exact": float(np.median(mae)) if mae else float("nan"),
        "n_scored": len(rho),
    }
    return {k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in out.items()}


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
            # The PREDICT loader. It does not shuffle AND it covers every
            # reflection, which the validation loader does not: the
            # train/val split is drawn per run, so two arms' validation
            # sets are different reflections and their first batches
            # shared none of them -- measured twice, zero of 2048 both
            # times, once with the training loader and once with the
            # validation loader.
            #
            # The predict loader is the one the prediction parquets come
            # from, and those join across arms on 265,856 reflections, so
            # it is the loader under which a gate can be compared with
            # another arm's gate and with the counts.
            loader = (dm.predict_dataloader()
                      if hasattr(dm, "predict_dataloader")
                      else dm.train_dataloader())
            got = accumulate(model, loader, device, N_BATCHES,
                             want_ids=WANT_IDS)
        except Exception as exc:
            print(f"  {arm}: {type(exc).__name__}: {str(exc)[:110]}")
            continue
        if got is None:
            print(f"  {arm}: the forward returned no profile")
            continue
        total, cells, extent, gate_total, examples, all_p, all_g, all_c = got
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
        # Examples spread over RESOLUTION, not over brightness. The question
        # is whether a model draws the same shape at 1.8 A as at 4 A, and
        # d is the axis it would vary along: the spot gets weaker and the
        # background does not, so the gate has progressively less to work
        # with. Sorting by counts instead would mostly reorder the same
        # resolution shell.
        ex_d = examples.get("d")
        ex_counts = examples.get("counts")
        if ex_counts is not None:
            if ex_d is not None and np.ptp(ex_d) > 0:
                order = np.argsort(-ex_d)  # low resolution first
            else:
                order = np.argsort(
                    ex_counts.reshape(ex_counts.shape[0], -1).sum(1))
            pick = order[np.linspace(0, len(order) - 1, N_EXAMPLES).astype(int)]
            for key in ("counts", "profile", "gate", "d"):
                arr = examples.get(key)
                if arr is not None:
                    blobs[f"{arm}_ex_{key}"] = arr[pick].astype(np.float32)
        # the whole first batch, for arm-against-arm comparison by id
        for key in ("batch_refl_id", "batch_gate", "batch_profile",
                    "batch_counts", "batch_mask", "batch_d",
                    "named_refl_id", "named_gate", "named_profile",
                    "named_counts", "named_mask", "named_d"):
            arr = examples.get(key)
            if arr is not None:
                blobs[f"{arm}_{key}"] = (
                    arr.astype(np.int64) if key.endswith("refl_id")
                    else arr.astype(np.float32))

        n_eff, outside = profile_stats(total, h, w)
        # The gate's fidelity to the profile. Only an SVAE has a gate; the
        # hierarchical family returns none and the columns stay blank.
        fid = (gate_fidelity(all_p, all_g, all_c) if all_g is not None
               else {"gate_profile_rho": "", "gate_pearson_weak": "",
                     "gate_mae_vs_exact": "", "n_scored": 0})
        rows.append({"method": arm, "dataset": DATASET, "profile_model": qp,
                     "cell": "all", "row": "", "col": "",
                     "n": int(counts.sum()), "n_eff": round(n_eff, 4),
                     "mass_outside_5x5": round(outside, 4), **fid})
        for r in range(GRID):
            for c in range(GRID):
                if counts[r, c] == 0:
                    continue
                n_eff, outside = profile_stats(grid[r, c], h, w)
                rows.append({"method": arm, "dataset": DATASET,
                             "profile_model": qp, "cell": f"{r}{c}",
                             "row": r, "col": c, "n": int(counts[r, c]),
                             "n_eff": round(n_eff, 4),
                             "mass_outside_5x5": round(outside, 4),
                             "gate_profile_rho": "", "gate_pearson_weak": "",
                             "gate_mae_vs_exact": "", "n_scored": 0})
        rho = fid.get("gate_profile_rho", "")
        print(f"  {arm}: {qp}, shoebox {h}x{w}, {int(counts.sum()):,} "
              f"reflections, n_eff {n_eff:.1f}"
              + (f", gate-profile rho {rho} (1.0 is forced by the form)"
                 if rho != "" else ", no gate"))

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
