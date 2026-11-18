"""Recover the learned profile basis over training, from the checkpoints.

`ProfileBasisLogger` already writes this during training, into
`basis_snapshots.npz` and `basis_diagnostics.parquet` -- but it is gated
behind `integrator.train --figures`, which no production run passed, so no
arm has those files. The decoder is a parameter, every checkpoint holds it,
and mono runs keep every checkpoint (`save_top_k: -1`), so the whole
trajectory is recoverable after the fact and nothing has to be retrained.

This reads the checkpoints' state dicts directly. No model is constructed:
the basis is `surrogates.qp.decoder.weight`, shape `(d*h*w, n_components)`,
and reading it needs neither the data, nor a GPU, nor a config the
checkpoint was trained under. The arrays are handed to the project's own
`BasisRecorder`, so the files come out in the layout `load_basis` and
`render_basis` already read and the convergence diagnostics -- effective
rank, principal angle to the final basis, relative step -- keep one
definition.

Not every profile model has a basis. `learned_basis_profile` and
`geometric_basis_profile` both decode through `nn.Linear`, so both work.
`shapelet_profile`, `physical_profile` and `fixed_basis_profile` do not
learn one: the shapelet and physical models evaluate an analytic family and
the fixed basis is frozen at its initial value by construction, so a
snapshot of it over epochs is a constant and this says so rather than
writing a flat trajectory.

Usage:
    python dump_basis.py <out-dir> <run-dir> [run-dir ...]
    STRIDE=5 python dump_basis.py ...     # every 5th checkpoint

Writes `<out-dir>/<arm>/basis_snapshots.npz` and `basis_diagnostics.parquet`
per run, plus one `basis_components.csv` across runs: the long-format table
the figure reads, one row per (method, dataset, epoch, component) with the
component's norm and its cosine against the same component at the last
epoch. The images stay in the npz because an image is not a table.
"""

import csv
import glob
import os
import re
import sys

import numpy as np

from integrator.reporting.figure_data import BasisRecorder, column_cosines

# No default. Every file this writes carries `method` and `dataset` as key
# columns, and a default silently labelled the PDZ2 pair `hewl_9b7c`, which
# is the kind of error that survives into a figure legend unchallenged.
DATASET = os.environ.get("DATASET", "")
STRIDE = int(os.environ.get("STRIDE", "1"))
# where the basis lives in a checkpoint's state dict
WEIGHT_KEY = "surrogates.qp.decoder.weight"
BIAS_KEY = "surrogates.qp.decoder.bias"
# profile models that decode through a learned nn.Linear, and so have a
# basis that moves during training
HAS_BASIS = ("learned_basis_profile", "geometric_basis_profile",
             "hellinger_basis_profile")
EPOCH_RE = re.compile(r"epoch=?_?(\d+)")


def epoch_of(path: str) -> int | None:
    m = EPOCH_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def surrogate_name(run_dir: str) -> str:
    """The `qp` surrogate this run trained, from its own config log."""
    path = f"{run_dir}/config_log.yaml"
    if not os.path.exists(path):
        return ""
    import yaml

    cfg = yaml.safe_load(open(path).read()) or {}
    qp = (cfg.get("surrogates") or {}).get("qp") or {}
    return str(qp.get("name", ""))


def geometry(run_dir: str) -> tuple[int, ...]:
    """The shoebox shape `(d, h, w)` the basis columns reshape to."""
    path = f"{run_dir}/config_log.yaml"
    if not os.path.exists(path):
        return ()
    import yaml

    cfg = yaml.safe_load(open(path).read()) or {}
    args = (cfg.get("integrator") or {}).get("args") or {}
    got = tuple(int(args[k]) for k in ("d", "h", "w") if k in args)
    return got if len(got) == 3 else ()


def checkpoints(run_dir: str) -> list[str]:
    """Every checkpoint with an epoch in its name, in epoch order.

    `last.ckpt` and `last-v1.ckpt` are left out: they duplicate an epoch
    that is already in the list and carry no epoch of their own, so
    including them would put one epoch in the trajectory twice.
    """
    hits = [p for p in glob.glob(f"{run_dir}/files/checkpoints/*.ckpt")
            if epoch_of(p) is not None]
    return sorted(hits, key=lambda p: epoch_of(p) or -1)


def read_basis(path: str):
    """The decoder weight and bias in one checkpoint, or None."""
    import torch

    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except (RuntimeError, EOFError, KeyError) as exc:
        print(f"    {os.path.basename(path)}: unreadable ({exc})")
        return None
    sd = blob.get("state_dict", blob)
    if WEIGHT_KEY not in sd:
        return None
    weight = sd[WEIGHT_KEY].detach().cpu().numpy()
    bias = sd.get(BIAS_KEY)
    bias = (bias.detach().cpu().numpy() if bias is not None
            else np.zeros(weight.shape[0], dtype=np.float32))
    return weight, bias


def component_rows(arm: str, epochs, weights) -> list[dict]:
    """Per-component norm and cosine against the final epoch.

    The norm says whether a component is carrying any amplitude at all --
    a dead component is one the encoder stopped using -- and the cosine
    against the final basis says when it stopped moving. Both are per
    component rather than aggregated, because the failure that matters is
    one component collapsing while the others look settled.
    """
    if not weights:
        return []
    final = weights[-1]
    rows = []
    for epoch, w in zip(epochs, weights, strict=True):
        cos = column_cosines(w, final)
        norms = np.linalg.norm(w, axis=0)
        for j in range(w.shape[1]):
            rows.append({
                "method": arm,
                "dataset": DATASET,
                "epoch": float(epoch),
                "component": j,
                "norm": float(norms[j]),
                "cos_to_final": float(cos[j]),
            })
    return rows


def dump_one(out_dir: str, run_dir: str) -> list[dict]:
    arm = os.path.basename(run_dir.rstrip("/"))
    name = surrogate_name(run_dir)
    if name and name not in HAS_BASIS:
        print(f"  {arm}: qp is `{name}`, which learns no basis; skipped")
        return []
    ckpts = checkpoints(run_dir)[::STRIDE]
    if not ckpts:
        print(f"  {arm}: no checkpoints with an epoch in the name")
        return []

    shape = geometry(run_dir)
    recorder = None
    epochs, weights = [], []
    for path in ckpts:
        got = read_basis(path)
        if got is None:
            continue
        weight, bias = got
        if recorder is None:
            # the shape the columns reshape to; fall back to a flat basis
            # rather than guessing a factorisation that may be wrong
            recorder = BasisRecorder(shape or (weight.shape[0],))
        recorder.record(float(epoch_of(path) or 0), weight, bias)
        epochs.append(epoch_of(path) or 0)
        weights.append(weight)

    if recorder is None or not epochs:
        print(f"  {arm}: no checkpoint carries {WEIGHT_KEY}")
        return []
    where = f"{out_dir}/{arm}"
    recorder.save(where)
    print(f"  {arm}: {len(epochs)} epochs, basis {weights[-1].shape}, "
          f"qp `{name or 'unknown'}` -> {where}")
    return component_rows(arm, epochs, weights)


def main() -> None:
    if not DATASET:
        raise SystemExit(
            "set DATASET to the dataset these runs trained on, e.g.\n"
            "    DATASET=pdz2_1116 python dump_basis.py <out-dir> <run-dir>...\n"
            "It is a key column in every file written here and cannot be "
            "guessed from the run directory."
        )
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out_dir, runs = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for run in runs:
        rows += dump_one(out_dir, run)
    if not rows:
        print("  basis_components.csv: nothing to write")
        return
    path = f"{out_dir}/basis_components.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  basis_components.csv: {len(rows)} rows")


if __name__ == "__main__":
    main()
