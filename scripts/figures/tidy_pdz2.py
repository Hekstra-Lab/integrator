"""Reduce the PDZ2 / 1116 electric-field TR-Laue arms to plot-ready tables.

Runs on the cluster, like `tidy_mono.py`, and writes the long-format CSVs the
figures read. PDZ2 is the only dataset whose endpoint is **CCsym** -- the
half-dataset correlation of `F(h) - F(2-fold h)`, which is zero without a
field -- so its tables are not the same ones the monochromatic arm writes.

`pdz2_ccsym.csv`
    method, dataset, delay, bin, repeat, d_max, d_min, ccsym.
    From each arm's `e35_ccsym.csv`, written by the deposited
    `custom_ccsym.py`. Two things about that file need undoing here. Its
    `bin` column is **doubled** (`result[0]["bin"] = result[0]["bin"]*2`,
    so ten bins are numbered 0, 2, ..., 18) because the plot it makes puts
    the tick labels between the bins. And it saves no resolution at all --
    the shell edges exist only as tick labels on its SVG -- so they are
    recomputed here by calling that script's own `make_halves_ccsym`, which
    is copied into each tree, rather than by reimplementing its binning.
    The correlation is **Spearman**: `custom_ccsym.py` defaults to it and
    `stats_for_production.sh` does not pass `-m`.

`pdz2_refinement.csv`
    method, dataset, replicate, r_work, r_free.
    The deposited pipeline computes no refinement for PDZ2, so this chain is
    ours: the field-off state refined against `merged_e35_0.mtz`, which is
    already in C2 and matches `5e1y.pdb`, with one R-free set shared across
    every arm so they are compared on the same held-out reflections.
    `replicate` is the careless merging seed, which dominates the refinement
    seed about ninefold on Laue data.

`pdz2_training_loss.csv`
    method, dataset, epoch, split, term, value -- the same columns
    `tidy_mono.py` writes, so one figure draws either arm.

`careless_stats.csv`
    The per-arm tables written by `run_pipeline.py --steps stats
    --out-base <...>/merged_e35`, concatenated. Already long format and
    already carrying `method` and `dataset`; this only puts them in one file.

Usage:
    python tidy_pdz2.py <out-dir> [arm ...]

With no arm named, every arm in `ARMS` that has output on disk.
"""

import csv
import glob
import os
import re
import sys

S = "/n/netscratch/hekstra_lab/Lab/laldama"
DATASET = os.environ.get("DATASET", "pdz2_1116")
# Where each arm's copy of the reference chain sits. `laue` is the reference
# itself; everything else is a tree built by swapping our intensities into
# the same 16 integrated.mtz slots and running the identical downstream.
ARMS = {
    "laue": f"{S}/laue_runs/pdz2_ref",
    "w21_k7m": f"{S}/laue_runs/pdz2_model/tree_w21_k7m",
    "k7m_bfix": f"{S}/laue_runs/pdz2_model/tree_w21_k7m_bfix",
    "geom_both": f"{S}/laue_runs/pdz2_model/tree_geom_both",
    "k7m_mlclamp": f"{S}/laue_runs/pdz2_model/tree_k7m_mlclamp",
    "combo": f"{S}/laue_runs/pdz2_model/tree_combo",
}
# The training run behind each arm, for the loss curves. The export-only arms
# (`k7m_ml`, `k7m_mlclamp`, `combo`, the swap trees) have no training of
# their own: they rewrite a trained arm's intensities, so they share its
# loss and are left out rather than drawn as a duplicate line.
TRAINING_RUN = {
    "w21_k7m": "pdz2_w21_k7m",
    "k7m_bfix": "pdz2_w21_k7m",
    "geom_both": "pdz2_geom_both",
}
# inside a tree, the one careless run the whole chain writes to
CARELESS = "gain_0,3_from_stills/careless-cdef-0,3-ohp-mlpw"
# `custom_ccsym.py` is called with these, in `stats_for_production.sh`
CCSYM_ARGS = dict(spacegroup=5, isym=1, cb_op="a+b,a-b,-c", bins=10)
# the ELBO terms the training loop writes, as in tidy_mono.py
TERMS = ("loss", "nll", "kl", "kl_prf", "kl_i", "kl_bg")
# the four pump delays, in time order rather than alphabetical
DELAYS = ("off", "50ns", "100ns", "200ns")


def write(path: str, rows: list[dict]) -> None:
    if not rows:
        print(f"  {os.path.basename(path)}: nothing to write")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"  {os.path.basename(path)}: {len(rows)} rows")


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def shell_edges(tree: str) -> dict[int, tuple[float, float]]:
    """The resolution span of each CCsym bin, keyed by the undoubled index.

    Computed by calling the deposited `custom_ccsym.make_halves_ccsym`, which
    is copied into every tree, so the binning cannot drift from the one the
    CCsym values were computed under. Reimplementing it here was the
    alternative and it would have been a second definition of the same shells.
    """
    script = f"{tree}/scripts/custom_ccsym.py"
    if not os.path.exists(script):
        script = f"{S}/laue_runs/pdz2_ref/scripts/custom_ccsym.py"
    if not os.path.exists(script):
        print("  shell edges: no custom_ccsym.py found")
        return {}
    xval = f"{tree}/{CARELESS}/merged_e35_xval_4.mtz"
    if not os.path.exists(xval):
        print(f"  shell edges: no {os.path.basename(xval)}")
        return {}

    import gemmi
    import reciprocalspaceship as rs

    sys.path.insert(0, os.path.dirname(script))
    import custom_ccsym  # noqa: E402  (a deposited script, loaded by path)

    sg = gemmi.SpaceGroup(str(CCSYM_ARGS["spacegroup"]))
    op = sg.operations().sym_ops[CCSYM_ARGS["isym"]]
    _, labels = custom_ccsym.make_halves_ccsym(
        rs.read_mtz(xval), op, CCSYM_ARGS["cb_op"], CCSYM_ARGS["bins"]
    )
    out = {}
    for i, text in enumerate(labels):
        span = re.findall(r"[0-9]*\.?[0-9]+", str(text))
        if len(span) >= 2:
            out[i] = (float(span[0]), float(span[1]))
    return out


def ccsym(arms: dict[str, str]) -> list[dict]:
    """CCsym per arm, delay, resolution bin and cross-validation repeat."""
    out = []
    edges: dict[int, tuple[float, float]] = {}
    for arm, tree in arms.items():
        path = f"{tree}/{CARELESS}/e35_ccsym.csv"
        if not os.path.exists(path):
            continue
        if not edges:
            edges = shell_edges(tree)
        with open(path) as fh:
            for rec in csv.DictReader(fh):
                value = _float(rec.get("CCsym"))
                doubled = _float(rec.get("bin"))
                if value is None or doubled is None:
                    continue
                # `custom_ccsym.py` doubles the bin index before saving
                index = int(doubled) // 2
                d_max, d_min = edges.get(index, ("", ""))
                out.append({
                    "method": arm,
                    "dataset": DATASET,
                    "delay": rec.get("delay", ""),
                    "bin": index + 1,
                    "repeat": rec.get("repeat", ""),
                    "d_max": d_max,
                    "d_min": d_min,
                    "ccsym": value,
                })
    return out


def refinement(arms: dict[str, str]) -> list[dict]:
    """R-work and R-free of the field-off refinement, per merging seed.

    One directory per arm and replicate under `refine_off/`, named
    `run_<arm>[_r<N>]`. The numbers are read by `integrator.utils.metrics`,
    which holds the only parser for a phenix log in this project.
    """
    from integrator.utils import metrics

    base = f"{S}/laue_runs/pdz2_model/refine_off"
    # the directory names do not all match the CCsym arm labels
    aliases = {"k7m_mlclamp": "ml_clamp", "k7m_bfix": "k7m_bfix"}
    out = []
    for arm in arms:
        stem = aliases.get(arm, arm)
        for d in sorted(glob.glob(f"{base}/run_{stem}") +
                        glob.glob(f"{base}/run_{stem}_r[0-9]*")):
            log = f"{d}/phenix.log"
            if not os.path.exists(log):
                continue
            got = metrics.refinement_from_phenix_log(log)
            if not got:
                continue
            m = re.search(r"_r(\d+)$", os.path.basename(d))
            out.append({
                "method": arm,
                "dataset": DATASET,
                "replicate": int(m.group(1)) if m else 1,
                "r_work": got["r_work"],
                "r_free": got["r_free"],
            })
    return out


def training_loss(arms: dict[str, str]) -> list[dict]:
    """The ELBO terms per epoch, in the columns tidy_mono.py writes."""
    out = []
    for arm in arms:
        run = TRAINING_RUN.get(arm)
        if not run:
            continue
        path = f"{S}/laue_runs/pdz2_model/runs/{run}/plots/loss_history.csv"
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for row in csv.DictReader(fh):
                for split in ("train", "val"):
                    for term in TERMS:
                        v = _float(row.get(f"{split}_{term}", ""))
                        if v is None:
                            continue
                        out.append({
                            "method": arm,
                            "dataset": DATASET,
                            "epoch": int(float(row["epoch"])),
                            "split": split,
                            "term": term,
                            "value": v,
                        })
    return out


def careless_stats(arms: dict[str, str]) -> list[dict]:
    """Every arm's `careless_stats.csv`, concatenated unchanged.

    The stats step already writes long format with `method` and `dataset`,
    so nothing is recomputed here; the rows are only gathered into one file
    and the `method` key is taken from the arm name in case the step was run
    without `--method`.
    """
    out = []
    for arm, tree in arms.items():
        path = f"{tree}/{CARELESS}/stats/careless_stats.csv"
        if not os.path.exists(path):
            print(f"  careless_stats: {arm} has none")
            continue
        with open(path) as fh:
            for rec in csv.DictReader(fh):
                rec["method"] = arm
                rec["dataset"] = DATASET
                out.append(rec)
    return out


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    out_dir = sys.argv[1]
    named = sys.argv[2:] or list(ARMS)
    unknown = [a for a in named if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}; known: {sorted(ARMS)}")
    arms = {a: ARMS[a] for a in named if os.path.isdir(ARMS[a])}
    missing = [a for a in named if a not in arms]
    if missing:
        print(f"  no tree on disk: {missing}")
    os.makedirs(out_dir, exist_ok=True)

    write(f"{out_dir}/pdz2_ccsym.csv", ccsym(arms))
    write(f"{out_dir}/pdz2_refinement.csv", refinement(arms))
    write(f"{out_dir}/pdz2_training_loss.csv", training_loss(arms))
    write(f"{out_dir}/careless_stats_pdz2.csv", careless_stats(arms))


if __name__ == "__main__":
    main()
