"""Reduce a polychromatic arm to the same plot-ready tables the mono arm writes.

Runs on the cluster, like `tidy_mono.py`, and writes the same columns, so one
figure draws either arm. `tidy_mono.py` hardcodes the monochromatic chain's
paths -- `$S/dials9b7c/<arm>_<est>/merge.log`, `$S/sim_runs/<arm>/plots/` --
and the polychromatic chain puts the same measurements somewhere else, which
is the whole reason this file exists.

Where each measurement lives, per arm, under `--root`:

| what | path |
|---|---|
| training loss | `<arm>/plots/loss_history.csv` |
| Wilson scale and B | `<arm>/plots/wilson_params.csv` |
| the learned spectrum | `<arm>/plots/wilson_spectrum.csv` |
| merging by shell | `<arm>/predictions/epoch_<N>/scaling/merging_stats.csv` |
| refinement | `.../scaling/config<C>_refine/refine{1,2}/` |
| anomalous peaks | `.../config<C>_refine/peaks.csv` |
| CCanom | `.../config<C>_refine/ccanom.csv` |
| careless stats | `.../scaling/stats/careless_stats.csv` |

An arm can be scaled TWICE from one checkpoint: `predictions/` holds the
posterior mean and variance, `rb_preds/` the Rao-Blackwellized fit, and the
two merge to different structure factors because careless weights by the
sigma it is handed. Both are collected, and the Rao-Blackwellized rows are
keyed `<arm>_rb` so they do not overwrite the posterior rows in a table
keyed on `method`. An arm with only `predictions/` is named exactly as
before.

**Every careless output sits under `epoch_0000` whatever epoch the checkpoint
was**, because the export writes there; the epoch in the table is read from
the checkpoint the predictions came from where that can be told, and from the
directory otherwise.

Not every arm has every table. A training-only arm -- the nanoBragg DHFR
simulation arms are scored against truth by `score_simulated_merge.py` rather
than through careless and phenix -- has `plots/` and no `predictions/`, and
that is a complete arm, not a broken one. Missing inputs are reported and
skipped.

Usage:
    python tidy_poly.py <out-dir> <arm> [arm ...]
    ROOT=/n/netscratch/.../ablate2 DATASET=hewl_1118 python tidy_poly.py ...
    ROOT=/n/netscratch/.../sim_runs DATASET=dhfr_sim python tidy_poly.py ...
"""

import csv
import glob
import os
import re
import sys

from integrator.utils import metrics

S = "/n/netscratch/hekstra_lab/Lab/laldama"
ROOT = os.environ.get("ROOT", f"{S}/ablate2")
DATASET = os.environ.get("DATASET", "hewl_1118")
# careless config the chain scaled with; `config4` is the anomalous recipe
CONFIG = os.environ.get("CARELESS_CONFIG", "4")
# the ELBO terms, as in tidy_mono.py
TERMS = ("loss", "nll", "kl", "kl_prf", "kl_i", "kl_bg")
SHELL_COLS = metrics.SHELL_COLUMNS
# `merging_stats.csv` spells one column differently from the shared schema
SHELL_ALIASES = {"n_uniq": "n_unique"}
EPOCH_RE = re.compile(r"epoch_(\d+)")
# The export directories an arm may have been scaled from, in the order
# they are collected. `predictions` is the posterior mean and variance and
# carries no suffix, so every table written before `rb_preds` existed keeps
# the method names it had.
EXPORT_LABEL = {"predictions": "", "rb_preds": "rb"}
EXPORTS = tuple(EXPORT_LABEL)


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


def scaling_dirs(arm: str) -> list[tuple[int, str]]:
    """Every `(epoch, scaling dir)` this arm scaled, in epoch order."""
    out = []
    for export in EXPORTS:
        for d in sorted(glob.glob(f"{ROOT}/{arm}/{export}/epoch_*/scaling")):
            m = EPOCH_RE.search(d)
            out.append((int(m.group(1)) if m else 0, d))
    return out


def export_of(where: str) -> str:
    """Which export a scaling directory belongs to, from its path.

    An arm can be scaled twice from the same checkpoint -- the posterior
    mean and variance under `predictions/`, the Rao-Blackwellized fit
    under `rb_preds/` -- and the two merge to different structure factors
    because careless weights by the sigma it is handed. They are different
    measurements of the same model and have to be different rows, or one
    silently overwrites the other in every table keyed by `method`.
    """
    for export, label in EXPORT_LABEL.items():
        if f"/{export}/" in where:
            return label
    return ""


def method_of(arm: str, where: str) -> str:
    """The `method` key for a row: the arm, and the export where it is not
    the default one."""
    label = export_of(where)
    return f"{arm}_{label}" if label else arm


def training_loss(arms: list[str]) -> list[dict]:
    out = []
    for arm in arms:
        path = f"{ROOT}/{arm}/plots/loss_history.csv"
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
                            "method": arm, "dataset": DATASET,
                            "epoch": int(float(row["epoch"])),
                            "split": split, "term": term, "value": v,
                        })
    return out


def wilson_params(arms: list[str]) -> list[dict]:
    """The prior's learned scale and B-factor, melted as the mono arm melts it."""
    out = []
    for arm in arms:
        path = f"{ROOT}/{arm}/plots/wilson_params.csv"
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for row in csv.DictReader(fh):
                for name, value in row.items():
                    v = _float(value)
                    if name == "epoch" or v is None:
                        continue
                    out.append({
                        "method": arm, "dataset": DATASET,
                        "epoch": int(float(row.get("epoch", 0))),
                        "parameter": name, "value": v,
                    })
    return out


def spectrum(arms: list[str]) -> list[dict]:
    """G(lambda), the one figure that is polychromatic by nature.

    Written at the last epoch only: `WilsonParamLogger` logs the table for
    the final snapshot and the figure over epochs reads checkpoints instead.
    """
    out = []
    for arm in arms:
        path = f"{ROOT}/{arm}/plots/wilson_spectrum.csv"
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for row in csv.DictReader(fh):
                lam, g = _float(row.get("wavelength")), _float(row.get("G"))
                if lam is None or g is None:
                    continue
                out.append({
                    "method": arm, "dataset": DATASET,
                    "wavelength": lam, "G": g,
                })
    return out


def merging_shells(arms: list[str]) -> list[dict]:
    """The per-shell table, in the shared column order.

    careless reports fewer columns than `dials.merge` does -- there is no
    Rmerge, Rmeas or multiplicity in `merging_stats.csv` -- so the missing
    ones are left empty rather than computed from something else, which
    would put two different quantities under one name.
    """
    out = []
    for arm in arms:
        for epoch, d in scaling_dirs(arm):
            path = f"{d}/merging_stats.csv"
            if not os.path.exists(path):
                continue
            with open(path) as fh:
                for row in csv.DictReader(fh):
                    rec = {"method": method_of(arm, d),
                           "dataset": DATASET, "epoch": epoch}
                    for col in SHELL_COLS:
                        name = SHELL_ALIASES.get(col, col)
                        rec[col] = _float(row.get(name, row.get(col, "")))
                        if rec[col] is None:
                            rec[col] = ""
                    out.append(rec)
    return out


def refinement(arms: list[str]) -> list[dict]:
    """R-work and R-free of each refinement pass.

    Two passes are run, so both are reported: pass 2 is the number quoted,
    and a pass 1 that is already at pass 2's value means the second pass did
    nothing, which is worth seeing rather than hiding.
    """
    out = []
    for arm in arms:
        for epoch, d in scaling_dirs(arm):
            for n in (1, 2):
                where = f"{d}/config{CONFIG}_refine/refine{n}"
                logs = sorted(glob.glob(f"{where}/*.log"))
                if not logs:
                    continue
                got = metrics.refinement_from_phenix_log(logs)
                if not got:
                    continue
                out.append({
                    "method": method_of(arm, where), "dataset": DATASET,
                    "epoch": epoch, "pass": n,
                    "r_work": got["r_work"], "r_free": got["r_free"],
                })
    return out


def anomalous_peaks(arms: list[str]) -> list[dict]:
    """Peak heights, with the atom carrying each peak resolved where possible.

    `peaks.csv` is preferred over `peak_heights.csv`: the older file carries
    no positions, so the atom can only be the heaviest of the named residue
    rather than the one nearest the peak. Both are read through
    `integrator.utils.metrics`, which holds the only parser for either.
    """
    out = []
    for arm in arms:
        for epoch, d in scaling_dirs(arm):
            where = f"{d}/config{CONFIG}_refine"
            model = refined_model(where)
            if os.path.exists(f"{where}/peaks.csv"):
                rows = metrics.peaks_from_find_peaks_csv(
                    f"{where}/peaks.csv", model=model)
                key = "peakz"
            elif os.path.exists(f"{where}/peak_heights.csv"):
                rows = metrics.peak_heights_from_csv(
                    f"{where}/peak_heights.csv", model=model)
                key = "peak_height"
            else:
                continue
            for peak in rows:
                out.append({
                    "method": method_of(arm, where), "dataset": DATASET,
                    "epoch": epoch,
                    "site": peak.get("label", ""),
                    "chain": peak.get("chain", ""),
                    "residue": peak.get("residue", ""),
                    "seqid": peak.get("seqid", ""),
                    "atom": peak.get("atom", ""),
                    "element": peak.get("element", ""),
                    "peak_height": peak.get(key, peak.get("peak_height", "")),
                    "dist": peak.get("dist", ""),
                    "atom_from_position": peak.get(
                        "atom_from_position", False),
                })
    return out


def refined_model(where: str) -> str | None:
    """The refined PDB beside a peak list; the newest wins."""
    hits = sorted(glob.glob(f"{where}/refine2/*[0-9].pdb")
                  + glob.glob(f"{where}/refine1/*[0-9].pdb")
                  + glob.glob(f"{where}/*[0-9].pdb"),
                  key=os.path.getmtime)
    return hits[-1] if hits else None


def ccanom(arms: list[str]) -> list[dict]:
    """CCanom by shell, which careless computes and DIALS reports differently."""
    out = []
    for arm in arms:
        for epoch, d in scaling_dirs(arm):
            path = f"{d}/config{CONFIG}_refine/ccanom.csv"
            if not os.path.exists(path):
                continue
            with open(path) as fh:
                for i, row in enumerate(csv.DictReader(fh)):
                    value = next(
                        (_float(v) for k, v in row.items()
                         if k and "cc" in k.lower() and _float(v) is not None),
                        None)
                    if value is None:
                        continue
                    out.append({
                        "method": method_of(arm, d), "dataset": DATASET,
                        "epoch": epoch, "bin": i + 1,
                        "shell": row.get("Resolution Range (Å)",
                                         row.get("bin", "")),
                        "cc_anom": value,
                    })
    return out


def careless_stats(arms: list[str]) -> list[dict]:
    """Every arm's `careless_stats.csv`, concatenated, one row per export.

    The pipeline's `stats` step already writes long format with `method`
    and `dataset`, so nothing is recomputed here. What is rewritten is
    `method`: the step was run once per export directory and stamped both
    with the same name, so without this the Rao-Blackwellized rows and the
    posterior rows are indistinguishable and a figure keyed on `method`
    draws one on top of the other.

    This is the table `fig_careless_stats.py` reads -- CChalf, CCpred,
    CCanom, Rsplit, I/sigma and completeness, all by shell.
    """
    out = []
    for arm in arms:
        for epoch, d in scaling_dirs(arm):
            path = f"{d}/stats/careless_stats.csv"
            if not os.path.exists(path):
                continue
            name = method_of(arm, d)
            n = 0
            with open(path) as fh:
                for rec in csv.DictReader(fh):
                    rec["method"] = name
                    rec["dataset"] = DATASET
                    rec["epoch"] = epoch
                    out.append(rec)
                    n += 1
            print(f"  careless_stats: {name} epoch {epoch}, {n} rows")
    return out


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out_dir, arms = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    present = [a for a in arms if os.path.isdir(f"{ROOT}/{a}")]
    if missing := [a for a in arms if a not in present]:
        print(f"  no run directory under {ROOT}: {missing}")
    print(f"  root {ROOT}, dataset {DATASET}, {len(present)} arm(s)")

    write(f"{out_dir}/poly_training_loss.csv", training_loss(present))
    write(f"{out_dir}/poly_wilson_params.csv", wilson_params(present))
    write(f"{out_dir}/poly_spectrum.csv", spectrum(present))
    write(f"{out_dir}/poly_merging_shells.csv", merging_shells(present))
    write(f"{out_dir}/poly_refinement.csv", refinement(present))
    write(f"{out_dir}/poly_anomalous_peaks.csv", anomalous_peaks(present))
    write(f"{out_dir}/poly_ccanom.csv", ccanom(present))
    write(f"{out_dir}/careless_stats_poly.csv", careless_stats(present))


if __name__ == "__main__":
    main()
