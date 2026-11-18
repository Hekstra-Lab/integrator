"""Reduce a monochromatic run to the long-format tables its figures draw.

Runs on the cluster, where the logs and the exports are; writes CSVs small
enough to pull down. Rendering happens on the laptop because the fonts the
documents are set in live there, so nothing larger than a few thousand rows
should cross the wire.

Four kinds, all with `method` and `dataset` key columns:

`training_loss.csv`
    epoch, split (train | val), term (loss | nll | kl | kl_prf | kl_i |
    kl_bg), value. From the run's `plots/loss_history.csv`, which carries no
    test split: the test reflections are the exported ones, so a test loss
    needs an evaluation pass rather than a column.

`wilson_params.csv`
    epoch, parameter, value -- the Wilson prior's learned scale and
    B-factor, from `plots/wilson_params.csv`. Long format because a
    B-factor in square angstroms and a scale in counts do not share an axis.

`merging_shells.csv`
    epoch, estimator, d_max, d_min, and the per-shell columns of the table
    `dials.merge` prints: n_obs, n_uniq, multiplicity, completeness, i_mean,
    i_over_sigma, r_merge, r_meas, r_pim, r_anom, cc_half, cc_anom.
    `dials_ref` is the reference integrator, which has no epoch.

`refinement.csv`
    epoch, estimator, variant (F | I), r_work, r_free. Both variants are
    always written: a merged MTZ carries F(+)/SigF(+) and I(+)/SigI(+), they
    should agree, and a gap between them is an export defect.

`anomalous_peaks.csv`
    epoch, estimator, variant, site, chain, residue, seqid, atom, element,
    peak_height, dist. From the refinement's `peaks.csv`, with the atom
    carrying each peak resolved against the refined PDB beside it -- a
    sulfur peak is SD on a methionine and SG on a cysteine, and the residue
    name alone does not say which. `site` is the panel title, `IOD 204 I`.
    The older `peak_heights.csv` is read where that is all a refinement
    wrote, and those rows have no chain, seqid or atom.

Usage:
    python tidy_mono.py <out-dir> <arm> [arm ...]
    EPOCH_RUNS=1 python tidy_mono.py ...   # arms that are checkpoints of one run
"""

import csv
import glob
import os
import re
import sys

from integrator.utils import metrics

S = "/n/netscratch/hekstra_lab/Lab/laldama"
DATASET = os.environ.get("DATASET", "hewl_9b7c")
# The estimators, variants and the `dials.merge` column order are defined
# once, in `integrator.utils.metrics`, with the regular expressions that read
# the chain's own output. They were duplicated here and in `plot_merging.py`,
# which is how two figures came to describe one run differently.
ESTIMATORS = metrics.ESTIMATORS
VARIANTS = metrics.VARIANTS
SHELL_COLS = metrics.SHELL_COLUMNS
# the ELBO terms, which are this script's own: they come from the training
# loop, not from any of the downstream tools
TERMS = ("loss", "nll", "kl", "kl_prf", "kl_i", "kl_bg")
# the element carrying the anomalous signal, from the residue name
ELEMENTS = {"CYS": "S", "MET": "S", "IOD": "I"}


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


def latest_log(arm: str, est: str) -> str | None:
    hits = sorted(
        glob.glob(f"{S}/dialsarm_{arm}_{est}_[0-9]*.log"), key=os.path.getmtime
    )
    if not hits and est == "rb":
        hits = sorted(
            glob.glob(f"{S}/dialsarm_{arm}_[0-9]*.log"), key=os.path.getmtime
        ) or sorted(
            glob.glob(f"{S}/dialsarm_{arm.replace('_', '')}_[0-9]*.log"),
            key=os.path.getmtime,
        )
    return hits[-1] if hits else None


def training_loss(arms: list[str]) -> list[dict]:
    out = []
    for arm in arms:
        path = f"{S}/sim_runs/{arm}/plots/loss_history.csv"
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for row in csv.DictReader(fh):
                for split in ("train", "val"):
                    for term in TERMS:
                        v = row.get(f"{split}_{term}", "")
                        if v in ("", None):
                            continue
                        out.append({
                            "method": arm,
                            "dataset": DATASET,
                            "epoch": int(float(row["epoch"])),
                            "split": split,
                            "term": term,
                            "value": float(v),
                        })
    return out


def wilson_params(arms: list[str]) -> list[dict]:
    """The Wilson prior's learned scale and B-factor per epoch.

    From the run's `plots/wilson_params.csv`, written every epoch by
    `WilsonParamLogger`, which reads the loss's own `get_G()` and `get_B()`
    rather than a stored copy. The file has one column per parameter; this is
    melted to one row per (parameter, epoch) because a B-factor in square
    angstroms and a scale in counts do not share an axis.

    `G_mean` is the mean over scale groups where there is more than one, and
    the logger then also writes `G_min` and `G_max`: the spread across groups
    says how much the crystals differ, which a single G would hide. All of
    them are carried through under their own names.
    """
    out = []
    for arm in arms:
        path = f"{S}/sim_runs/{arm}/plots/wilson_params.csv"
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for row in csv.DictReader(fh):
                epoch = row.get("epoch", "")
                for name, value in row.items():
                    if name == "epoch" or value in ("", None):
                        continue
                    try:
                        v = float(value)
                    except ValueError:
                        continue
                    out.append({
                        "method": arm,
                        "dataset": DATASET,
                        "epoch": int(float(epoch)),
                        "parameter": name,
                        "value": v,
                    })
    return out


def shell_rows(log: str) -> list[dict]:
    """The per-shell table `dials.merge` prints, already typed.

    `metrics.shells_from_merge_log` strips the `*` that marks a significant
    correlation and the `%` on completeness, so `cc_half` and `cc_anom` come
    back as floats rather than as strings that sort as text.
    """
    return metrics.shells_from_merge_log(log)


def merging_shells(arms: list[str], epoch_of) -> list[dict]:
    out = []
    for arm in arms:
        for est in ESTIMATORS:
            merged = f"{S}/dials9b7c/{arm}_{est}/merge.log"
            if not os.path.exists(merged):
                # the reference integrator was merged without an estimator
                # suffix, because it has only one set of intensities
                merged = f"{S}/dials9b7c/{arm}/merge.log"
                if est != "rb" or not os.path.exists(merged):
                    continue
            for shell in shell_rows(merged):
                row = {"method": arm, "dataset": DATASET, "estimator": est}
                e = epoch_of(arm)
                if e is not None:
                    row["epoch"] = e
                row.update({k: shell[k] for k in SHELL_COLS})
                out.append(row)
    return out


def refinement(arms: list[str], epoch_of) -> list[dict]:
    out = []
    for arm in arms:
        for est in ESTIMATORS:
            log = latest_log(arm, est)
            if log is None:
                continue
            with open(log, errors="replace") as fh:
                text = fh.read()
            # the chain appends on a rerun, so the LAST line per variant is
            # the result; `metrics` keeps that one by default
            for got in metrics.parse_result_lines(text):
                row = {
                    "method": arm,
                    "dataset": DATASET,
                    "estimator": est,
                    "variant": got["variant"],
                    "r_work": got["r_work"],
                    "r_free": got["r_free"],
                }
                e = epoch_of(arm)
                if e is not None:
                    row["epoch"] = e
                out.append(row)
    return out


def refined_model(d: str) -> str | None:
    """The refined PDB in a refinement directory.

    The same glob the peak search itself used (`rs.find_peaks *[0-9].mtz
    *[0-9].pdb`), so the model the atom names are read from is the model the
    peaks were found in. The newest wins when a directory holds several
    rounds.
    """
    hits = sorted(glob.glob(f"{d}/*[0-9].pdb"), key=os.path.getmtime)
    return hits[-1] if hits else None


def anomalous_peaks(arms: list[str], epoch_of) -> list[dict]:
    """Peak heights per site, with the atom carrying each peak.

    `peaks.csv` is preferred over `peak_heights.csv`: it has one row per
    site rather than a transposed pair of rows, and it carries the chain,
    the residue number and the peak's own position. The atom name is
    resolved from the refined model beside it, because the fork of
    `rs.find_peaks` in use writes its `name` column empty -- and the residue
    does not say which atom holds the peak, since a methionine sulfur is SD
    and a cysteine sulfur is SG.

    `peak_heights.csv` is the fallback for the older refinements that wrote
    only that file; those rows carry no chain and no atom.
    """
    out = []
    for arm in arms:
        for est in ESTIMATORS:
            for variant in VARIANTS:
                d = f"{S}/refine9b7c/{arm}_{est}_{variant}"
                key = {"method": arm, "dataset": DATASET,
                       "estimator": est, "variant": variant}
                e = epoch_of(arm)
                if e is not None:
                    key["epoch"] = e
                if os.path.exists(f"{d}/peaks.csv"):
                    out += _peaks_rows(d, key)
                elif os.path.exists(f"{d}/peak_heights.csv"):
                    out += _peak_height_rows(d, key)
    return out


def _peaks_rows(d: str, key: dict) -> list[dict]:
    """One row per site from `peaks.csv`, atom names from the refined model."""
    model = refined_model(d)
    if model is None:
        print(f"  {os.path.basename(d)}: no refined PDB; atom names omitted")
    rows = []
    for peak in metrics.peaks_from_find_peaks_csv(f"{d}/peaks.csv", model=model):
        rows.append({
            **key,
            "site": peak["label"],
            "chain": peak["chain"],
            "residue": peak["residue"],
            "seqid": peak["seqid"],
            "atom": peak["atom"],
            # the element is read from the model when there is one, and
            # otherwise from the residue name, which is all `peak_heights.csv`
            # ever carried
            "element": peak["element"] or ELEMENTS.get(peak["residue"], ""),
            "peak_height": peak["peakz"],
            "dist": peak["dist"],
            "atom_from_position": peak["atom_from_position"],
        })
    return rows


def _peak_height_rows(d: str, key: dict) -> list[dict]:
    """The transposed two-row `peak_heights.csv`, which has no positions.

    The atom is then the heaviest one in the residue the peak search named
    rather than the atom nearest the peak, so `atom_from_position` is False
    on every row and `dist` is empty: this file carries neither.
    """
    model = refined_model(d)
    rows = []
    for peak in metrics.peak_heights_from_csv(
        f"{d}/peak_heights.csv", model=model
    ):
        rows.append({
            **key,
            "site": peak["label"],
            "chain": peak["chain"],
            "residue": peak["residue"],
            "seqid": peak["seqid"] if peak["seqid"] is not None else "",
            "atom": peak["atom"],
            "element": peak["element"] or ELEMENTS.get(peak["residue"], ""),
            "peak_height": peak["peakz"],
            "dist": "",
            "atom_from_position": False,
        })
    return rows


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out_dir, arms = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)

    # An arm name ending in `_ep<N>` is one checkpoint of a trajectory; the
    # epoch then belongs in the table and the arms are one method.
    pat = re.compile(r"^(.*)_ep(\d+)$")

    def epoch_of(arm: str):
        m = pat.match(arm)
        return int(m.group(2)) if m else None

    write(f"{out_dir}/training_loss.csv", training_loss(arms))
    write(f"{out_dir}/wilson_params.csv", wilson_params(arms))
    write(f"{out_dir}/merging_shells.csv", merging_shells(arms, epoch_of))
    write(f"{out_dir}/refinement.csv", refinement(arms, epoch_of))
    write(f"{out_dir}/anomalous_peaks.csv", anomalous_peaks(arms, epoch_of))


if __name__ == "__main__":
    main()
