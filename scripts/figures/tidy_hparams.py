"""Hyperparameter tables for a set of runs, from their own config logs.

Runs on the cluster beside the other reducers. The reading is in
`integrator.utils.hparams`, so this file only locates the runs and writes
the two files out; the rules about what counts as a hyperparameter, how
repeated encoders collapse and how `varies` is computed live in one place
and are pinned by `tests/test_hparams.py`.

`varies` is a property of the SET, not of a run, so every arm that belongs
in one table has to be passed in one invocation. Running this twice over
halves of a comparison and concatenating the output gives a `varies` column
that is wrong in both halves.

Writes:

    hparams.csv         method, dataset, group, parameter, value, unit, varies
    hparam_labels.csv   parameter, header, unit, group, fmt, description

Usage:
    python tidy_hparams.py <out-dir> <run-dir> [run-dir ...]
    DATASET=pdz2_1116 python tidy_hparams.py ...
    VARYING_ONLY=1 python tidy_hparams.py ...   # drop the constant rows
"""

import csv
import os
import sys

from integrator.utils import hparams

DATASET = os.environ.get("DATASET", "")
VARYING_ONLY = os.environ.get("VARYING_ONLY", "") not in ("", "0", "false")


def write(path: str, rows: list[dict], columns) -> None:
    if not rows:
        print(f"  {os.path.basename(path)}: nothing to write")
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  {os.path.basename(path)}: {len(rows)} rows")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out_dir, runs = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)

    rows: list[dict] = []
    for run in runs:
        cfg = hparams.read_config_log(run)
        arm = os.path.basename(run.rstrip("/"))
        if not cfg:
            print(f"  {arm}: no config_log.yaml")
            continue
        got = hparams.hparam_rows(cfg, arm, DATASET)
        rows += got
        print(f"  {arm}: {len(got)} parameters")

    if not rows:
        raise SystemExit("no config logs read")
    hparams.mark_varying(rows)
    n_varying = len({r["parameter"] for r in rows if r["varies"]})
    print(f"  {n_varying} parameter(s) differ across "
          f"{len({r['method'] for r in rows})} arm(s)")

    kept = [r for r in rows if r["varies"]] if VARYING_ONLY else rows
    write(f"{out_dir}/hparams.csv", kept, hparams.COLUMNS)
    write(f"{out_dir}/hparam_labels.csv",
          hparams.hparam_labels(kept), hparams.LABEL_COLUMNS)


if __name__ == "__main__":
    main()
