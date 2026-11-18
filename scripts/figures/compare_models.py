"""Parameter counts and training throughput, per run, for comparing families.

Answers "how big is an svae against a hierarchical integrator, and what does
each cost to train" without retraining anything: both numbers are recoverable
from a run directory that already exists.

PARAMETERS come from the checkpoint's state dict, summed per top-level
module as well as in total, because the interesting comparison is usually
where the parameters sit rather than how many there are -- an svae and a
hierarchical model can match on the total and differ entirely in whether the
mass is in the encoders or the decoder.

TRAINING TIME is measured from the checkpoint file timestamps: consecutive
checkpoints are a fixed number of epochs apart, so the gap between them
divided by that stride is seconds per epoch. The MEDIAN over the intervals
is reported, not the mean, because a requeued or preempted job leaves one
enormous gap that a mean would absorb and report as a slow model. The number
of intervals is reported beside it so a run with too few to be meaningful is
visible rather than silently thin.

Two things this does NOT measure, and both need a timed run rather than a
file on disk:

- INFERENCE time. Nothing records it. It needs a predict pass with a clock
  around it, on fixed hardware.
- Training throughput ACROSS runs on different GPUs. The seconds per epoch
  below are only comparable between runs that had the same hardware and the
  same dataset. Two arms of one ablation are comparable; an arm on 9b7c and
  an arm on 1118 are not, and neither are two runs that landed on different
  nodes. `gpu` is reported where the run recorded it so this can be checked.

Usage:
    python compare_models.py <out-csv> <run-dir> [run-dir ...]
    DATASET=hewl_9b7c python compare_models.py ...
"""

import csv
import glob
import os
import re
import subprocess
import sys

EPOCH_RE = re.compile(r"epoch=?_?(\d+)")
# No default, and empty is refused in main(). `method` and `dataset` are the
# key columns every file here carries, and a blank one does not merely look
# untidy: concatenating these tables for a figure then joins rows from
# different datasets under one empty key.
DATASET = os.environ.get("DATASET", "")
# the top-level module names worth reporting separately; anything else is
# summed into "other" rather than dropped
MODULES = ("encoders", "surrogates", "loss", "decoder", "scale")


def epoch_of(path: str) -> int | None:
    m = EPOCH_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def checkpoints(run_dir: str) -> list[tuple[int, str]]:
    """`(epoch, path)` for every checkpoint naming an epoch, in order."""
    out = []
    for p in glob.glob(f"{run_dir}/files/checkpoints/*.ckpt"):
        e = epoch_of(p)
        if e is not None:
            out.append((e, p))
    return sorted(out)


def config_of(run_dir: str) -> dict:
    path = f"{run_dir}/config_log.yaml"
    if not os.path.exists(path):
        return {}
    import yaml

    return yaml.safe_load(open(path).read()) or {}


def names(cfg: dict) -> tuple[str, str]:
    """The integrator and the profile surrogate this run trained."""
    integrator = str((cfg.get("integrator") or {}).get("name", ""))
    qp = str(((cfg.get("surrogates") or {}).get("qp") or {}).get("name", ""))
    return integrator, qp


def parameters(ckpt: str) -> dict[str, int]:
    """Parameter count in total and per top-level module.

    Counted from the state dict rather than by constructing the model: that
    needs no config resolution, no GPU and no data, and it counts what was
    actually trained rather than what a config says would be built.
    """
    import torch

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = blob.get("state_dict", blob)
    out: dict[str, int] = {}
    total = 0
    for key, value in sd.items():
        try:
            n = int(value.numel())
        except AttributeError:
            continue  # a buffer that is not a tensor
        total += n
        head = key.split(".", 1)[0]
        bucket = head if head in MODULES else "other"
        out[bucket] = out.get(bucket, 0) + n
    out["total"] = total
    return out


def seconds_per_epoch(ckpts: list[tuple[int, str]]) -> tuple[float, int]:
    """Median seconds per epoch from the checkpoint timestamps.

    Median, not mean: a requeue leaves one huge interval, and a mean would
    report that as the model being slow.
    """
    rates = []
    for (e0, p0), (e1, p1) in zip(ckpts, ckpts[1:], strict=False):
        span = e1 - e0
        gap = os.path.getmtime(p1) - os.path.getmtime(p0)
        if span > 0 and gap > 0:
            rates.append(gap / span)
    if not rates:
        return float("nan"), 0
    rates.sort()
    mid = len(rates) // 2
    median = (rates[mid] if len(rates) % 2
              else 0.5 * (rates[mid - 1] + rates[mid]))
    return median, len(rates)


def logs_of(run_dir: str) -> list[str]:
    """The run's SLURM logs, wherever the submitting script put them.

    Not always beside the run: the PDZ2 scripts write to a shared `logs/`
    one level up from `runs/`, named `<arm>_<jobid>.out`. Looking only in
    the run directory is why `seconds_per_epoch` was reported for months
    with no hardware beside it, and an unmatched pair -- 8 CPUs against 16,
    two months apart -- read as a model difference.
    """
    arm = os.path.basename(run_dir.rstrip("/"))
    up = os.path.dirname(os.path.dirname(run_dir.rstrip("/")))
    # A log is not always named after its run directory: `runs/pdz2_w21_k7m`
    # was submitted as `train_w21_k7m_<jobid>.out`, the dataset tag being in
    # the path rather than the job name. Both spellings are tried.
    parent = os.path.dirname(run_dir.rstrip("/"))
    cands = [arm] + ([arm.split("_", 1)[1]] if "_" in arm else [])
    found = sorted(glob.glob(f"{run_dir}/*.out"))
    # the immediate parent as well as one above it: PDZ2 puts runs in
    # `<base>/runs/<arm>` and logs in `<base>/logs`, while a job submitted
    # from the directory holding the runs leaves its log right beside them
    for cand in cands:
        for pattern in (f"{run_dir}/logs/*{cand}*.out",
                        f"{parent}/logs/*{cand}*.out",
                        f"{parent}/*{cand}*.out",
                        f"{up}/logs/*{cand}*.out",
                        f"{up}/*{cand}*.out"):
            found += sorted(glob.glob(pattern))
    # an arm name is a prefix of others (`w21_k7m` of `w21_k7m_b`), so keep
    # only logs whose name is the arm plus a job id and nothing else
    keep = []
    for f in found:
        base = os.path.basename(f)
        if any(re.fullmatch(rf"{re.escape(c)}[_.]?\d*\.out"
                            rf"|.*_{re.escape(c)}_\d+\.out", base)
               for c in cands):
            keep.append(f)
    return list(dict.fromkeys(keep)) or found


def gpu_of(run_dir: str) -> str:
    """The GPU the run reported, where a log recorded it."""
    for path in logs_of(run_dir)[:2]:
        try:
            head = open(path, errors="replace").read(4000)
        except OSError:
            continue
        m = re.search(r"(NVIDIA [^\n,]+|[Aa]100[^\n,]*|H100[^\n,]*)", head)
        if m:
            return m.group(1).strip()
    return ""


def allocation_of(run_dir: str) -> dict[str, str]:
    """Node and CPU count from SLURM accounting, via the log's job id.

    Seconds per epoch is only a model number when the allocation is the
    same, and on this cluster it frequently is not: data loading scales
    with `-c`, and the partitions mix A100, H100 and H200. Recording the
    node and the CPU count is what makes a pair checkable rather than
    assumed. Returns empty off-cluster, where `sacct` does not exist.
    """
    ids = []
    for path in logs_of(run_dir):
        m = re.search(r"(\d{6,})\.out$", os.path.basename(path))
        if m:
            ids.append(m.group(1))
    if not ids:
        return {}
    try:
        out = subprocess.run(
            ["sacct", "-j", ",".join(sorted(set(ids))), "--parsable2",
             "--noheader", "-o", "JobID,NodeList,ReqTRES,Elapsed"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 4 or "." in parts[0]:
            continue  # a step, not the job
        node, tres = parts[1], parts[2]
        cpu = re.search(r"cpu=(\d+)", tres)
        if node and node != "None assigned":
            return {"node": node, "cpus": cpu.group(1) if cpu else ""}
    return {}


def rows_for(run_dir: str) -> list[dict]:
    arm = os.path.basename(run_dir.rstrip("/"))
    ckpts = checkpoints(run_dir)
    if not ckpts:
        print(f"  {arm}: no checkpoints naming an epoch")
        return []
    cfg = config_of(run_dir)
    integrator, qp = names(cfg)
    params = parameters(ckpts[-1][1])
    rate, n_intervals = seconds_per_epoch(ckpts)

    common = {
        "method": arm,
        "dataset": DATASET,
        "integrator": integrator,
        "qp": qp,
    }
    out = []
    for bucket, n in sorted(params.items()):
        out.append({**common, "quantity": f"n_params_{bucket}",
                    "value": n, "unit": "parameters"})
    out.append({**common, "quantity": "n_epochs",
                "value": ckpts[-1][0] + 1, "unit": "epochs"})
    out.append({**common, "quantity": "n_checkpoints",
                "value": len(ckpts), "unit": "checkpoints"})
    out.append({**common, "quantity": "seconds_per_epoch",
                "value": round(rate, 2), "unit": "s"})
    out.append({**common, "quantity": "timing_intervals",
                "value": n_intervals, "unit": "intervals"})
    gpu = gpu_of(run_dir)
    if gpu:
        out.append({**common, "quantity": "gpu", "value": gpu, "unit": ""})
    alloc = allocation_of(run_dir)
    for key, value in alloc.items():
        if value:
            out.append({**common, "quantity": key, "value": value,
                        "unit": "cpus" if key == "cpus" else ""})
    where = " ".join(filter(None, [gpu, alloc.get("node", ""),
                                   f"-c {alloc['cpus']}" if alloc.get("cpus")
                                   else ""]))
    print(f"  {arm}: {integrator}/{qp}, {params['total']:,} params, "
          f"{rate:.1f} s/epoch over {n_intervals} interval(s)"
          + (f"  [{where}]" if where else "  [hardware unrecorded]"))
    return out


def main() -> None:
    if not DATASET:
        raise SystemExit(
            "set DATASET to the dataset these runs trained on, e.g.\n"
            "    DATASET=pdz2_1116 python compare_models.py out.csv <run-dir>...\n"
            "It is a key column in the table written here and cannot be "
            "guessed from the run directory."
        )
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out_csv, runs = sys.argv[1], sys.argv[2:]
    rows = []
    for run in runs:
        if not os.path.isdir(run):
            print(f"  no run directory: {run}")
            continue
        rows += rows_for(run)
    if not rows:
        raise SystemExit("nothing to write")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  {os.path.basename(out_csv)}: {len(rows)} rows")


if __name__ == "__main__":
    main()
