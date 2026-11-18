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
import sys

EPOCH_RE = re.compile(r"epoch=?_?(\d+)")
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


def gpu_of(run_dir: str) -> str:
    """The GPU the run reported, where a log recorded it."""
    for path in sorted(glob.glob(f"{run_dir}/*.out"))[:1]:
        try:
            head = open(path, errors="replace").read(4000)
        except OSError:
            continue
        m = re.search(r"(NVIDIA [^\n,]+|[Aa]100[^\n,]*|H100[^\n,]*)", head)
        if m:
            return m.group(1).strip()
    return ""


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
    print(f"  {arm}: {integrator}/{qp}, {params['total']:,} params, "
          f"{rate:.1f} s/epoch over {n_intervals} interval(s)")
    return out


def main() -> None:
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
