"""Average a run's checkpoints over several epochs into one checkpoint that predict and rb_export.sbatch read.

Every floating-point tensor of the state_dict is the plain mean over the given
epochs, except the loss's globals (`loss.*`, the Wilson G and B), which are
taken from the last epoch given, as are integer buffers. Optimizer, scheduler
and callback state are dropped. The result is written as
`<out>/files/checkpoints/epoch=<last>.ckpt` beside a copy of the run's
`config_log.yaml`, the layout `scripts/poly/pdz2/rb_export.sbatch` expects of
an entry of RB_RUNS.

Why: with a constant learning rate the last step moves the narrowest profile
component's weight by about its own run-to-run spread, and faint-reflection
bias follows that weight; averaging epochs 39, 44, 49, 54, 59 cut the
replicate spread of that bias from 0.026 to 0.004 on mos010 (B1, five runs).
This is the averaging those mos010 `_avg` runs used.

    python average_checkpoints.py RUN_DIR OUT_DIR [--epochs 39,44,49,54,59]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


def average(paths: list[Path]) -> dict:
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    last = cks[-1]
    sd = last["state_dict"]
    for c in cks[:-1]:
        if c["state_dict"].keys() != sd.keys():
            raise SystemExit("checkpoints differ in their state_dict keys")
    last["state_dict"] = {
        k: (v if (k.startswith("loss.") or not v.is_floating_point())
            else torch.stack([c["state_dict"][k] for c in cks]).mean(0))
        for k, v in sd.items()
    }
    for k in ("optimizer_states", "lr_schedulers", "callbacks"):
        last.pop(k, None)
    return last


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="a training run directory (files/checkpoints/epoch=NNNN.ckpt)")
    ap.add_argument("out", type=Path, help="the averaged run directory to write")
    ap.add_argument("--epochs", default="39,44,49,54,59")
    a = ap.parse_args()

    epochs = [int(e) for e in a.epochs.split(",")]
    paths = [a.run / "files" / "checkpoints" / f"epoch={e:04d}.ckpt" for e in epochs]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")
    dst = a.out / "files" / "checkpoints" / f"epoch={epochs[-1]:04d}.ckpt"
    if dst.exists():
        raise SystemExit(f"{dst} exists; refusing to overwrite")
    ck = average(paths)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, dst)
    shutil.copy(a.run / "config_log.yaml", a.out / "config_log.yaml")
    print(f"{a.run.name}: mean of epochs {epochs} (loss.* from {epochs[-1]}) -> {dst}")


if __name__ == "__main__":
    main()
