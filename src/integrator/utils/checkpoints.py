"""Which checkpoints of a run the downstream chain should be scored at.

Both arms ask the same question and used to answer it twice, in Python for
the polychromatic driver and in shell for the monochromatic one. The answer
differs only in the default, and the default follows from the cost of the
chain rather than from anything about the model:

- **Monochromatic** scales and merges with DIALS and refines with phenix, all
  on CPU, so scoring every n-th checkpoint is affordable and a trajectory is
  worth more than an endpoint.
- **Polychromatic** scales with careless, which is tens of GPU-minutes per
  checkpoint, so the default is the last one alone and anything more is a
  deliberate choice made once the data to test is settled.

Usable from the shell, one path per line:

    python -m integrator.utils.checkpoints <run-dir> --scope every:5
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

__all__ = ["epoch_of", "log_dir_of", "select"]


def epoch_of(path: Path) -> int:
    """Epoch parsed from a checkpoint filename, or -1 when it carries none."""
    match = re.search(r"epoch[=_](\d+)", Path(path).name)
    return int(match.group(1)) if match else -1


def log_dir_of(run_dir: Path) -> Path:
    """Where a run writes its checkpoints.

    With W&B the output root moves under the wandb run directory, so
    `<run_dir>/files` is only right for local logging; `run_paths.yaml`
    carries the real one.
    """
    run_dir = Path(run_dir)
    rp = run_dir / "run_paths.yaml"
    if rp.is_file():
        meta = yaml.safe_load(rp.read_text()) or {}
        if meta.get("log_dir"):
            return Path(meta["log_dir"])
        if meta.get("output_root"):
            return Path(meta["output_root"]) / "files"
    return run_dir / "files"


def select(run_dir: Path, scope: str | int | None = "last") -> list[Path]:
    """The checkpoints `scope` names, in epoch order.

        "last"      the highest epoch; the default
        "all"       every checkpoint
        "every:n"   checkpoints at least n epochs apart, starting from the
                    first, and always the last one, so a trajectory never
                    stops short of where the run ended
        <int>       that one epoch

    `every:n` is a stride, not a divisibility test. A run that checkpoints
    every 5 epochs from epoch 4 has them at 4, 9, ..., 49, and none of those
    is a multiple of 5; keeping `epoch % n == 0` would return the last
    checkpoint alone and quietly look like it had worked.

    `last.ckpt` is excluded throughout: it is a copy of some epoch's file
    under a name that carries no epoch, so counting it would score one
    checkpoint twice and label one of them -1.
    """
    log_dir = log_dir_of(run_dir)
    ckpts = sorted(
        (c for c in log_dir.glob("**/epoch*.ckpt") if c.name != "last.ckpt"),
        key=epoch_of,
    )
    if not ckpts:
        return []
    if scope == "all":
        return ckpts
    if scope in (None, "last"):
        return [ckpts[-1]]
    if isinstance(scope, str) and scope.startswith("every:"):
        n = int(scope.split(":", 1)[1])
        if n < 1:
            raise ValueError(f"every:n needs n >= 1, got {scope!r}")
        keep, last_kept = [], None
        for c in ckpts:
            e = epoch_of(c)
            if last_kept is None or e - last_kept >= n:
                keep.append(c)
                last_kept = e
        if ckpts[-1] not in keep:
            keep.append(ckpts[-1])
        return keep
    wanted = int(scope)
    match = [c for c in ckpts if epoch_of(c) == wanted]
    if not match:
        raise ValueError(f"no checkpoint for epoch {wanted} under {log_dir}")
    return match


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Select a run's checkpoints")
    p.add_argument("run_dir", type=Path)
    p.add_argument(
        "--scope", default="last",
        help="last | all | every:<n> | <epoch>  (default: last)",
    )
    p.add_argument(
        "--epochs", action="store_true", help="Print epoch numbers, not paths"
    )
    args = p.parse_args(argv)
    got = select(args.run_dir, args.scope)
    if not got:
        print(f"no checkpoints under {log_dir_of(args.run_dir)}", file=sys.stderr)
        return 1
    for c in got:
        print(epoch_of(c) if args.epochs else c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
