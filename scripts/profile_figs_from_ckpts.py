#!/usr/bin/env python
"""Render basis and tracked-profile figures from a finished run's checkpoints.

A thin wrapper over `make_training_figures.py` (Route B) that removes the two
things which stop it working on real run directories:

* It finds `config_log.yaml` and `files/checkpoints/` itself, rather than
  requiring a `config:` key in `run_paths.yaml`. The downstream chain scripts
  overwrite that file with just `predictions_dir`, so runs that have been
  scored cannot be replayed through the documented path.

* It drops config arguments the current code no longer accepts, so a run from
  before an argument was removed still replays. Deleting `smoothness_weight`
  made every earlier run unloadable: `construct_integrator` rejects unknown
  surrogate args, which is right for training and useless for forensics.

Usage:

    uv run python scripts/profile_figs_from_ckpts.py RUN_DIR [RUN_DIR ...]
    uv run python scripts/profile_figs_from_ckpts.py RUN_A RUN_B --compare

`--compare` additionally writes one side-by-side panel of the arms' final
profiles, which is the form that actually answers "did this change anything".
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _valid_args(kind: str, name: str) -> set[str] | None:
    """Argument names the current code accepts for a registry entry."""
    try:
        from integrator.registry import REGISTRY
        from integrator.utils.factory_utils import _constructor_arg_names

        target = REGISTRY[kind][name]
    except Exception:
        return None
    try:
        return _constructor_arg_names(target)
    except Exception:
        return None


def sanitize(cfg: dict) -> tuple[dict, list[str]]:
    """Drop args the current code would reject, reporting what went.

    Only touches surrogates and the loss, where arguments are validated
    strictly; everything else is left alone.
    """
    dropped: list[str] = []
    for kind, section in (("surrogates", "surrogates"), ("losses", "loss")):
        node = cfg.get(section)
        if not node:
            continue
        entries = node.items() if section == "surrogates" else [(None, node)]
        for key, entry in entries:
            if not isinstance(entry, dict):
                continue
            name, args = entry.get("name"), entry.get("args")
            if not name or not isinstance(args, dict):
                continue
            ok = _valid_args(kind, name)
            if ok is None:
                continue
            for bad in sorted(set(args) - ok):
                args.pop(bad)
                label = f"{key}:{name}" if key else name
                dropped.append(f"{label}.{bad}")
    return cfg, dropped


def run_one(run_dir: Path, out_dir: Path | None, stages: str,
            n_per_regime: int, subdir: bool = False) -> Path | None:
    import yaml

    cfg_path = run_dir / "config_log.yaml"
    ckpt_dir = run_dir / "files" / "checkpoints"
    if not cfg_path.is_file():
        print(f"  {run_dir.name}: no config_log.yaml, skipping")
        return None
    n_ckpt = len(list(ckpt_dir.glob("*.ckpt"))) if ckpt_dir.is_dir() else 0
    if not n_ckpt:
        print(f"  {run_dir.name}: no checkpoints, skipping")
        return None

    cfg, dropped = sanitize(yaml.safe_load(cfg_path.read_text()))
    patched = run_dir / "config_log.replay.yaml"
    yaml.safe_dump(cfg, patched.open("w"), sort_keys=False)
    note = f"  (dropped {', '.join(dropped)})" if dropped else ""
    print(f"  {run_dir.name}: {n_ckpt} checkpoints{note}")

    # With several runs sharing one --out-dir, each needs its own subdirectory
    # or the last one silently overwrites the rest. The parent is part of the
    # name because the same arm reruns under the same basename.
    if out_dir is not None and subdir:
        out = out_dir / f"{run_dir.parent.name}_{run_dir.name}"
    else:
        out = out_dir or (run_dir / "figures")
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(REPO / "scripts" / "make_training_figures.py"),
        "--config", str(patched), "--ckpt", str(ckpt_dir),
        "--stages", stages, "--n-per-regime", str(n_per_regime),
        "--out-dir", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
        print(f"    FAILED: {' / '.join(tail)}")
        return None
    pngs = sorted(out.glob("*.png"))
    print(f"    wrote {len(pngs)} figures to {out}")
    return out


def compare(outs: list[tuple[str, Path]], dest: Path) -> None:
    """One panel: each run's final tracked profiles, same rows, same scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    strips = [(n, o / "fig_tracked_filmstrip_profile.png") for n, o in outs]
    strips = [(n, p) for n, p in strips if p.is_file()]
    if len(strips) < 2:
        print("  need two filmstrips to compare; skipping")
        return
    fig, axes = plt.subplots(1, len(strips), figsize=(7 * len(strips), 9))
    for ax, (name, p) in zip(axes, strips):
        ax.imshow(mpimg.imread(p))
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(dest, dpi=130, bbox_inches="tight")
    print(f"  comparison -> {dest}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: <run_dir>/figures for each run")
    p.add_argument("--stages", default="basis,tracked")
    p.add_argument("--n-per-regime", type=int, default=4)
    p.add_argument("--compare", action="store_true",
                   help="also write one side-by-side panel of the filmstrips")
    a = p.parse_args()

    produced: list[tuple[str, Path]] = []
    many = len(a.run_dirs) > 1
    for rd in a.run_dirs:
        out = run_one(rd, a.out_dir, a.stages, a.n_per_regime, subdir=many)
        if out is not None:
            produced.append((f"{rd.parent.name}/{rd.name}", out))
    if a.compare and len(produced) >= 2:
        dest = (a.out_dir or produced[0][1]) / "fig_compare_profiles.png"
        compare(produced, dest)


if __name__ == "__main__":
    main()
