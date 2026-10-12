"""Resolve a run directory to its config and log directory.

Two layouts exist. `integrator.cli.train --run-dir` writes

    <run>/config_log.yaml
    <run>/run_paths.yaml
    <run>/files/checkpoints/*.ckpt

while the older wandb-driven driver wrote

    <run>/run_metadata.yaml   # {config: <path>, wandb: {log_dir: <path>}}

`pred` and `run_phenix_pipeline` read only the second, so a run produced by
the current `train` fails with `FileNotFoundError: run_metadata.yaml` or
`KeyError: 'wandb'` even though everything they need is present. This reads
either, preferring an explicit `run_metadata.yaml` so existing runs keep
resolving exactly as before.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def resolve_run(run_dir: Path | str) -> tuple[Path, Path]:
    """Return `(config_path, log_dir)` for a run directory.

    `log_dir` is the directory holding `checkpoints/`; callers derive the
    predictions directory from its parent, which both layouts agree on.

    Raises:
        FileNotFoundError: Neither layout is present, naming what was looked
            for so the caller can see which half is missing.
    """
    run_dir = Path(run_dir)

    meta_path = run_dir / "run_metadata.yaml"
    if meta_path.is_file():
        meta = yaml.safe_load(meta_path.read_text()) or {}
        config = meta.get("config")
        log_dir = (meta.get("wandb") or {}).get("log_dir")
        if config and log_dir:
            return Path(config), Path(log_dir)
        # a partial file falls through to the current layout rather than
        # raising, so a half-written metadata does not block a usable run

    config_path = run_dir / "config_log.yaml"
    log_dir = run_dir / "files"
    if config_path.is_file() and (log_dir / "checkpoints").is_dir():
        return config_path, log_dir

    raise FileNotFoundError(
        f"{run_dir} is not a recognizable run directory: expected either "
        f"run_metadata.yaml with 'config' and 'wandb.log_dir', or "
        f"config_log.yaml beside files/checkpoints/. Found "
        f"config_log.yaml={config_path.is_file()}, "
        f"files/checkpoints={ (log_dir / 'checkpoints').is_dir() }, "
        f"run_metadata.yaml={meta_path.is_file()}."
    )
