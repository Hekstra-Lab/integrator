"""A run directory resolves under either layout.

`integrator.cli.train --run-dir` writes config_log.yaml beside files/, while
the older wandb driver wrote run_metadata.yaml carrying `config` and
`wandb.log_dir`. `pred` and `run_phenix_pipeline` read only the latter, so a
run from the current CLI failed with `FileNotFoundError: run_metadata.yaml`
or `KeyError: 'wandb'` with everything they needed already on disk.
"""

import pytest
import yaml

from integrator.cli.utils.run_layout import resolve_run


def _current_layout(root):
    (root / "config_log.yaml").write_text(yaml.safe_dump({"mode": "mono"}))
    (root / "files" / "checkpoints").mkdir(parents=True)
    (root / "files" / "checkpoints" / "epoch=0001.ckpt").write_text("x")
    return root


def _legacy_layout(root, log_dir):
    (log_dir / "checkpoints").mkdir(parents=True)
    cfg = root / "elsewhere.yaml"
    cfg.write_text(yaml.safe_dump({"mode": "mono"}))
    (root / "run_metadata.yaml").write_text(
        yaml.safe_dump({"config": str(cfg), "wandb": {"log_dir": str(log_dir)}})
    )
    return cfg


def test_current_train_layout(tmp_path):
    _current_layout(tmp_path)
    config, log_dir = resolve_run(tmp_path)
    assert config == tmp_path / "config_log.yaml"
    assert log_dir == tmp_path / "files"
    # callers derive the predictions dir from the parent; both layouts agree
    assert log_dir.parent == tmp_path


def test_legacy_wandb_layout_is_unchanged(tmp_path):
    log_dir = tmp_path / "wandb" / "run-xyz" / "files"
    cfg = _legacy_layout(tmp_path, log_dir)
    config, resolved = resolve_run(tmp_path)
    assert config == cfg
    assert resolved == log_dir


def test_legacy_wins_when_both_are_present(tmp_path):
    """An explicit run_metadata.yaml keeps resolving exactly as before."""
    _current_layout(tmp_path)
    log_dir = tmp_path / "wandb" / "run-xyz" / "files"
    cfg = _legacy_layout(tmp_path, log_dir)
    config, resolved = resolve_run(tmp_path)
    assert (config, resolved) == (cfg, log_dir)


def test_partial_metadata_falls_through(tmp_path):
    """A half-written metadata must not block an otherwise usable run."""
    _current_layout(tmp_path)
    (tmp_path / "run_metadata.yaml").write_text(yaml.safe_dump({"config": "x"}))
    config, log_dir = resolve_run(tmp_path)
    assert config == tmp_path / "config_log.yaml"
    assert log_dir == tmp_path / "files"


def test_unrecognizable_directory_says_what_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="config_log.yaml"):
        resolve_run(tmp_path)
