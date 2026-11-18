"""Which checkpoints a downstream chain is run at.

The rule that matters: `every:n` is a stride in epochs, not a divisibility
test. Checkpoints do not land on multiples of n -- a run saving every 5
epochs from epoch 4 has them at 4, 9, ..., 49 -- and a divisibility test
returns the last checkpoint alone while looking like it worked.
"""

import pytest
import yaml

from integrator.utils import checkpoints


def make_run(tmp_path, epochs, with_last=True):
    """A run directory holding one checkpoint per epoch in `epochs`."""
    ckpt_dir = tmp_path / "files" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    for e in epochs:
        (ckpt_dir / f"epoch={e:04d}.ckpt").touch()
    if with_last:
        (ckpt_dir / "last.ckpt").touch()
    (tmp_path / "run_paths.yaml").write_text(
        yaml.safe_dump({"log_dir": str(tmp_path / "files")})
    )
    return tmp_path


def epochs_of(paths):
    return [checkpoints.epoch_of(p) for p in paths]


# the real 9b7c cadence: every 5 epochs, starting at 4
NINE_B7C = list(range(4, 50, 5))


def test_last_is_the_highest_epoch(tmp_path):
    run = make_run(tmp_path, NINE_B7C)
    assert epochs_of(checkpoints.select(run, "last")) == [49]


def test_default_scope_is_last(tmp_path):
    run = make_run(tmp_path, NINE_B7C)
    assert epochs_of(checkpoints.select(run)) == [49]


def test_all_returns_every_checkpoint_in_order(tmp_path):
    run = make_run(tmp_path, NINE_B7C)
    assert epochs_of(checkpoints.select(run, "all")) == NINE_B7C


def test_last_ckpt_is_never_selected(tmp_path):
    """`last.ckpt` carries no epoch, so counting it scores one twice."""
    run = make_run(tmp_path, NINE_B7C)
    for scope in ("last", "all", "every:10"):
        assert -1 not in epochs_of(checkpoints.select(run, scope))


def test_every_n_is_a_stride_not_a_divisibility_test(tmp_path):
    """None of 4, 9, ..., 49 is a multiple of 5 or 10."""
    run = make_run(tmp_path, NINE_B7C)
    assert epochs_of(checkpoints.select(run, "every:5")) == NINE_B7C
    assert epochs_of(checkpoints.select(run, "every:10")) == [
        4, 14, 24, 34, 44, 49
    ]
    assert epochs_of(checkpoints.select(run, "every:20")) == [4, 24, 44, 49]


def test_every_n_always_includes_the_last_epoch(tmp_path):
    """A trajectory must not stop short of where the run ended."""
    run = make_run(tmp_path, NINE_B7C)
    for n in range(1, 60):
        got = epochs_of(checkpoints.select(run, f"every:{n}"))
        assert got[-1] == 49, f"every:{n} dropped the final checkpoint"
        assert got == sorted(set(got)), f"every:{n} repeated or unsorted"


def test_every_n_spacing_is_at_least_n(tmp_path):
    run = make_run(tmp_path, NINE_B7C)
    for n in (2, 7, 13):
        got = epochs_of(checkpoints.select(run, f"every:{n}"))
        # the appended final epoch is allowed to sit closer than n
        gaps = [b - a for a, b in zip(got, got[1:], strict=False)][:-1]
        assert all(g >= n for g in gaps), f"every:{n} gave gaps {gaps}"


def test_an_integer_scope_picks_that_epoch(tmp_path):
    run = make_run(tmp_path, NINE_B7C)
    assert epochs_of(checkpoints.select(run, 24)) == [24]
    assert epochs_of(checkpoints.select(run, "24")) == [24]


def test_a_missing_epoch_raises_rather_than_falling_back(tmp_path):
    run = make_run(tmp_path, NINE_B7C)
    with pytest.raises(ValueError, match="no checkpoint for epoch 25"):
        checkpoints.select(run, 25)


def test_every_n_rejects_a_nonpositive_stride(tmp_path):
    run = make_run(tmp_path, NINE_B7C)
    with pytest.raises(ValueError, match="needs n >= 1"):
        checkpoints.select(run, "every:0")


def test_a_run_with_no_checkpoints_is_empty_not_an_error(tmp_path):
    run = make_run(tmp_path, [], with_last=False)
    assert checkpoints.select(run, "all") == []
    assert checkpoints.select(run, "last") == []


def test_log_dir_falls_back_when_run_paths_is_absent(tmp_path):
    run = make_run(tmp_path, NINE_B7C)
    (run / "run_paths.yaml").unlink()
    assert checkpoints.log_dir_of(run) == run / "files"
    assert epochs_of(checkpoints.select(run, "last")) == [49]


def test_log_dir_reads_output_root_when_log_dir_is_absent(tmp_path):
    run = make_run(tmp_path, NINE_B7C)
    (run / "run_paths.yaml").write_text(
        yaml.safe_dump({"output_root": str(tmp_path)})
    )
    assert checkpoints.log_dir_of(run) == tmp_path / "files"
