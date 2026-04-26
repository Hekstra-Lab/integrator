"""Test the fixed-pipeline ShoeboxDataModule's log1p transform path.

Mirrors the ragged-pipeline log1p semantics: log1p applied to raw counts,
NO global z-score, masked voxels zeroed. This is the scvi-tools
`log_variational=True` recipe.
"""

from __future__ import annotations

from pathlib import Path

import torch

from integrator.data_loaders.data_module import ShoeboxDataModule


def _write_toy_dataset(tmp_path: Path, n: int = 64, V: int = 24 * 24 * 3):
    """Synthesize a minimal fixed-pipeline data dir.

    Layout matches what `_load_shoebox_array` expects: counts.pt, masks.pt,
    stats.pt, reference.pt at the data_dir root, with shapes (N, V).
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    rng = torch.Generator().manual_seed(0)
    # Mix of low (0–10), mid (100–1000), and bright (50k+) pixels — exactly
    # the distribution that breaks Anscombe + global-z-score.
    counts = torch.zeros(n, V, dtype=torch.int32)
    counts[:, :V // 2] = torch.randint(0, 10, (n, V // 2), generator=rng, dtype=torch.int32)
    counts[:, V // 2:V * 3 // 4] = torch.randint(100, 1000, (n, V * 3 // 4 - V // 2), generator=rng, dtype=torch.int32)
    counts[:, -1] = 50_000  # one bright pixel per shoebox

    masks = torch.ones(n, V, dtype=torch.bool)
    # Stats are required by .setup() but the log1p path ignores them.
    raw_mean = counts.float().mean()
    raw_var = counts.float().var()
    stats = torch.tensor([raw_mean, raw_var], dtype=torch.float32)

    reference = {
        "d": torch.linspace(1.5, 30.0, n),
        "is_test": torch.zeros(n, dtype=torch.bool),
        "intensity.prf.variance": torch.ones(n),
        "refl_ids": torch.arange(n, dtype=torch.int32),
    }

    torch.save(counts, data_dir / "counts.pt")
    torch.save(masks, data_dir / "masks.pt")
    torch.save(stats, data_dir / "stats.pt")
    torch.save(reference, data_dir / "reference.pt")

    return data_dir, counts, masks


def _make_dm(data_dir: Path, **kwargs) -> ShoeboxDataModule:
    return ShoeboxDataModule(
        data_dir=str(data_dir),
        batch_size=8,
        val_split=0.2,
        test_split=0.0,
        num_workers=0,
        include_test=True,
        subset_size=None,
        cutoff=None,
        shoebox_file_names={
            "counts": "counts.pt",
            "masks": "masks.pt",
            "stats": "stats.pt",
            "reference": "reference.pt",
            "standardized_counts": None,
        },
        H=24,
        W=24,
        D=3,
        **kwargs,
    )


def test_log1p_transform_applies_log1p_without_zscoring(tmp_path):
    """log1p path: standardized_counts == log1p(counts) * masks; NO z-score."""
    data_dir, counts, masks = _write_toy_dataset(tmp_path)
    dm = _make_dm(data_dir, transform="log1p")
    dm.setup()

    # standardized_counts is stored on the IntegratorDataset
    std = dm.full_dataset.standardized_counts
    expected = torch.log1p(counts.clamp(min=0).float()) * masks.float()
    assert torch.allclose(std, expected, atol=1e-6), (
        f"log1p path should produce log1p(counts)*mask without z-scoring; "
        f"max abs diff = {(std - expected).abs().max()}"
    )

    # Sanity: bright pixel raw=50k → log1p ≈ 10.82; bulk should be small.
    assert std.max().item() < 12.0, f"log1p max too large: {std.max()}"
    assert std.mean().item() < 5.0, f"log1p mean unexpectedly large: {std.mean()}"


def test_anscombe_path_still_z_scores(tmp_path):
    """Regression: explicit transform='anscombe' preserves the legacy
    anscombe + global z-score behavior. Same effect as anscombe=True."""
    data_dir, counts, masks = _write_toy_dataset(tmp_path)

    dm_explicit = _make_dm(data_dir, transform="anscombe")
    dm_explicit.setup()

    dm_legacy = _make_dm(data_dir, anscombe=True)
    dm_legacy.setup()

    std_e = dm_explicit.full_dataset.standardized_counts
    std_l = dm_legacy.full_dataset.standardized_counts
    assert torch.allclose(std_e, std_l), (
        "transform='anscombe' must match anscombe=True legacy path"
    )

    # Sanity: anscombe path differs from log1p path (different transform,
    # different normalization). The two should NOT be the same tensor.
    dm_log = _make_dm(data_dir, transform="log1p")
    dm_log.setup()
    std_log = dm_log.full_dataset.standardized_counts
    assert not torch.allclose(std_e, std_log), (
        "anscombe and log1p paths should produce different outputs"
    )


def test_default_anscombe_false_preserves_legacy_behavior(tmp_path):
    """Without `transform`, legacy `anscombe=False` → 'none' path
    ((counts*mask - mean) / std). Confirms backward compat."""
    data_dir, counts, masks = _write_toy_dataset(tmp_path)
    dm = _make_dm(data_dir)  # transform=None, anscombe=False
    dm.setup()
    assert dm.transform == "none"


def test_invalid_transform_raises(tmp_path):
    data_dir, _, _ = _write_toy_dataset(tmp_path)
    try:
        _make_dm(data_dir, transform="zscore_per_pixel")
    except ValueError as e:
        assert "transform must be" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown transform")


if __name__ == "__main__":
    import tempfile

    for name, fn in [
        ("log1p applies log1p without z-score", test_log1p_transform_applies_log1p_without_zscoring),
        ("anscombe explicit == anscombe=True", test_anscombe_path_still_z_scores),
        ("default preserves legacy", test_default_anscombe_false_preserves_legacy_behavior),
        ("invalid transform raises", test_invalid_transform_raises),
    ]:
        with tempfile.TemporaryDirectory() as td:
            fn(Path(td))
        print(f"PASSED: {name}")
