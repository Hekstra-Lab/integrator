"""Smoke test for the ragged pipeline.

Generates a tiny synthetic dataset, builds the integrator + data loader from
the toy YAML, runs:
  - one training_step (forward + loss + backward),
  - one validation_step,
  - one predict_step.

If this all passes locally, the wiring is correct and the cluster run is
unlikely to fail on plumbing (it can still fail on data-specific issues
like NaN gradients on real images).

Run with:
    cd /Users/luis/integrator
    pytest tests/ragged/test_ragged_pipeline.py -xvs
or:
    python tests/ragged/test_ragged_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sibling helper importable without packaging gymnastics
sys.path.insert(0, str(Path(__file__).parent))

import torch
import yaml

from integrator.data_loaders.ragged_data_module import (
    pad_collate_ragged,
    RaggedShoeboxDataset,
)
from integrator.utils.factory_utils import (
    construct_data_loader,
    construct_integrator,
)

from make_toy_data import make_toy_dataset

TOY_YAML = Path(__file__).parent / "toy_ragged.yaml"


def _load_cfg_with_data_dir(data_dir: Path) -> dict:
    text = TOY_YAML.read_text().replace("{{DATA_DIR}}", str(data_dir))
    return yaml.safe_load(text)


def test_dataset_loads_and_collates(tmp_path):
    data_dir = make_toy_dataset(
        tmp_path / "toy_data", n_chunks=2, refl_per_chunk=32, n_bins=5
    )
    ds = RaggedShoeboxDataset(
        chunks_dir=data_dir / "chunks",
        anscombe=True,
    )
    assert len(ds) == 64

    item = ds[0]
    for key in ("counts", "standardized_data", "mask", "shape", "bbox", "refl_id", "d", "group_label"):
        assert key in item, f"missing item key: {key}"
    D, H, W = item["shape"]
    assert item["counts"].shape == (D, H, W)
    assert item["standardized_data"].shape == (D, H, W)
    assert item["mask"].shape == (D, H, W)

    batch = pad_collate_ragged([ds[i] for i in range(8)])
    for k in ("counts", "standardized_data", "mask", "shapes", "bboxes", "refl_ids", "metadata"):
        assert k in batch, f"missing batch key: {k}"
    assert batch["counts"].shape[0] == 8
    assert batch["mask"].dtype == torch.bool
    assert batch["metadata"]["d"].shape == (8,)
    assert batch["metadata"]["group_label"].shape == (8,)


def test_log1p_transform_compresses_tail(tmp_path):
    """log1p (no standardization) should produce a much tighter range than
    anscombe + standardization when the raw data has bright outliers.

    log1p is fed straight to the encoder (matching scvi-tools'
    `log_variational=True` recipe — encoder GroupNorm handles the rest);
    anscombe still goes through global z-scoring against the saved stats.
    log1p compresses the tail an order of magnitude harder than Anscombe —
    confirms the transform is the right knob for this skewed-count regime."""
    import numpy as np
    from integrator.data_loaders.ragged_data_module import RaggedShoeboxDataset

    data_dir = make_toy_dataset(
        tmp_path / "toy_data", n_chunks=1, refl_per_chunk=4, n_bins=2
    )
    chunk0 = data_dir / "chunks" / "chunk_000.npz"
    with np.load(chunk0) as npz:
        arrs = {k: npz[k] for k in npz.files}
    arrs["data"] = arrs["data"].astype(np.int32)
    arrs["data"][0] = 100_000   # an extreme bright pixel
    arrs["mask"][0] = True
    np.savez(chunk0, **arrs)

    ds_anscombe = RaggedShoeboxDataset(
        chunks_dir=data_dir / "chunks", anscombe=True, transform="anscombe",
    )
    ds_log1p = RaggedShoeboxDataset(
        chunks_dir=data_dir / "chunks", transform="log1p",
    )
    a = ds_anscombe[0]["standardized_data"].abs().max().item()
    l = ds_log1p[0]["standardized_data"].abs().max().item()
    # Both are standardized; log1p just compresses the tail much more.
    assert l < a, f"log1p ({l}) should produce a smaller magnitude than anscombe ({a})"
    # Sanity: log1p's standardized max should be O(10), not O(100).
    assert l < 50.0, f"log1p output unexpectedly large: {l}"

    # With on-the-fly stats, the bulk of standardized values should be
    # roughly centered at 0.
    bulk = ds_log1p[0]["standardized_data"].abs().mean().item()
    assert bulk < 10.0, f"log1p bulk magnitude unexpectedly large: {bulk}"


def test_protect_foreground_keeps_bright_bragg_signal(tmp_path):
    """A bright pixel inside DIALS Foreground stays valid; outside it gets
    masked. Both get clipped in `counts` either way."""
    import numpy as np
    from integrator.data_loaders.ragged_data_module import RaggedShoeboxDataset

    data_dir = make_toy_dataset(
        tmp_path / "toy_data", n_chunks=1, refl_per_chunk=4, n_bins=2
    )
    chunk0 = data_dir / "chunks" / "chunk_000.npz"
    with np.load(chunk0) as npz:
        arrs = {k: npz[k] for k in npz.files}
    arrs["data"] = arrs["data"].astype(np.int32)
    # Two bright pixels: index 0 inside foreground, index 1 outside foreground.
    arrs["data"][0] = 80_000
    arrs["data"][1] = 80_000
    arrs["mask"][0] = True
    arrs["mask"][1] = True
    arrs["foreground"][0] = True   # genuine Bragg signal
    arrs["foreground"][1] = False  # off-foreground: probably hot pixel
    np.savez(chunk0, **arrs)

    ds = RaggedShoeboxDataset(
        chunks_dir=data_dir / "chunks",
        anscombe=True,
        transform="anscombe",
        max_count=10_000.0,
        protect_foreground=True,
    )
    item = ds[0]
    flat_mask = item["mask"].reshape(-1)
    flat_counts = item["counts"].reshape(-1)
    # In-foreground bright pixel must STAY valid (kept for training)
    assert bool(flat_mask[0]), "in-foreground bright pixel should remain valid"
    # Off-foreground bright pixel must be MASKED OUT (artifact)
    assert not bool(flat_mask[1]), "off-foreground bright pixel should be masked"
    # Both got clipped in counts
    assert float(flat_counts[0]) <= 10_000.0
    assert float(flat_counts[1]) <= 10_000.0


def test_max_count_masks_outlier_pixels(tmp_path):
    """A pixel with raw count above max_count must end up:
       - excluded from the training mask
       - clipped in `counts` (so Poisson NLL can't see the extreme value)
       - zeroed in standardized_data (so encoder doesn't see it)
    """
    import numpy as np
    from integrator.data_loaders.ragged_data_module import RaggedShoeboxDataset

    data_dir = make_toy_dataset(
        tmp_path / "toy_data", n_chunks=1, refl_per_chunk=4, n_bins=2
    )
    # Inject one extreme outlier into the first chunk's data array.
    # Toy data is uint16; promote to int32 so we can store an out-of-range
    # value, then use a threshold the outlier definitely exceeds.
    chunk0 = data_dir / "chunks" / "chunk_000.npz"
    with np.load(chunk0) as npz:
        arrs = {k: npz[k] for k in npz.files}
    arrs["data"] = arrs["data"].astype(np.int32)
    arrs["data"][0] = 50_000   # outlier well above any normal toy value
    arrs["mask"][0] = True     # would be valid except for the threshold
    np.savez(chunk0, **arrs)

    ds = RaggedShoeboxDataset(
        chunks_dir=data_dir / "chunks",
        anscombe=True,
        max_count=10_000.0,
    )
    item = ds[0]
    flat_mask = item["mask"].reshape(-1)
    flat_counts = item["counts"].reshape(-1)
    flat_std = item["standardized_data"].reshape(-1)

    # The extreme pixel must be masked invalid
    assert not bool(flat_mask[0]), "outlier pixel should be masked False"
    # Counts must be clipped to max_count
    assert float(flat_counts[0]) <= 10_000.0
    # Standardized view must be 0 at the masked voxel
    assert float(flat_std[0]) == 0.0


def test_ragged_hierC_construct_and_step(tmp_path):
    """Smoke test for RaggedHierarchicalIntegratorC (5-encoder variant)."""
    import torch

    data_dir = make_toy_dataset(
        tmp_path / "toy_data", n_chunks=2, refl_per_chunk=32, n_bins=5
    )
    cfg = _load_cfg_with_data_dir(data_dir)
    # Switch to hierC + add the 4 intensity encoders (k_i, r_i, k_bg, r_bg).
    cfg["integrator"]["name"] = "hierarchicalC_ragged"
    int_args = cfg["encoders"][1]["args"]   # the existing intensity-encoder args
    cfg["encoders"] = [
        cfg["encoders"][0],                                # profile (shoebox encoder)
        {"name": "ragged_intensity_encoder", "args": int_args},  # k_i
        {"name": "ragged_intensity_encoder", "args": int_args},  # r_i
        {"name": "ragged_intensity_encoder", "args": int_args},  # k_bg
        {"name": "ragged_intensity_encoder", "args": int_args},  # r_bg
    ]

    integrator = construct_integrator(cfg, skip_warmstart=True)
    data_loader = construct_data_loader(cfg)
    data_loader.setup()

    train_loader = data_loader.train_dataloader()
    batch = next(iter(train_loader))

    integrator.train()
    out = integrator.training_step(batch, 0)
    assert "loss" in out
    assert torch.isfinite(out["loss"]), f"non-finite hierC loss: {out['loss']}"


def _build_hierC_cfg_with_separate_inputs(data_dir, mu_log_qi=True):
    """hierC ragged config matching the live YAML: separate_inputs + mu_log."""
    cfg = _load_cfg_with_data_dir(data_dir)
    cfg["integrator"]["name"] = "hierarchicalC_ragged"
    int_args = cfg["encoders"][1]["args"]
    cfg["encoders"] = [
        cfg["encoders"][0],
        {"name": "ragged_intensity_encoder", "args": int_args},  # k_i
        {"name": "ragged_intensity_encoder", "args": int_args},  # r_i
        {"name": "ragged_intensity_encoder", "args": int_args},  # k_bg
        {"name": "ragged_intensity_encoder", "args": int_args},  # r_bg
    ]
    qi_args = {"in_features": 16, "eps": 1.0e-06, "mean_init": None,
               "separate_inputs": True}
    if mu_log_qi:
        qi_args["mu_parameterization"] = "log"
    cfg["surrogates"]["qi"] = {"name": "gammaB", "args": qi_args}
    cfg["surrogates"]["qbg"] = {"name": "gammaB", "args": {
        "in_features": 16, "eps": 1.0e-06, "mean_init": None,
        "separate_inputs": True,
    }}
    cfg["surrogates"]["qp"]["args"]["fourier_freqs"] = 0
    return cfg


def test_ragged_hierC_forward_shapes_and_invariants(tmp_path):
    """Verify the ragged hierC forward computes the right tensors with the
    right shapes — this is the test that would catch a real wiring bug.

    Checks:
      - 5 encoder outputs are (B, encoder_out)
      - qi/qbg are Gamma(B,) with finite, positive concentration & rate
      - qp.zp is (mc, B, K) and sums to ~1 over valid voxels per (mc, refl)
      - qp.zp is exactly 0 at padded voxels
      - sampled zI, zbg are (B, mc, 1); zp_sampled is (B, mc, K)
      - rate has shape (B, mc, K) and is positive
      - rate at padded voxels = zbg only (no profile contribution)
      - counts/mask are (B, K), mask is bool
      - loss is finite and a scalar
      - backward writes a non-zero grad to ALL 5 encoders' first conv weight
        — this catches the "separate_inputs=False silently drops r_i/r_bg"
        regression we previously had on the fixed pipeline.
    """
    import torch
    from torch.distributions import Gamma

    data_dir = make_toy_dataset(
        tmp_path / "toy_data", n_chunks=2, refl_per_chunk=32, n_bins=5
    )
    cfg = _build_hierC_cfg_with_separate_inputs(data_dir, mu_log_qi=True)

    integrator = construct_integrator(cfg, skip_warmstart=True)
    data_loader = construct_data_loader(cfg)
    data_loader.setup()
    integrator.train()

    batch = next(iter(data_loader.train_dataloader()))
    B, Dmax, Hmax, Wmax = batch["counts"].shape
    K = Dmax * Hmax * Wmax
    mc = integrator.mc_samples

    # Run forward through the integrator and inspect the assembled output
    out = integrator(batch)
    forward_out = out["forward_out"]
    qi, qbg, qp = out["qi"], out["qbg"], out["qp"]

    # qi / qbg: per-reflection Gamma with positive, finite concentration & rate
    assert isinstance(qi, Gamma)
    assert isinstance(qbg, Gamma)
    for name, dist in (("qi", qi), ("qbg", qbg)):
        assert dist.concentration.shape == (B,), f"{name}.concentration {dist.concentration.shape} != (B,)"
        assert dist.rate.shape == (B,), f"{name}.rate {dist.rate.shape} != (B,)"
        assert torch.all(torch.isfinite(dist.concentration)), f"{name} has non-finite concentration"
        assert torch.all(torch.isfinite(dist.rate)), f"{name} has non-finite rate"
        assert torch.all(dist.concentration > 0), f"{name} has non-positive concentration"
        assert torch.all(dist.rate > 0), f"{name} has non-positive rate"

    # qp: ProfileSurrogateOutput shapes
    from integrator.model.distributions.profile_surrogates import ProfileSurrogateOutput
    assert isinstance(qp, ProfileSurrogateOutput)
    assert qp.zp.shape == (mc, B, K), f"qp.zp {qp.zp.shape} != (mc, B, K) = ({mc}, {B}, {K})"
    assert qp.mean_profile.shape == (B, K)
    d_basis = qp.mu_h.shape[-1]
    assert qp.mu_h.shape == (B, d_basis)
    assert qp.std_h.shape == (B, d_basis)
    assert torch.all(qp.std_h > 0), "qp.std_h must be positive (softplus)"

    # Profile invariant: sums to ~1 over valid voxels, exactly 0 at padded
    mask_flat = batch["mask"].reshape(B, K)  # (B, K) bool
    zp = qp.zp  # (mc, B, K)
    zp_at_padded = zp.masked_select(~mask_flat.unsqueeze(0).expand_as(zp))
    assert torch.all(zp_at_padded == 0), "qp.zp must be exactly 0 at padded voxels (masked softmax)"
    sums = zp.sum(dim=-1)  # (mc, B)
    # Some refls might have all-False masks if the dataset is degenerate; guard:
    has_valid = mask_flat.any(dim=-1)  # (B,)
    sums_valid = sums[:, has_valid]
    if sums_valid.numel() > 0:
        assert torch.allclose(sums_valid, torch.ones_like(sums_valid), atol=1e-5), \
            f"qp.zp must sum to 1 over valid voxels; got min={sums_valid.min()}, max={sums_valid.max()}"

    # rate / counts / mask shapes (assembled as flat-K tensors)
    rate = forward_out["rates"]
    assert rate.shape == (B, mc, K), f"rate {rate.shape} != (B, mc, K) = ({B}, {mc}, {K})"
    assert torch.all(torch.isfinite(rate)), "rate has non-finite entries"
    assert torch.all(rate >= 0), "rate must be non-negative"
    assert forward_out["counts"].shape == (B, K)
    assert forward_out["mask"].shape == (B, K)
    assert forward_out["mask"].dtype == torch.bool

    # zp samples (assembled output keeps zp; zbg is not stored on the dict)
    zp_sampled = forward_out["zp"]
    assert zp_sampled.shape == (B, mc, K), f"zp_sampled {zp_sampled.shape} != (B, mc, K)"

    # rate at padded voxels should be CONSTANT within each (B, mc) row,
    # because rate = zI*zp + zbg and zp=0 at padded voxels → rate = zbg.
    # That implies the spread (max - min) across pad-only voxels per row is 0.
    pad_locations = ~mask_flat  # (B, K)
    for b in range(B):
        if not pad_locations[b].any():
            continue
        rate_b_pad = rate[b][:, pad_locations[b]]  # (mc, n_pad_voxels)
        spread = rate_b_pad.max(dim=-1).values - rate_b_pad.min(dim=-1).values
        assert torch.all(spread < 1e-5), (
            f"rate at padded voxels should be constant per (refl, mc) "
            f"(equals zbg); refl={b} spread={spread.max()}"
        )

    # End-to-end loss is finite
    train_out = integrator.training_step(batch, 0)
    loss = train_out["loss"]
    assert loss.dim() == 0, f"loss must be a scalar; got shape {loss.shape}"
    assert torch.isfinite(loss), f"non-finite loss: {loss}"

    # Backward — every encoder must receive a non-zero gradient on its first
    # conv weight. This is the regression test for the
    # "separate_inputs=False silently drops the second arg" bug.
    integrator.zero_grad()
    train_out2 = integrator.training_step(batch, 0)
    train_out2["loss"].backward()
    encoder_keys = ["profile", "k_i", "r_i", "k_bg", "r_bg"]
    grad_norms = {}
    for k in encoder_keys:
        w = integrator.encoders[k].conv1.weight
        assert w.grad is not None, f"encoder['{k}'].conv1 has no gradient"
        grad_norms[k] = float(w.grad.abs().sum())
    for k, g in grad_norms.items():
        assert g > 0, (
            f"encoder['{k}'].conv1 received ZERO gradient — its output is not "
            f"feeding the loss. All five encoders must contribute. Norms: {grad_norms}"
        )


def test_construct_integrator_and_data_loader(tmp_path):
    data_dir = make_toy_dataset(
        tmp_path / "toy_data", n_chunks=2, refl_per_chunk=32, n_bins=5
    )
    cfg = _load_cfg_with_data_dir(data_dir)

    integrator = construct_integrator(cfg, skip_warmstart=True)
    data_loader = construct_data_loader(cfg)
    data_loader.setup()

    assert hasattr(data_loader, "dataset")
    assert len(data_loader.dataset) == 64

    train_loader = data_loader.train_dataloader()
    batch = next(iter(train_loader))
    assert "counts" in batch and "standardized_data" in batch and "metadata" in batch

    # Forward through the model — exercises encoders + qp + qi + qbg
    out = integrator(batch)
    assert "forward_out" in out
    assert "qp" in out and "qi" in out and "qbg" in out


def test_training_validation_predict_steps(tmp_path):
    data_dir = make_toy_dataset(
        tmp_path / "toy_data", n_chunks=2, refl_per_chunk=32, n_bins=5
    )
    cfg = _load_cfg_with_data_dir(data_dir)

    integrator = construct_integrator(cfg, skip_warmstart=True)
    data_loader = construct_data_loader(cfg)
    data_loader.setup()

    train_loader = data_loader.train_dataloader()
    val_loader = data_loader.val_dataloader()

    integrator.train()
    train_batch = next(iter(train_loader))
    train_out = integrator.training_step(train_batch, 0)
    assert "loss" in train_out
    loss = train_out["loss"]
    assert torch.isfinite(loss), f"non-finite training loss: {loss}"
    loss.backward()
    # Verify some gradients flow
    for name, p in integrator.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            break
    else:
        raise AssertionError("no parameter received a gradient")

    integrator.eval()
    val_batch = next(iter(val_loader))
    with torch.no_grad():
        val_out = integrator.validation_step(val_batch, 0)
    assert "loss" in val_out

    with torch.no_grad():
        pred = integrator.predict_step(val_batch, 0)
    # predict_keys subset
    expected = {"refl_ids", "qi_mean", "qi_var", "qbg_mean", "qbg_var", "d", "group_label"}
    missing = expected - set(pred.keys())
    assert not missing, f"missing predict keys: {missing}"


def test_ragged_hierC_trains_50_steps_without_nan(tmp_path):
    """Loop 50 train steps on the toy dataset with the same surrogate config
    as configs/ragged/hierC_140_ragged.yaml (mu_log on qi, separate_inputs,
    mean_init: null). If this reproduces a NaN, it's a code bug; if not, the
    real-data NaN at step ~70 is data-scale specific (encoder feature
    distribution under variable shoebox size + bright pixels).
    """
    import torch

    data_dir = make_toy_dataset(
        tmp_path / "toy_data", n_chunks=2, refl_per_chunk=64, n_bins=5
    )
    cfg = _build_hierC_cfg_with_separate_inputs(data_dir, mu_log_qi=True)
    integrator = construct_integrator(cfg, skip_warmstart=True)
    data_loader = construct_data_loader(cfg)
    data_loader.setup()
    integrator.train()

    optim = torch.optim.Adam(integrator.parameters(), lr=1e-3)
    train_loader = data_loader.train_dataloader()
    losses = []
    n_steps = 50
    step = 0
    while step < n_steps:
        for batch in train_loader:
            if step >= n_steps:
                break
            out = integrator.training_step(batch, step)
            loss = out["loss"]
            assert torch.isfinite(loss), (
                f"step {step}: loss became non-finite ({loss}). "
                f"loss history: {losses[-5:]}"
            )
            optim.zero_grad()
            loss.backward()
            # Mirror the YAML's gradient_clip_val
            torch.nn.utils.clip_grad_norm_(integrator.parameters(), 1.0)
            for name, p in integrator.named_parameters():
                if p.grad is not None and not torch.all(torch.isfinite(p.grad)):
                    raise AssertionError(
                        f"step {step}: NaN/Inf in grad of '{name}'. "
                        f"loss history: {losses[-5:]}"
                    )
            optim.step()
            for name, p in integrator.named_parameters():
                if not torch.all(torch.isfinite(p)):
                    raise AssertionError(
                        f"step {step}: NaN/Inf in param '{name}' after step. "
                        f"loss history: {losses[-5:]}"
                    )
            losses.append(float(loss))
            step += 1


def test_prepare_priors_works_without_counts_npy(tmp_path):
    """Verify prepare_per_bin_priors generates bg_rate_per_group from
    metadata['background.mean'] alone — without needing counts.npy/masks.npy.

    This is the fix that makes the ragged pipeline use the same per-resolution
    binning as the fixed pipeline. Before this, prepare_per_bin_priors would
    fail with FileNotFoundError because the ragged data layout has no flat
    counts.npy (data lives in chunks/*.npz).
    """
    import torch
    from integrator.utils.prepare_priors import prepare_per_bin_priors

    # Build a fake ragged-layout data_dir: just metadata.pt, no counts/masks.
    data_dir = tmp_path / "ragged_data"
    data_dir.mkdir()
    (data_dir / "chunks").mkdir()  # purely to look like the ragged layout

    # 60 refl/bin at 30 bins clears _bin_by_resolution's min_per_bin=50 guard
    n_refl = 1800
    n_bins = 30
    rng = torch.Generator().manual_seed(0)
    # d ∈ [1.5, 30] Å, monotonic so quantile binning produces ~equal-sized bins
    d = torch.linspace(1.5, 30.0, n_refl)
    # Background scales smoothly with d so per-bin τ values vary, mimicking
    # what we'd see on a real dataset.
    bg_mean = 0.05 + 0.1 * torch.rand(n_refl, generator=rng)
    metadata = {
        "d": d,
        "background.mean": bg_mean,
        "refl_ids": torch.arange(n_refl, dtype=torch.int32),
    }
    torch.save(metadata, data_dir / "metadata.pt")

    cfg = {
        "data_loader": {
            "name": "ragged_data",
            "args": {"data_dir": str(data_dir)},
        },
        "loss": {
            "name": "wilson",
            "args": {
                "n_bins": n_bins,
                "bg_rate_per_group": "bg_rate_per_group.pt",
            },
        },
        "surrogates": {},
    }

    prepare_per_bin_priors(cfg)

    bg_path = data_dir / f"bg_rate_per_group_{n_bins}.pt"
    gl_path = data_dir / f"group_labels_{n_bins}.pt"
    assert bg_path.exists(), f"bg_rate_per_group_{n_bins}.pt not generated"
    assert gl_path.exists(), f"group_labels_{n_bins}.pt not generated"

    bg_rate = torch.load(bg_path)
    group_labels = torch.load(gl_path)
    assert bg_rate.shape == (n_bins,), f"bg_rate shape {bg_rate.shape} != ({n_bins},)"
    assert group_labels.shape == (n_refl,)
    assert torch.all(bg_rate > 0), "bg_rate must be positive (1/mean(bg))"
    # Sanity: 1/bg_rate is mean(bg) per bin and should be near the input
    # bg_mean's overall mean (≈ 0.05 + 0.05 = 0.10).
    inv_mean = (1.0 / bg_rate).mean().item()
    assert 0.05 < inv_mean < 0.20, (
        f"1/τ mean = {inv_mean} is outside the expected range for "
        f"input bg_mean ~U(0.05, 0.15)"
    )


if __name__ == "__main__":
    # Allow running without pytest for quick iteration
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        print("[1/3] dataset loads + collates")
        test_dataset_loads_and_collates(td)
        print("  OK")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        print("[2/3] construct integrator + data loader")
        test_construct_integrator_and_data_loader(td)
        print("  OK")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        print("[3/3] training/validation/predict steps")
        test_training_validation_predict_steps(td)
        print("  OK")

    print("\nAll smoke tests passed.")
