"""Reproduce the step-65 NaN locally with full instrumentation.

Loads the same YAML the cluster uses, builds the integrator + ragged data
loader, and steps through training with:
  - torch.autograd.set_detect_anomaly(True): traceback identifies the
    backward op that first produced NaN/Inf.
  - Forward NaN hooks on every nn.Module in encoders + surrogates: prints
    the FIRST module name whose output contains NaN, with batch index and
    output stats.
  - Per-step tensor sanity logs (encoder output ranges, qi/qbg parameters)
    so you can see when the input regime starts going off.

Run on the cluster:
    uv run python scripts/debug_ragged_train.py \
        --config configs/ragged/hierB_140_ragged.yaml \
        --max-steps 80 \
        --first-step 0
"""

import argparse
from pathlib import Path

import torch
import yaml


def install_nan_hooks(integrator):
    """Register forward hooks that flag the FIRST NaN/Inf in any submodule
    output. The hook prints the offending module name and aborts with a
    RuntimeError so the trainer's main loop stops cleanly."""

    def _scan_for_nan(name, output):
        # Output may be a tensor, tuple of tensors, or a distribution-like obj.
        tensors = []
        if isinstance(output, torch.Tensor):
            tensors.append(("out", output))
        elif isinstance(output, (tuple, list)):
            for i, t in enumerate(output):
                if isinstance(t, torch.Tensor):
                    tensors.append((f"out[{i}]", t))
        else:
            for attr in ("zp", "mean_profile", "mu_h", "std_h",
                         "concentration", "rate", "loc", "scale"):
                t = getattr(output, attr, None)
                if isinstance(t, torch.Tensor):
                    tensors.append((attr, t))
        for label, t in tensors:
            if not torch.isfinite(t).all():
                n_nan = int(torch.isnan(t).sum())
                n_inf = int(torch.isinf(t).sum())
                t_min = float(t[torch.isfinite(t)].min()) if torch.isfinite(t).any() else float("nan")
                t_max = float(t[torch.isfinite(t)].max()) if torch.isfinite(t).any() else float("nan")
                raise RuntimeError(
                    f"[NaN-hook] FIRST non-finite output detected in:\n"
                    f"  module: {name}\n"
                    f"  output {label}: shape={tuple(t.shape)}, dtype={t.dtype}\n"
                    f"  n_nan={n_nan}, n_inf={n_inf}\n"
                    f"  finite_min={t_min:.4g}, finite_max={t_max:.4g}"
                )

    for name, module in integrator.named_modules():
        # Skip top-level container; only hook leaves + small composites.
        if name == "":
            continue
        module.register_forward_hook(
            lambda mod, _inp, out, _name=name: _scan_for_nan(_name, out)
        )


def log_batch_stats(step, batch, integrator):
    """Cheap per-step diagnostic to see when input regime drifts."""
    counts = batch["counts"]
    std = batch["standardized_data"]
    mask = batch["mask"]
    print(
        f"  step {step:>4d}  "
        f"counts: max={counts.max().item():.1f} median={counts.median().item():.1f}  "
        f"std: min={std.min().item():.2f} max={std.max().item():.2f}  "
        f"mask_frac={mask.float().mean().item():.3f}"
    )

    # Run encoders + surrogates without grad to inspect Gamma parameters.
    # This is a SECOND forward, so it doesn't perturb training, but it
    # mirrors what the actual training step about to run will produce.
    with torch.no_grad():
        x = batch["standardized_data"].unsqueeze(1)
        m = batch["mask"]
        x_profile = integrator.encoders["profile"](x, m)
        x_k = integrator.encoders["k"](x, m)
        x_r = integrator.encoders["r"](x, m)

        for name, t in [("x_prof", x_profile), ("x_k", x_k), ("x_r", x_r)]:
            print(
                f"    {name:>6s}: min={t.min().item():.3f} max={t.max().item():.3f} "
                f"mean={t.mean().item():.3f} std={t.std().item():.3f}"
            )

        # Build qi and qbg distributions and print their parameters.
        # gammaB exposes raw_mu/raw_fano internally; we surface the final
        # `concentration` (= k) and `rate` (= r) which are what _standard_gamma
        # consumes.
        for name, surrogate_key in [("qi", "qi"), ("qbg", "qbg")]:
            surr = integrator.surrogates[surrogate_key]
            dist = surr(x_k, x_r)
            k = dist.concentration
            r = dist.rate
            print(
                f"    {name}.k:   min={k.min().item():.3e} max={k.max().item():.3e} "
                f"mean={k.mean().item():.3e}  "
                f"n_above_500={int((k > 500).sum())}  "
                f"n_above_950={int((k > 950).sum())}  "
                f"n_below_0.1={int((k < 0.1).sum())}"
            )
            print(
                f"    {name}.r:   min={r.min().item():.3e} max={r.max().item():.3e} "
                f"mean={r.mean().item():.3e}"
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--anomaly", action="store_true", default=True,
                    help="Enable torch.autograd.set_detect_anomaly(True)")
    ap.add_argument("--no-anomaly", dest="anomaly", action="store_false")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    if args.anomaly:
        torch.autograd.set_detect_anomaly(True)
        print("torch.autograd anomaly detection: ON")

    from integrator.utils.factory_utils import (
        construct_integrator, construct_data_loader,
    )
    integrator = construct_integrator(cfg, skip_warmstart=True).cuda()
    data_loader = construct_data_loader(cfg)
    data_loader.setup()

    install_nan_hooks(integrator)

    # Optimizer (Adam with the lr in the YAML)
    lr = float(cfg["integrator"]["args"].get("lr", 1e-3))
    opt = torch.optim.Adam(integrator.parameters(), lr=lr)

    train_loader = data_loader.train_dataloader()
    integrator.train()
    print(f"Stepping through up to {args.max_steps} batches...")

    for step, batch in enumerate(train_loader):
        if step >= args.max_steps:
            break
        # Move batch to GPU
        batch = {
            k: (v.cuda() if isinstance(v, torch.Tensor) else
                {kk: vv.cuda() if isinstance(vv, torch.Tensor) else vv
                 for kk, vv in v.items()} if isinstance(v, dict) else v)
            for k, v in batch.items()
        }

        if step % args.log_every == 0 or step >= 60:
            log_batch_stats(step, batch, integrator)

        opt.zero_grad()
        out = integrator._step(batch, step="train")
        loss = out["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss became non-finite at step {step}: {loss}")
        loss.backward()
        # Match YAML gradient_clip_val if set
        clip = cfg.get("trainer", {}).get("gradient_clip_val")
        if clip:
            torch.nn.utils.clip_grad_norm_(integrator.parameters(), float(clip))
        opt.step()

        if step >= 60 and step % 1 == 0:
            print(f"  step {step}: loss={loss.item():.4f}")

    print("Reached max_steps without a NaN.")


if __name__ == "__main__":
    main()
