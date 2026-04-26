"""Step-by-step comparison of gammaA vs gammaB on real data.

Hypothesis under test: gammaA trains stably while gammaB NaN's at step ~50
on dataset 140 even with floor_k_min, mean_init zero-init, and lr warmup.
Goal: identify which step-level quantity diverges between the two so we
can pin the failure mode.

Strategy: train each variant on the SAME minibatch with the SAME seed for
N steps, logging per-step:
  - Surrogate forward state: k, r, mu (if available), fano (if available)
    with min / median / max / frac<thresh
  - Sampled zI, zbg (rsample output) statistics
  - Loss components (nll, kl_i, kl_bg, kl_prf, kl_hyper)
  - Gradient norms / max-abs / non-finite fraction per surrogate head
  - Encoder feature norms per encoder
  - Adam first/second moment estimates on the surrogate head weights

Both variants take the SAME YAML config; we only swap `surrogates.qi.name`
and `surrogates.qbg.name` and drop the incompatible kwargs (e.g.
`mu_parameterization` and `floor_k_min` are gammaB-only). Everything
else — encoders, data loader, loss, lr, warmup — is identical.

Usage on the cluster (A100):
    uv run python scripts/gammaA_vs_gammaB_trace.py \\
        --config configs/ragged/hierC_140_ragged.yaml \\
        --n-steps 200 \\
        --device cuda \\
        --seed 0 \\
        --out-dir gamma_trace/

Reading the output:
    Two CSVs are produced (gammaA.csv, gammaB.csv), one row per training
    step, ~50 columns each. Side-by-side compare:
      df_a = pd.read_csv("gamma_trace/gammaA.csv")
      df_b = pd.read_csv("gamma_trace/gammaB.csv")
      df_b[["step","qbg_k_min","qbg_k_max","qbg_linear_mu_grad_max",
            "kl_i","nll","loss"]].head(60)
    Look for: the first step where gammaB's qbg_k_min drops well below
    gammaA's, and which gradient/state metric started diverging in the
    steps before.
"""

import argparse
import copy
import csv
from pathlib import Path
from typing import Any

import torch
import yaml

from integrator.utils.factory_utils import (
    construct_data_loader,
    construct_integrator,
)


# ---------- config plumbing ----------


GAMMAB_ONLY_KEYS = {"mean_init", "fano_init", "mu_parameterization", "floor_k_min"}


def _override_surrogate(cfg: dict, sur_key: str, target_name: str) -> None:
    """Swap surrogates[sur_key].name to target_name and drop kwargs that
    don't apply to the target."""
    sur = cfg["surrogates"][sur_key]
    sur["name"] = target_name
    args = sur.setdefault("args", {})
    if target_name == "gammaA":
        for k in list(args.keys()):
            if k in GAMMAB_ONLY_KEYS:
                args.pop(k)


def _build_variant_cfg(base_cfg: dict, gamma_name: str) -> dict:
    """Deep-copy base_cfg and rewrite qi/qbg to use `gamma_name`."""
    cfg = copy.deepcopy(base_cfg)
    _override_surrogate(cfg, "qi", gamma_name)
    _override_surrogate(cfg, "qbg", gamma_name)
    return cfg


# ---------- per-tensor descriptive stats ----------


def _stats(t: torch.Tensor, threshold: float = 1e-2) -> dict[str, float]:
    """min / median / max / frac<threshold of a tensor (flattened)."""
    flat = t.detach().reshape(-1).float()
    if flat.numel() == 0:
        return {"min": float("nan"), "median": float("nan"),
                "max": float("nan"), "frac_lt_thresh": float("nan")}
    return {
        "min": float(flat.min()),
        "median": float(flat.median()),
        "max": float(flat.max()),
        "frac_lt_thresh": float((flat < threshold).float().mean()),
    }


def _grad_stats(module: torch.nn.Module) -> dict[str, float]:
    """grad.norm, grad.abs().max(), frac_nonfinite over module's params."""
    norms_sq = 0.0
    max_abs = 0.0
    n_total = 0
    n_finite = 0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        n_total += g.numel()
        n_finite += int(torch.isfinite(g).sum())
        g_finite = g[torch.isfinite(g)] if g.numel() else g
        if g_finite.numel():
            norms_sq += float(g_finite.float().pow(2).sum())
            max_abs = max(max_abs, float(g_finite.abs().max()))
    if n_total == 0:
        return {"grad_norm": 0.0, "grad_max": 0.0, "frac_finite": 1.0}
    return {
        "grad_norm": norms_sq ** 0.5,
        "grad_max": max_abs,
        "frac_finite": n_finite / max(n_total, 1),
    }


def _param_norm(module: torch.nn.Module) -> float:
    sq = 0.0
    for p in module.parameters():
        sq += float(p.detach().float().pow(2).sum())
    return sq ** 0.5


# ---------- surrogate-head accessors ----------


def _surrogate_heads(sur) -> dict[str, torch.nn.Module]:
    """Return {head_name: nn.Module} dict for the two heads of a Gamma surrogate.

    Maps the parameterization-specific names to a uniform schema:
      head_a = the concentration-related head (k for gammaA, mu for gammaB)
      head_b = the rate-related head (r for gammaA, fano for gammaB)
    """
    out: dict[str, torch.nn.Module] = {}
    if hasattr(sur, "linear_k"):
        out["head_a"] = sur.linear_k       # gammaA, separate_inputs
        out["head_b"] = sur.linear_r
    elif hasattr(sur, "linear_mu"):
        out["head_a"] = sur.linear_mu      # gammaB, separate_inputs
        out["head_b"] = sur.linear_fano
    elif hasattr(sur, "fc"):
        out["fc"] = sur.fc                  # combined head (either repam)
    return out


def _adam_moment_max(opt: torch.optim.Optimizer, param: torch.nn.Parameter,
                     moment: str) -> float:
    """Pull max |Adam moment estimate| for a given param. moment ∈ {'exp_avg','exp_avg_sq'}."""
    state = opt.state.get(param, None)
    if state is None or moment not in state:
        return float("nan")
    return float(state[moment].abs().max())


# ---------- core trace loop ----------


def _move_batch(batch, device):
    if isinstance(batch, dict):
        out = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        if isinstance(out.get("metadata"), dict):
            out["metadata"] = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in out["metadata"].items()
            }
        return out
    return tuple(v.to(device) if torch.is_tensor(v) else v for v in batch)


def trace_one_variant(
    cfg: dict,
    variant: str,
    n_steps: int,
    device: str,
    seed: int,
    out_path: Path,
    mode: str = "fixed",
) -> None:
    """Train `variant` (gammaA or gammaB) for n_steps, logging per-step state.

    mode:
      "fixed"  — same minibatch every step (isolates optimization dynamics
                 from data-distribution variance).
      "stream" — fresh minibatch every step (cycles through the train
                 dataloader, reusing it when exhausted). Reproduces real
                 training conditions; needed if the failure is batch-
                 dependent.
    """
    print(f"\n=== {variant} (mode={mode}) ===")
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    integrator = construct_integrator(cfg, skip_warmstart=True)
    integrator.to(device)
    integrator.train()

    dl = construct_data_loader(cfg)
    dl.setup()

    train_iter = iter(dl.train_dataloader())
    # In "fixed" mode we hold the first batch; in "stream" mode we draw
    # a fresh one each step inside the loop.
    fixed_batch = _move_batch(next(train_iter), device) if mode == "fixed" else None

    optimizer = integrator.configure_optimizers()
    if isinstance(optimizer, dict):
        optimizer = optimizer["optimizer"]

    # Identify a small set of "tracked" surrogate heads up front so we can
    # log a consistent column set across variants.
    qi_heads = _surrogate_heads(integrator.surrogates["qi"])
    qbg_heads = _surrogate_heads(integrator.surrogates["qbg"])

    rows: list[dict[str, Any]] = []
    nan_step: int | None = None

    batch_idx_in_epoch = 0
    epoch_idx = 0

    for step in range(n_steps):
        row: dict[str, Any] = {"step": step, "variant": variant}

        # Pick the batch to feed to this step.
        if mode == "fixed":
            current_batch = fixed_batch
        else:  # mode == "stream"
            try:
                current_batch = _move_batch(next(train_iter), device)
            except StopIteration:
                epoch_idx += 1
                batch_idx_in_epoch = 0
                train_iter = iter(dl.train_dataloader())
                current_batch = _move_batch(next(train_iter), device)
            batch_idx_in_epoch += 1
            row["batch_idx"] = batch_idx_in_epoch - 1
            row["epoch_idx"] = epoch_idx
            # Useful per-batch data signature for tying NaN to a specific batch.
            if isinstance(current_batch, dict):
                refl_ids = current_batch.get("refl_ids")
                if refl_ids is not None:
                    row["refl_id_first"] = int(refl_ids[0])
                    row["refl_id_last"] = int(refl_ids[-1])
                counts = current_batch.get("counts")
                if counts is not None:
                    cf = counts.detach().float()
                    row["batch_count_max"] = float(cf.max())
                    row["batch_count_mean"] = float(cf.mean())

        # ── forward + loss ─────────────────────────────────────────────
        out = integrator.training_step(current_batch, step)
        loss = out["loss"]
        components = out.get("loss_components", {})
        forward_out = out.get("forward_out", {})

        # Detect non-finite loss BEFORE backward to capture the failing step.
        loss_finite = bool(torch.isfinite(loss).item())
        if not loss_finite and nan_step is None:
            nan_step = step

        # Surrogate forward state (k, r). Both repams expose qi_params/qbg_params
        # via _assemble_outputs.
        for sur in ("qi", "qbg"):
            params = forward_out.get(f"{sur}_params", {})
            for pname in ("concentration", "rate"):
                p = params.get(pname)
                if p is not None:
                    s = _stats(p)
                    label = "k" if pname == "concentration" else "r"
                    row[f"{sur}_{label}_min"] = s["min"]
                    row[f"{sur}_{label}_median"] = s["median"]
                    row[f"{sur}_{label}_max"] = s["max"]
                    row[f"{sur}_{label}_frac_lt_1e-2"] = s["frac_lt_thresh"]

        # Sampled (zI, zbg) — taken from forward_out.zp / zbg if exposed
        zbg = forward_out.get("zbg")
        if zbg is not None:
            s = _stats(zbg)
            row["zbg_min"] = s["min"]
            row["zbg_median"] = s["median"]
            row["zbg_max"] = s["max"]

        # Loss components
        for key in ("loss", "nll", "kl", "kl_i", "kl_bg", "kl_prf", "kl_hyper"):
            if key in components:
                row[key] = float(components[key])
        row["loss_total"] = float(loss.detach()) if loss_finite else float("nan")
        row["loss_finite"] = int(loss_finite)

        # ── backward ───────────────────────────────────────────────────
        optimizer.zero_grad()
        if loss_finite:
            loss.backward()
        else:
            # Skip backward to avoid contaminating optimizer state with NaN
            # grads. The trace continues so we can see the post-NaN state.
            pass

        # Per-head gradient stats
        for sur_name, heads in (("qi", qi_heads), ("qbg", qbg_heads)):
            for head_name, head_module in heads.items():
                gs = _grad_stats(head_module)
                row[f"{sur_name}_{head_name}_grad_norm"] = gs["grad_norm"]
                row[f"{sur_name}_{head_name}_grad_max"] = gs["grad_max"]
                row[f"{sur_name}_{head_name}_grad_frac_finite"] = gs["frac_finite"]
                row[f"{sur_name}_{head_name}_param_norm"] = _param_norm(head_module)

        # Encoder feature norms — re-run encoders on the same batch we just
        # used so the norms reflect the actual forward state for this step.
        with torch.no_grad():
            if isinstance(current_batch, dict):
                shoebox_3d = current_batch.get(
                    "standardized_data", current_batch["counts"]
                ).float()
                mask_3d = current_batch["mask"]
                x = shoebox_3d.unsqueeze(1)
                for enc_name, enc in integrator.encoders.items():
                    feat = enc(x, mask_3d)
                    row[f"encoder_{enc_name}_norm"] = float(
                        feat.norm(dim=-1).mean()
                    )

        # Adam first/second moment max-abs on each surrogate head's weight.
        # Only meaningful after step 1 (Adam state is empty at step 0).
        if step >= 1:
            for sur_name, heads in (("qi", qi_heads), ("qbg", qbg_heads)):
                for head_name, head_module in heads.items():
                    for pname, param in head_module.named_parameters():
                        for moment in ("exp_avg", "exp_avg_sq"):
                            row[
                                f"{sur_name}_{head_name}_{pname}_{moment}_max"
                            ] = _adam_moment_max(optimizer, param, moment)

        # ── optimizer step (only if loss was finite) ───────────────────
        if loss_finite:
            optimizer.step()

        if step % 10 == 0 or step == n_steps - 1 or not loss_finite:
            qbg_kmin = row.get("qbg_k_min", float("nan"))
            qi_kmin = row.get("qi_k_min", float("nan"))
            print(
                f"  step={step:4d}  loss={row.get('loss_total', float('nan')):>14.2f}  "
                f"qbg_k_min={qbg_kmin:.4f}  qi_k_min={qi_kmin:.4f}  "
                f"loss_finite={loss_finite}"
            )
        rows.append(row)

        if not loss_finite:
            print(f"  ⚠ loss became non-finite at step {step}; continuing trace")
            # Continue tracing — even after NaN we want to see the state
            # propagate. But after a few more steps, bail to save time.
            assert nan_step is not None  # set on the loss_finite branch above
            if step - nan_step > 5:
                print(f"  Stopping {variant} trace (NaN persisted 5+ steps)")
                break

    # Write CSV
    all_keys: set[str] = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = ["step", "variant"] + sorted(
        k for k in all_keys if k not in ("step", "variant")
    )
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"  Wrote {out_path} ({len(rows)} rows, {len(fieldnames)} cols)")
    if nan_step is not None:
        print(f"  *** {variant} NaN'd at step {nan_step} ***")
    else:
        print(f"  {variant} survived all {n_steps} steps")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to YAML config")
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="gamma_trace", help="output directory")
    ap.add_argument(
        "--variants",
        default="gammaA,gammaB",
        help="comma-separated list of surrogate names to compare",
    )
    ap.add_argument(
        "--mode",
        default="fixed",
        choices=("fixed", "stream"),
        help="fixed = same batch every step (isolates dynamics); "
        "stream = fresh batch every step (reproduces real training).",
    )
    args = ap.parse_args()

    base_cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant in args.variants.split(","):
        variant = variant.strip()
        cfg = _build_variant_cfg(base_cfg, variant)
        out_path = out_dir / f"{variant}.csv"
        trace_one_variant(
            cfg=cfg,
            variant=variant,
            n_steps=args.n_steps,
            device=args.device,
            seed=args.seed,
            out_path=out_path,
            mode=args.mode,
        )

    print(
        f"\nDone. Compare with:\n"
        f"  python -c \"import pandas as pd; "
        f"a = pd.read_csv('{out_dir}/gammaA.csv'); "
        f"b = pd.read_csv('{out_dir}/gammaB.csv'); "
        f"print(b[['step','qbg_k_min','qbg_k_max','qbg_head_a_grad_max','kl_i','nll','loss_total']].head(60))\""
    )


if __name__ == "__main__":
    main()
