"""Sweep N random seeds and log gammaB-relevant initial-step statistics.

Hypothesis being tested: gammaB's NaN-or-not behavior on bright-tail
data is init-dependent. With `mean_init: null`, the linear_mu / linear_fano
biases fall through to PyTorch's default Kaiming-uniform random init, so
each seed lands in a different (mu, fano) starting region. We want to see
whether the seeds that NaN'd in production correspond to outlier
distributions of initial k = mu/fano, initial NLL, initial KL, or initial
gradient magnitudes — vs. seeds that train successfully.

What we log per seed (one minibatch, eval-mode forward + grad-attached loss):
  - qi/qbg concentration (k), rate (r): min / median / max / fraction below
    a "danger" threshold (1e-2). gammaB NaN's via _standard_gamma_grad when
    k < ~1e-2.
  - NLL per reflection: min / median / max
  - KL components (kl_i, kl_bg, kl_prf, kl_hyper)
  - Total ELBO
  - Encoder feature L2 norms per encoder
  - After loss.backward(): max |grad| in the qbg.linear_mu / qbg.linear_fano
    / qi.linear_mu / qi.linear_fano weights — the heads where Adam updates
    can push raw_mu / raw_fano into the unstable region

Usage:
    uv run python scripts/gamma_init_diagnostic.py \\
        --config configs/ragged/hierC_140_ragged.yaml \\
        --n-seeds 20 \\
        --device cuda \\
        --out gamma_init_diagnostic.csv

Reading the output:
    The CSV has one row per seed. Sort by `kl_i_mean + kl_bg_mean + nll_mean`
    or by `qbg_k_min` to see which seeds start closest to the NaN boundary.
    A seed where `qbg_k_min < 1e-2` AND `qbg_linear_mu_grad_max > 100` is
    a likely candidate for "this seed will NaN within ~70 steps under Adam
    at lr=1e-3".
"""

import argparse
import csv
from pathlib import Path

import torch
import yaml

from integrator.utils.factory_utils import (
    construct_data_loader,
    construct_integrator,
)


def _load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _grad_max_abs(module: torch.nn.Module) -> float:
    """Max |grad| across all parameters of `module` after .backward()."""
    out = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        out = max(out, float(p.grad.abs().max()))
    return out


def _stats(t: torch.Tensor) -> dict[str, float]:
    """min/median/max + n_below_1e-2 fraction. Tensor flattened."""
    flat = t.detach().reshape(-1).float()
    return {
        "min": float(flat.min()),
        "median": float(flat.median()),
        "max": float(flat.max()),
        "frac_below_1e-2": float((flat < 1e-2).float().mean()),
    }


def run_one_seed(cfg: dict, seed: int, device: str) -> dict:
    """Build model fresh under `seed`, run one minibatch, return stats."""
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    integrator = construct_integrator(cfg, skip_warmstart=True)
    integrator.to(device)
    integrator.train()

    dl = construct_data_loader(cfg)
    dl.setup()
    batch = next(iter(dl.train_dataloader()))

    # Move batch to device. Ragged batches are dicts; fixed batches are tuples.
    if isinstance(batch, dict):
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        if isinstance(batch.get("metadata"), dict):
            batch["metadata"] = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch["metadata"].items()
            }
    else:
        batch = tuple(
            v.to(device) if torch.is_tensor(v) else v for v in batch
        )

    out = integrator.training_step(batch, 0)
    loss = out["loss"]
    components = out.get("loss_components", {})

    # Pull qi / qbg parameters from the most recent forward — they were
    # constructed inside _step. We rebuild the dist params manually so we
    # don't rely on integrator caching.
    forward_out = out.get("forward_out", {})
    qi_k = forward_out.get("qi_params", {}).get("concentration")
    qi_r = forward_out.get("qi_params", {}).get("rate")
    qbg_k = forward_out.get("qbg_params", {}).get("concentration")
    qbg_r = forward_out.get("qbg_params", {}).get("rate")

    row: dict[str, float | int | str] = {"seed": seed}

    if qi_k is not None:
        for k, v in _stats(qi_k).items():
            row[f"qi_k_{k}"] = v
    if qi_r is not None:
        for k, v in _stats(qi_r).items():
            row[f"qi_r_{k}"] = v
    if qbg_k is not None:
        for k, v in _stats(qbg_k).items():
            row[f"qbg_k_{k}"] = v
    if qbg_r is not None:
        for k, v in _stats(qbg_r).items():
            row[f"qbg_r_{k}"] = v

    # ELBO components
    for name in ("loss", "nll", "kl", "kl_prf", "kl_i", "kl_bg", "kl_hyper"):
        if name in components:
            row[name] = float(components[name])
    row["loss_total"] = float(loss.detach())

    # Backward to get grad magnitudes on the surrogate heads
    integrator.zero_grad()
    loss.backward()
    surrogates = integrator.surrogates
    for sur_name in ("qi", "qbg"):
        sur = surrogates[sur_name]
        if hasattr(sur, "linear_mu"):
            row[f"{sur_name}_linear_mu_grad_max"] = _grad_max_abs(sur.linear_mu)
        if hasattr(sur, "linear_fano"):
            row[f"{sur_name}_linear_fano_grad_max"] = _grad_max_abs(sur.linear_fano)
        if hasattr(sur, "fc"):
            row[f"{sur_name}_fc_grad_max"] = _grad_max_abs(sur.fc)

    # Encoder feature norms — captured via a fresh forward in eval mode so
    # grads don't double-accumulate.
    integrator.zero_grad()
    integrator.eval()
    with torch.no_grad():
        if isinstance(batch, dict):
            forward_dict = integrator(batch)
        else:
            counts, shoebox, mask, meta = batch
            forward_dict = integrator(counts, shoebox, mask, meta)
        # Re-run encoders to capture feature norms — cheap
        if isinstance(batch, dict):
            shoebox_3d = batch.get("standardized_data", batch["counts"]).float()
            mask_3d = batch["mask"]
            x = shoebox_3d.unsqueeze(1)
            for enc_name, enc in integrator.encoders.items():
                feat = enc(x, mask_3d)
                row[f"encoder_{enc_name}_norm"] = float(feat.norm(dim=-1).mean())
    integrator.train()
    del forward_dict

    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to YAML config")
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--out",
        default="gamma_init_diagnostic.csv",
        help="output CSV path",
    )
    args = ap.parse_args()

    cfg = _load_cfg(Path(args.config))

    rows = []
    for seed in range(args.n_seeds):
        try:
            row = run_one_seed(cfg, seed, args.device)
            rows.append(row)
            print(
                f"[seed {seed:3d}] loss={row.get('loss_total', float('nan')):.2f} "
                f"qbg_k_min={row.get('qbg_k_min', float('nan')):.4f} "
                f"qi_k_min={row.get('qi_k_min', float('nan')):.4f} "
                f"qbg_lin_mu_gmax={row.get('qbg_linear_mu_grad_max', float('nan')):.2e}"
            )
        except Exception as exc:
            print(f"[seed {seed:3d}] FAILED at forward: {exc!r}")
            rows.append({"seed": seed, "error": str(exc)})

    # Union of all keys, write CSV
    all_keys: set[str] = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = ["seed"] + sorted(k for k in all_keys if k != "seed")

    out_path = Path(args.out)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nWrote {out_path} ({len(rows)} rows, {len(fieldnames)} columns)")


if __name__ == "__main__":
    main()
