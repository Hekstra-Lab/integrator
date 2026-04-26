"""
Cluster diagnostic suite for NaN investigation.
Run on GPU with the full dataset and real model.

Usage:
    python scripts/cluster_diagnostics.py --config <your_config.yaml> --max-steps 2000

Outputs a JSON file: diagnostics_output.json
"""

import argparse
import json

import torch

from integrator.utils import (
    construct_data_loader,
    construct_integrator,
    load_config,
)


def tensor_stats(t, name=""):
    """Summarize a tensor into a dict."""
    if t is None:
        return {"name": name, "value": None}
    t = t.detach().float()
    return {
        "name": name,
        "shape": list(t.shape),
        "min": t.min().item(),
        "max": t.max().item(),
        "mean": t.mean().item(),
        "std": t.std().item() if t.numel() > 1 else 0.0,
        "has_nan": t.isnan().any().item(),
        "has_inf": t.isinf().any().item(),
        "num_nan": t.isnan().sum().item(),
        "num_inf": t.isinf().sum().item(),
        "abs_max": t.abs().max().item(),
    }


def check_distribution(dist, name):
    """Check distribution parameters for NaN/Inf."""
    info = {"name": name}
    if hasattr(dist, "concentration"):
        info["concentration"] = tensor_stats(
            dist.concentration, f"{name}.concentration"
        )
    if hasattr(dist, "rate"):
        info["rate"] = tensor_stats(dist.rate, f"{name}.rate")
    if hasattr(dist, "loc"):
        info["loc"] = tensor_stats(dist.loc, f"{name}.loc")
    if hasattr(dist, "scale"):
        info["scale"] = tensor_stats(dist.scale, f"{name}.scale")
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument(
        "--output", type=str, default="diagnostics_output.json"
    )
    parser.add_argument("--check-every", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Build components ─────────────────────────────────────────────────
    data_loader = construct_data_loader(cfg)
    data_loader.setup()

    model = construct_integrator(cfg).to(device)
    loss_fn = model.loss
    optimizer = model.configure_optimizers()

    train_dl = data_loader.train_dataloader()

    # ── Diagnostic state ─────────────────────────────────────────────────
    results = {
        "config": args.config,
        "device": str(device),
        "steps": [],
        "nan_step": None,
        "nan_cause": None,
    }

    # ── Check data ───────────────────────────────────────────────────────
    batch = next(iter(train_dl))
    counts, shoebox, mask, meta = batch
    data_check = {
        "counts": tensor_stats(counts, "counts"),
        "shoebox": tensor_stats(shoebox, "shoebox (standardized)"),
        "mask": tensor_stats(mask, "mask"),
    }
    results["data_check"] = data_check
    print(
        f"Data check: shoebox range=[{shoebox.min():.2f}, {shoebox.max():.2f}]"
    )

    # ── Training loop ────────────────────────────────────────────────────
    model.train()
    step = 0
    dl_iter = iter(train_dl)

    for step in range(args.max_steps):
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(train_dl)
            batch = next(dl_iter)

        counts, shoebox, mask, meta = batch
        counts = counts.to(device)
        shoebox = shoebox.to(device)
        mask = mask.to(device)
        meta = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in meta.items()
        }

        # ── Forward pass ─────────────────────────────────────────────────
        try:
            outputs = model(counts, shoebox, mask, meta)
        except Exception as e:
            results["nan_step"] = step
            results["nan_cause"] = f"forward exception: {str(e)}"
            print(f"Step {step}: FORWARD EXCEPTION: {e}")
            break

        qp = outputs["qp"]
        qi = outputs["qi"]
        qbg = outputs["qbg"]
        forward_out = outputs["forward_out"]

        # ── Check distributions ──────────────────────────────────────────
        qi_info = check_distribution(qi, "qi")
        qbg_info = check_distribution(qbg, "qbg")
        qp_info = check_distribution(qp, "qp")

        # Check for NaN in distribution params
        qi_nan = any(
            v.get("has_nan", False)
            for v in qi_info.values()
            if isinstance(v, dict)
        )
        qbg_nan = any(
            v.get("has_nan", False)
            for v in qbg_info.values()
            if isinstance(v, dict)
        )

        if qi_nan or qbg_nan:
            results["nan_step"] = step
            results["nan_cause"] = "distribution params NaN before loss"
            step_data = {
                "step": step,
                "qi": qi_info,
                "qbg": qbg_info,
                "qp": qp_info,
            }
            # Check encoder outputs
            for name, p in model.named_parameters():
                if p.isnan().any():
                    step_data[f"nan_weight_{name}"] = True
            results["steps"].append(step_data)
            print(f"Step {step}: NaN in distribution params!")
            break

        # ── Loss computation ─────────────────────────────────────────────
        try:
            loss_dict = loss_fn(
                rate=forward_out["rates"],
                counts=forward_out["counts"],
                qp=qp,
                qi=qi,
                qbg=qbg,
                mask=forward_out["mask"],
            )
        except Exception as e:
            results["nan_step"] = step
            results["nan_cause"] = f"loss exception: {str(e)}"
            step_data = {
                "step": step,
                "qi": qi_info,
                "qbg": qbg_info,
                "qp": qp_info,
                "exception": str(e),
            }
            results["steps"].append(step_data)
            print(f"Step {step}: LOSS EXCEPTION: {e}")
            break

        loss = loss_dict["loss"]

        if loss.isnan() or loss.isinf():
            results["nan_step"] = step
            results["nan_cause"] = (
                f"loss is {'NaN' if loss.isnan() else 'Inf'}"
            )
            step_data = {
                "step": step,
                "loss": loss.item() if not loss.isnan() else "NaN",
                "neg_ll": loss_dict["neg_ll_mean"].item(),
                "kl_prf": loss_dict["kl_prf_mean"].item(),
                "kl_i": loss_dict["kl_i_mean"].item(),
                "kl_bg": loss_dict["kl_bg_mean"].item(),
                "qi": qi_info,
                "qbg": qbg_info,
                "qp": qp_info,
                "rate_max": forward_out["rates"].max().item(),
                "rate_min": forward_out["rates"].min().item(),
            }
            results["steps"].append(step_data)
            print(f"Step {step}: loss is {results['nan_cause']}!")
            break

        # ── Backward + clip ──────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()

        clip_val = cfg.get("trainer", {}).get("gradient_clip_val", 1.0) or 1.0
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip_val
        )

        if torch.isnan(grad_norm):
            results["nan_step"] = step
            results["nan_cause"] = "gradient norm NaN"
            step_data = {
                "step": step,
                "loss": loss.item(),
                "grad_norm": "NaN",
                "nan_grad_params": [],
            }
            for name, p in model.named_parameters():
                if p.grad is not None and p.grad.isnan().any():
                    step_data["nan_grad_params"].append(name)
            results["steps"].append(step_data)
            print(
                f"Step {step}: gradient NaN! Params: {step_data['nan_grad_params']}"
            )
            break

        optimizer.step()

        # ── Periodic logging ─────────────────────────────────────────────
        if step % args.check_every == 0:
            step_data = {
                "step": step,
                "loss": loss.item(),
                "neg_ll": loss_dict["neg_ll_mean"].item(),
                "kl_prf": loss_dict["kl_prf_mean"].item(),
                "kl_i": loss_dict["kl_i_mean"].item(),
                "kl_bg": loss_dict["kl_bg_mean"].item(),
                "grad_norm": grad_norm.item(),
                "rate_max": forward_out["rates"].max().item(),
                "rate_min": forward_out["rates"].min().item(),
                "qi": qi_info,
                "qbg": qbg_info,
                "qp_alpha_min": qp.concentration.min().item(),
                "qp_alpha_max": qp.concentration.max().item(),
                "qp_alpha_sum_mean": qp.concentration.sum(-1).mean().item(),
            }

            # Weight norms
            weight_norms = {}
            for name, p in model.named_parameters():
                weight_norms[name] = p.abs().max().item()
            step_data["weight_max_abs"] = weight_norms

            results["steps"].append(step_data)

            # Print summary — extract qi param range
            if "concentration" in qi_info:
                qi_stats = qi_info["concentration"]
            elif "loc" in qi_info:
                qi_stats = qi_info["loc"]
            else:
                qi_stats = {"min": "?", "max": "?"}
            qi_k_range = f"[{qi_stats['min']:.4g}, {qi_stats['max']:.4g}]"
            print(
                f"Step {step:5d}: loss={loss.item():>10.1f}  "
                f"nll={loss_dict['neg_ll_mean'].item():>9.1f}  "
                f"kl_prf={loss_dict['kl_prf_mean'].item():>7.0f}  "
                f"kl_i={loss_dict['kl_i_mean'].item():>7.1f}  "
                f"qi_params={qi_k_range}  "
                f"grad={grad_norm.item():.3f}  "
                f"rate_max={forward_out['rates'].max().item():.0f}"
            )

    # ── Save results ─────────────────────────────────────────────────────
    results["total_steps"] = step + 1

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {args.output}")
    if results["nan_step"] is not None:
        print(
            f"NaN detected at step {results['nan_step']}: {results['nan_cause']}"
        )
    else:
        print(f"No NaN in {results['total_steps']} steps")


if __name__ == "__main__":
    main()
