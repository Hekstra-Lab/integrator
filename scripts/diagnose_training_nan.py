"""Diagnose NaN during training — runs actual optimizer steps.

Usage:
    python scripts/diagnose_training_nan.py --config <path_to_yaml>

Runs training loop and prints one compact line per step.
Dumps full details when NaN first appears.
"""

import argparse

import torch

from integrator.utils.factory_utils import (
    construct_data_loader,
    construct_integrator,
    load_config,
)


def has_bad(x):
    """True if tensor has NaN or Inf."""
    if x is None:
        return False
    if isinstance(x, (int, float)):
        return x != x or abs(x) == float("inf")
    return torch.isnan(x).any().item() or torch.isinf(x).any().item()


def fmt(x):
    """Compact format for a scalar tensor."""
    if isinstance(x, torch.Tensor):
        x = x.item()
    if abs(x) < 1e-2 or abs(x) > 1e5:
        return f"{x:.3e}"
    return f"{x:.4f}"


def dump(name, x):
    """Full NaN/Inf report for a tensor."""
    n_nan = torch.isnan(x).sum().item()
    n_inf = torch.isinf(x).sum().item()
    finite = x[torch.isfinite(x)]
    rng = (
        f"[{finite.min():.4g}, {finite.max():.4g}]"
        if finite.numel() > 0
        else "[none]"
    )
    tag = " *** BAD ***" if (n_nan > 0 or n_inf > 0) else ""
    print(f"    {name:40s} NaN={n_nan:5d} Inf={n_inf:5d} range={rng}{tag}")


def dump_params(name, module):
    for pname, p in module.named_parameters():
        if has_bad(p.data):
            dump(f"{name}.{pname} (param)", p.data)
        if p.grad is not None and has_bad(p.grad):
            dump(f"{name}.{pname} (grad)", p.grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = construct_integrator(cfg)
    data = construct_data_loader(cfg)
    data.setup("fit")

    device = torch.device(args.device)
    model = model.to(device)
    model.train()

    if hasattr(model.loss, "dataset_size"):
        model.loss.dataset_size = len(data.train_dataset)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_train = len(data.train_dataset)
    batch_size = cfg["data_loader"]["args"]["batch_size"]
    steps_per_epoch = n_train // batch_size
    print(f"Device: {device}")
    print(
        f"Train set: {n_train}, batch_size: {batch_size}, steps/epoch: {steps_per_epoch}"
    )
    print(
        f"Running {args.steps} steps ({args.steps / steps_per_epoch:.1f} epochs)"
    )
    print()
    print(
        f"{'step':>5s} {'epoch':>5s} | {'loss':>10s} {'nll':>10s} {'kl_i':>8s} "
        f"{'kl_bg':>8s} {'kl_prf':>8s} {'kl_gl':>8s} {'beta':>5s} | "
        f"{'tau_mn':>8s} {'mu_mn':>8s} {'lv_mn':>8s} | "
        f"{'qi_k_mn':>8s} {'qi_k_mx':>8s} {'qi_b_mn':>8s} {'qi_b_mx':>8s}"
    )
    print("-" * 150)

    dl = data.train_dataloader()
    dl_iter = iter(dl)

    for step in range(args.steps):
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            batch = next(dl_iter)

        counts, shoebox, mask, metadata = batch
        counts = counts.to(device)
        shoebox = shoebox.to(device)
        mask = mask.to(device)
        metadata = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in metadata.items()
        }

        optimizer.zero_grad()

        # Update epoch for KL warmup
        epoch = step / steps_per_epoch
        if hasattr(model.loss, "current_epoch"):
            model.loss.current_epoch = int(epoch)

        # Forward
        try:
            outputs = model(counts, shoebox, mask, metadata)
        except Exception as e:
            print(
                f"\n*** FORWARD CRASHED at step {step} (epoch {step / steps_per_epoch:.2f}): {e}"
            )
            # Trace through manually
            B = shoebox.shape[0]
            shoebox_reshaped = shoebox.reshape(B, 1, *model.shoebox_shape)
            x_int = model.encoders["intensity"](shoebox_reshaped)
            dump("x_intensity", x_int)
            group_labels = metadata["group_label"].long()
            ge = model.group_encoder
            z = ge.phi(x_int)
            dump("phi(x_int)", z)
            unique_groups = torch.unique(group_labels)
            gm = [z[group_labels == k].mean(dim=0) for k in unique_groups]
            gf = torch.stack(gm)
            dump("group_features", gf)
            h = ge.rho(gf)
            dump("rho output", h)
            mu_dbg = ge.head_mu(h).squeeze(-1)
            lv_dbg = ge.head_logvar(h).squeeze(-1).clamp(-10, 4)
            print(f"    mu:     {mu_dbg.tolist()}")
            print(f"    logvar: {lv_dbg.tolist()}")
            std_dbg = torch.exp(0.5 * lv_dbg)
            print(f"    std:    {std_dbg.tolist()}")
            dump_params("group_encoder", ge)
            dump_params("intensity_enc", model.encoders["intensity"])
            break

        fwd = outputs["forward_out"]

        # Loss
        try:
            loss_dict = model.loss(
                rate=fwd["rates"],
                counts=fwd["counts"],
                qp=outputs["qp"],
                qi=outputs["qi"],
                qbg=outputs["qbg"],
                mask=fwd["mask"],
                mu=outputs["mu"],
                logvar=outputs["logvar"],
                tau_per_refl=outputs["tau_per_refl"],
                group_labels=outputs["group_labels"],
            )
        except Exception as e:
            print(
                f"\n*** LOSS CRASHED at step {step} (epoch {step / steps_per_epoch:.2f}): {e}"
            )
            dump("rates", fwd["rates"])
            dump("mu", outputs["mu"])
            dump("logvar", outputs["logvar"])
            dump("tau_per_refl", outputs["tau_per_refl"])
            print(f"    mu values:     {outputs['mu'].tolist()}")
            print(f"    logvar values: {outputs['logvar'].tolist()}")
            qi = outputs["qi"]
            if hasattr(qi, "concentration"):
                dump("qi.concentration", qi.concentration)
                dump("qi.rate", qi.rate)
            break

        loss = loss_dict["loss"]

        # Print compact line
        qi = outputs["qi"]
        qi_k = (
            qi.concentration
            if hasattr(qi, "concentration")
            else torch.tensor(0)
        )
        qi_b = qi.rate if hasattr(qi, "rate") else torch.tensor(0)

        beta = loss_dict.get("beta_kl_warmup", 1.0)
        line = (
            f"{step:5d} {epoch:5.2f} | "
            f"{fmt(loss):>10s} {fmt(loss_dict['neg_ll_mean']):>10s} "
            f"{fmt(loss_dict['kl_i_mean']):>8s} {fmt(loss_dict['kl_bg_mean']):>8s} "
            f"{fmt(loss_dict['kl_prf_mean']):>8s} {fmt(loss_dict['kl_global']):>8s} "
            f"{beta:5.2f} | "
            f"{fmt(loss_dict['tau_mean']):>8s} "
            f"{fmt(outputs['mu'].mean()):>8s} {fmt(outputs['logvar'].mean()):>8s} | "
            f"{fmt(qi_k.min()):>8s} {fmt(qi_k.max()):>8s} "
            f"{fmt(qi_b.min()):>8s} {fmt(qi_b.max()):>8s}"
        )
        print(line, flush=True)

        if has_bad(loss):
            print(f"\n*** LOSS IS NaN/Inf at step {step} ***")
            for k, v in loss_dict.items():
                print(f"    {k}: {v}")
            break

        # Backward
        try:
            loss.backward()
        except Exception as e:
            print(f"\n*** BACKWARD CRASHED at step {step}: {e}")
            break

        # Quick gradient check
        grad_bad = False
        for name, mod in [
            ("int_enc", model.encoders["intensity"]),
            ("prf_enc", model.encoders["profile"]),
            ("grp_enc", model.group_encoder),
            ("qi", model.surrogates["qi"]),
            ("qbg", model.surrogates["qbg"]),
            ("qp", model.surrogates["qp"]),
        ]:
            for pname, p in mod.named_parameters():
                if p.grad is not None and has_bad(p.grad):
                    if not grad_bad:
                        print(f"\n*** NaN/Inf GRADIENTS at step {step} ***")
                    grad_bad = True
                    dump(f"{name}.{pname} (grad)", p.grad)

        if grad_bad:
            break

        optimizer.step()

        # Quick param check
        param_bad = False
        for name, mod in [
            ("int_enc", model.encoders["intensity"]),
            ("grp_enc", model.group_encoder),
            ("qi", model.surrogates["qi"]),
            ("qbg", model.surrogates["qbg"]),
            ("qp", model.surrogates["qp"]),
        ]:
            for pname, p in mod.named_parameters():
                if has_bad(p.data):
                    if not param_bad:
                        print(f"\n*** NaN/Inf PARAMS at step {step} ***")
                    param_bad = True
                    dump(f"{name}.{pname}", p.data)

        if param_bad:
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
