"""Measure gradient interference between qi and qbg KL terms.

For each encoder used by both qi and qbg (the 'shared trunk'), computes:
  g_qi  = ∇_{trunk params} kl_i_mean
  g_qbg = ∇_{trunk params} kl_bg_mean
  cos_sim = <g_qi, g_qbg> / (‖g_qi‖ ‖g_qbg‖)

Interpretation:
  cos > 0        — tasks push trunk in the same direction; no interference
  cos ≈ 0        — tasks use orthogonal directions; neutral coexistence
  cos < 0        — tasks actively conflict at this trunk (the hypothesis)
  ‖g‖ ≈ 0        — encoder doesn't see this task's gradient (expected in hierC)

Gradients are accumulated over `--batches` batches of reflections; the
reported cosine is the similarity of the accumulated gradients (i.e. the
signal the optimizer actually follows).

Run:
    uv run python scripts/gradient_interference.py \\
        --config configs/wilson_comparison/hierA_learned.yaml \\
        --checkpoint /path/to/last.ckpt \\
        --batches 8 --subset-size 1000
"""

import argparse

import torch
import torch.nn.functional as F

from integrator.utils import (
    construct_data_loader,
    construct_integrator,
    load_config,
)


def _flatten_grads(
    grads: tuple, params: list, device: torch.device
) -> torch.Tensor:
    parts = []
    for g, p in zip(grads, params, strict=True):
        if g is None:
            parts.append(torch.zeros(p.numel(), device=device))
        else:
            parts.append(g.detach().flatten())
    return torch.cat(parts)


def _move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to(v, device) for k, v in obj.items()}
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    # Shrink data pipeline for fast diagnostic run
    cfg["data_loader"]["args"]["subset_size"] = args.subset_size
    cfg["data_loader"]["args"]["num_workers"] = 0
    cfg["data_loader"]["args"]["batch_size"] = args.batch_size

    integrator = construct_integrator(cfg)
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = integrator.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"missing keys: {missing[:5]} ...")
    if unexpected:
        print(f"unexpected keys: {unexpected[:5]} ...")
    integrator = integrator.to(args.device)
    integrator.eval()  # disables dropout/BN running updates; rsample still works

    dm = construct_data_loader(cfg)
    dm.prepare_data()
    dm.setup("fit")
    dl = dm.train_dataloader()

    # Identify encoders to test: any encoder that isn't 'profile'.
    # qi / qbg read from these; profile feeds qp only.
    trunk_names = [n for n in integrator.encoders.keys() if n != "profile"]
    print(f"Testing encoders: {trunk_names}")
    print(
        f"Data subset: N={args.subset_size}, batch_size={args.batch_size}, batches={args.batches}"
    )

    accumulated: dict[str, dict[str, torch.Tensor | None]] = {
        name: {"g_qi": None, "g_qbg": None} for name in trunk_names
    }

    n_used = 0
    for i, batch in enumerate(dl):
        if i >= args.batches:
            break
        counts, shoebox, mask, metadata = batch  # type: ignore[misc]
        counts = counts.to(args.device)
        shoebox = shoebox.to(args.device)
        mask = mask.to(args.device)
        metadata = _move_to(metadata, args.device)

        outputs = integrator(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]
        group_labels = metadata["group_label"].long()

        loss_dict = integrator.loss(
            rate=forward_out["rates"],
            counts=forward_out["counts"],
            qp=outputs["qp"],
            qi=outputs["qi"],
            qbg=outputs["qbg"],
            mask=forward_out["mask"],
            group_labels=group_labels,
            metadata=metadata,
        )
        if "kl_i_mean" not in loss_dict or "kl_bg_mean" not in loss_dict:
            raise RuntimeError(
                "loss_dict missing kl_i_mean/kl_bg_mean — is this a Wilson-style loss?"
            )
        kl_i = loss_dict["kl_i_mean"]
        kl_bg = loss_dict["kl_bg_mean"]

        for name in trunk_names:
            params = list(integrator.encoders[name].parameters())
            g_qi = torch.autograd.grad(
                kl_i,
                params,
                retain_graph=True,
                allow_unused=True,
            )
            g_qbg = torch.autograd.grad(
                kl_bg,
                params,
                retain_graph=True,
                allow_unused=True,
            )
            g_qi_flat = _flatten_grads(g_qi, params, torch.device(args.device))
            g_qbg_flat = _flatten_grads(
                g_qbg, params, torch.device(args.device)
            )

            prev_qi = accumulated[name]["g_qi"]
            prev_qbg = accumulated[name]["g_qbg"]
            if prev_qi is None:
                accumulated[name]["g_qi"] = g_qi_flat.clone()
                accumulated[name]["g_qbg"] = g_qbg_flat.clone()
            else:
                assert prev_qbg is not None
                accumulated[name]["g_qi"] = prev_qi + g_qi_flat
                accumulated[name]["g_qbg"] = prev_qbg + g_qbg_flat

        n_used += 1

    print(f"\nAccumulated over {n_used} batches")
    header = f"{'encoder':<16} {'cos(g_qi,g_qbg)':>18} {'angle':>8}  {'‖g_qi‖':>12} {'‖g_qbg‖':>12}  notes"
    print("\n" + header)
    print("-" * len(header))
    for name, grads in accumulated.items():
        g_qi = grads["g_qi"]
        g_qbg = grads["g_qbg"]
        if g_qi is None or g_qbg is None:
            print(f"{name:<16} {'no data':>18}")
            continue
        norm_qi = g_qi.norm().item()
        norm_qbg = g_qbg.norm().item()
        note = ""
        if norm_qi < 1e-10 and norm_qbg < 1e-10:
            note = "both ~0 (unused encoder)"
            cos = float("nan")
            angle = float("nan")
        elif norm_qi < 1e-10:
            note = "no qi gradient (encoder not in qi path)"
            cos = float("nan")
            angle = float("nan")
        elif norm_qbg < 1e-10:
            note = "no qbg gradient (encoder not in qbg path)"
            cos = float("nan")
            angle = float("nan")
        else:
            cos = F.cosine_similarity(g_qi, g_qbg, dim=0).item()
            angle = torch.rad2deg(
                torch.acos(torch.tensor(cos).clamp(-1.0, 1.0))
            ).item()

        cos_str = f"{cos:+.6f}" if not torch.isnan(torch.tensor(cos)) else "—"
        ang_str = (
            f"{angle:6.1f}°" if not torch.isnan(torch.tensor(angle)) else "—"
        )
        print(
            f"{name:<16} {cos_str:>18} {ang_str:>8}  "
            f"{norm_qi:>12.2e} {norm_qbg:>12.2e}  {note}"
        )

    print("\nInterpretation:")
    print(
        "  cos > 0      tasks push trunk in same direction — no interference"
    )
    print("  cos ≈ 0      tasks use orthogonal subspaces — neutral")
    print(
        "  cos < 0      tasks actively conflict at this trunk — interference"
    )
    print(
        "  cos far from 0 with ‖g‖ of same order on both sides = strong signal"
    )


if __name__ == "__main__":
    main()
