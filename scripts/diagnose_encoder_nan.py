"""Diagnose NaN in intensity encoder — run on GPU with actual data.

Usage:
    python scripts/diagnose_encoder_nan.py --config <path_to_yaml>

Traces each layer of the intensity encoder to find where NaN first appears.
"""

import argparse

import torch
import torch.nn.functional as F

from integrator.utils.factory_utils import (
    construct_data_loader,
    construct_integrator,
    load_config,
)


def check(name, x):
    """Report NaN/Inf stats for a tensor."""
    n_nan = torch.isnan(x).sum().item()
    n_inf = torch.isinf(x).sum().item()
    n_total = x.numel()
    status = "OK" if (n_nan == 0 and n_inf == 0) else "*** PROBLEM ***"
    print(
        f"  {name:35s} shape={str(list(x.shape)):20s} "
        f"NaN={n_nan:6d} Inf={n_inf:6d} "
        f"range=[{x[torch.isfinite(x)].min():.4f}, {x[torch.isfinite(x)].max():.4f}] "
        f"{status}"
    )
    return n_nan > 0 or n_inf > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = construct_integrator(cfg)
    data = construct_data_loader(cfg)
    data.setup("fit")

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    print(f"Device: {device}")
    print(f"Batch size: {cfg['data_loader']['args']['batch_size']}")
    print()

    # Check multiple batches
    dl = data.train_dataloader()
    for batch_idx, batch in enumerate(dl):
        if batch_idx >= 3:  # check first 3 batches
            break

        counts, shoebox, mask, metadata = batch
        counts = counts.to(device)
        shoebox = shoebox.to(device)
        mask = mask.to(device)
        metadata = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in metadata.items()
        }

        B = shoebox.shape[0]
        print(f"{'=' * 70}")
        print(f"BATCH {batch_idx} (B={B})")
        print(f"{'=' * 70}")

        # Input stats
        print("\n## Input data")
        check("counts", counts)
        check("shoebox (standardized)", shoebox)

        # Check for problematic shoeboxes
        sb_per_refl = shoebox.reshape(B, -1)
        sb_std = sb_per_refl.std(dim=1)
        sb_range = (
            sb_per_refl.max(dim=1).values - sb_per_refl.min(dim=1).values
        )
        n_const = (sb_std < 1e-8).sum().item()
        n_extreme = (sb_per_refl.abs() > 1e4).any(dim=1).sum().item()
        print(f"  {'Constant shoeboxes (std<1e-8)':35s} count={n_const}")
        print(f"  {'Extreme values (|x|>1e4)':35s} count={n_extreme}")
        print(
            f"  {'shoebox std range':35s} [{sb_std.min():.6f}, {sb_std.max():.6f}]"
        )
        print(
            f"  {'shoebox value range':35s} [{shoebox.min():.4f}, {shoebox.max():.4f}]"
        )

        # Trace through intensity encoder layer by layer
        enc = model.encoders["intensity"]
        shoebox_reshaped = shoebox.reshape(B, 1, *model.shoebox_shape)
        x = shoebox_reshaped

        print("\n## Intensity encoder layer trace")
        check("input", x)

        x = enc.conv1(x)
        found = check("after conv1", x)

        x = enc.norm1(x)
        found = check("after norm1 (GroupNorm)", x)
        if found:
            # Detailed GroupNorm analysis
            print("\n  *** GroupNorm produced NaN! Investigating... ***")
            x_pre = enc.conv1(shoebox_reshaped)
            # Check per-group stats
            C = x_pre.shape[1]
            G = enc.norm1.num_groups
            ch_per_g = C // G
            for g in range(G):
                ch_slice = x_pre[:, g * ch_per_g : (g + 1) * ch_per_g]
                var = ch_slice.var(dim=(1, 2, 3))
                n_zero_var = (var < 1e-10).sum().item()
                print(
                    f"    Group {g}: var range=[{var.min():.8f}, {var.max():.8f}], "
                    f"zero_var={n_zero_var}/{B}"
                )

        x = F.relu(x)
        check("after relu1", x)

        x = enc.pool(x)
        check("after maxpool", x)

        x = enc.conv2(x)
        found = check("after conv2", x)

        x = enc.norm2(x)
        found = check("after norm2 (GroupNorm)", x)
        if found:
            print("\n  *** GroupNorm2 produced NaN! ***")

        x = F.relu(x)
        check("after relu2", x)

        x = enc.conv3(x)
        found = check("after conv3", x)

        x = enc.norm3(x)
        found = check("after norm3 (GroupNorm)", x)
        if found:
            print("\n  *** GroupNorm3 produced NaN! ***")

        x = F.relu(x)
        check("after relu3", x)

        x = enc.adaptive_pool(x)
        check("after adaptive_pool", x)

        print(f"  {'shape before squeeze':35s} {list(x.shape)}")
        x = x.squeeze()
        check("after squeeze", x)
        print(f"  {'shape after squeeze':35s} {list(x.shape)}")
        if x.dim() != 2:
            print(f"  *** SHAPE BUG: expected 2D (B, C), got {x.dim()}D ***")

        x = enc.fc(x)
        check("after fc", x)

        x = F.relu(x)
        found = check("after final relu (encoder out)", x)

        if found:
            # Which reflections have NaN?
            nan_mask = torch.isnan(x).any(dim=1)
            n_nan_refls = nan_mask.sum().item()
            print(
                f"\n  {n_nan_refls}/{B} reflections have NaN in encoder output"
            )
            if n_nan_refls < 20:
                nan_indices = torch.where(nan_mask)[0]
                print(f"  Indices: {nan_indices.tolist()}")
                for idx in nan_indices[:5]:
                    sb = shoebox[idx]
                    print(
                        f"    refl {idx.item()}: shoebox range=[{sb.min():.4f}, {sb.max():.4f}], "
                        f"std={sb.std():.6f}"
                    )
        else:
            print(f"\n  Encoder output is clean for batch {batch_idx}")

        # Also check profile encoder
        print("\n## Profile encoder (for comparison)")
        x_prof = model.encoders["profile"](shoebox_reshaped)
        check("profile encoder output", x_prof)

        print()


if __name__ == "__main__":
    main()
