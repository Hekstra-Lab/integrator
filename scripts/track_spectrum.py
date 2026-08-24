"""Track spectrum parameter evolution across checkpoints.

Usage:
    uv run python scripts/track_spectrum.py /path/to/checkpoints/
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _get_B(sd):
    if "loss.raw_B" in sd:
        return F.softplus(sd["loss.raw_B"]).item()
    return F.softplus(sd["loss.q_log_B_loc"]).item()


def main():
    ckpt_dir = Path(sys.argv[1])
    ckpts = sorted(ckpt_dir.glob("epoch*.ckpt"))
    if not ckpts:
        print(f"No epoch*.ckpt found in {ckpt_dir}")
        return

    for ckpt_path in ckpts:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        epoch = ckpt_path.stem
        B = _get_B(sd)

        # Point-estimate spectral (new)
        if "loss.spectrum.c" in sd:
            c = sd["loss.spectrum.c"]
            print(
                f"{epoch:>12s}  "
                f"c: [{c.min():.3f}, {c.max():.3f}]  "
                f"n_coeff: {c.shape[0]}  "
                f"B: {B:.3f}"
            )
        # Variational spectral (old)
        elif "loss.spectrum.coeff_loc" in sd:
            loc = sd["loss.spectrum.coeff_loc"]
            log_scale = sd["loss.spectrum.coeff_log_scale"]
            std = F.softplus(log_scale)
            print(
                f"{epoch:>12s}  "
                f"coeff_loc: [{loc.min():.3f}, {loc.max():.3f}]  "
                f"coeff_std: [{std.min():.3f}, {std.max():.3f}]  "
                f"mean_std: {std.mean():.4f}  "
                f"B: {B:.3f}"
            )
        # Binned
        elif "loss.q_log_K_loc" in sd:
            loc = sd["loss.q_log_K_loc"]
            log_scale = sd["loss.q_log_K_log_scale"]
            std = F.softplus(log_scale)
            print(
                f"{epoch:>12s}  "
                f"log_K_loc: [{loc.min():.3f}, {loc.max():.3f}]  "
                f"log_K_std: [{std.min():.3f}, {std.max():.3f}]  "
                f"mean_std: {std.mean():.4f}  "
                f"B: {B:.3f}"
            )


if __name__ == "__main__":
    main()
