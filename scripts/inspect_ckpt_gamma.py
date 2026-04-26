"""Inspect qi / qbg GammaB head weights across checkpoints.

Reports per checkpoint: bias value, weight norm, and the implied
softplus(bias ± ‖w‖) range — a rough proxy for the μ/fano dynamic
range the layer is currently capable of producing.

Usage:
    uv run python scripts/inspect_ckpt_gamma.py <dir_or_files...>

Examples:
    # Whole run
    uv run python scripts/inspect_ckpt_gamma.py \\
        /n/netscratch/.../run-20260418_134025-addypgrg/files/checkpoints/

    # Specific epochs of interest (peak signal 39/44/49)
    uv run python scripts/inspect_ckpt_gamma.py \\
        .../epoch=0039.ckpt .../epoch=0044.ckpt .../epoch=0049.ckpt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _find(sd: dict, sur: str, head: str, field: str):
    keys = [
        k
        for k in sd
        if f".{sur}." in k and f".{head}." in k and k.endswith(f".{field}")
    ]
    return sd[keys[0]] if keys else None


def _softplus(x: float) -> float:
    return float(F.softplus(torch.tensor(x)))


def _report_head(
    sur: str, head: str, bias: torch.Tensor, weight: torch.Tensor
):
    b = float(bias.flatten()[0])
    w = weight.flatten()
    wn = float(w.norm())
    wmax = float(w.abs().max())
    lo = _softplus(b - wn)
    hi = _softplus(b + wn)
    print(
        f"  {sur}.{head:<12} bias={b:>12.4f}  ‖w‖={wn:>8.4f}  "
        f"max|w|={wmax:>7.4f}  softplus(b±‖w‖) ≈ [{lo:.4g}, {hi:.4g}]"
    )


def inspect(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    print(f"\n=== {ckpt_path.name} ===")

    for sur in ("qi", "qbg"):
        found = False
        # Separate-heads layout (hierarchicalC)
        for head in ("linear_mu", "linear_fano"):
            w = _find(sd, sur, head, "weight")
            b = _find(sd, sur, head, "bias")
            if w is not None and b is not None:
                _report_head(sur, head, b, w)
                found = True
        # Shared fc layout (row 0 = mu, row 1 = fano)
        w = _find(sd, sur, "fc", "weight")
        b = _find(sd, sur, "fc", "bias")
        if w is not None and b is not None:
            _report_head(sur, "fc[mu]", b[0:1], w[0:1].flatten())
            _report_head(sur, "fc[fano]", b[1:2], w[1:2].flatten())
            found = True
        if not found:
            print(f"  {sur}: no GammaB heads found")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", type=Path)
    args = p.parse_args()

    ckpts: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            ckpts.extend(sorted(path.glob("epoch=*.ckpt")))
        elif path.is_file():
            ckpts.append(path)
        else:
            print(f"skip (not found): {path}", file=sys.stderr)
    if not ckpts:
        sys.exit("no checkpoints found")

    # Print an init-reference line so the numbers have a baseline.
    # softplus-inverse-shifted of mean_init ≈ 477 is ~477 (short-circuit
    # for delta>30); of fano_init=1.0 is ~0.541; of qbg mean_init≈0.57
    # is ~-0.25.
    print("Initialization reference (softplus mu, fano_init=1):")
    print("  qi.linear_mu.bias      ≈ mean_init (≈ 477 for typical data)")
    print("  qi.linear_fano.bias    ≈ 0.541")
    print("  qbg.linear_mu.bias     ≈ log(expm1(mean_init − eps)) ≈ -0.25")
    print("  qbg.linear_fano.bias   ≈ 0.541")
    print("  All weights start ~N(0, 1/sqrt(in_features)), ‖w‖ ≈ 0.18")

    for c in ckpts:
        inspect(c)


if __name__ == "__main__":
    main()
