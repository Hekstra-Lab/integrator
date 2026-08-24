"""Diagnose Bijvoet differences in the scaling model across training.

For each checkpoint, extracts the learned F(+) and F(-) from the HKL
table, matches Friedel pairs, and reports DANO statistics and how
they evolve over epochs.

Usage
-----
    uv run python scripts/diagnose_bijvoet.py <run_dir> [--out bijvoet.png]
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

import gemmi
import reciprocalspaceship as rs


def load_run_info(run_dir: Path) -> tuple[dict, Path]:
    meta = yaml.safe_load((run_dir / "run_paths.yaml").read_text())
    cfg = yaml.safe_load((run_dir / "config_log.yaml").read_text())
    log_dir = Path(meta["wandb"]["log_dir"])
    ckpt_dir = log_dir / "checkpoints"
    return cfg, ckpt_dir


def detect_table_type(state_dict: dict) -> str:
    if "hkl_table.raw_fano.weight" in state_dict:
        return "gamma"
    if "hkl_table.raw_sigma.weight" in state_dict:
        return "amplitude"
    if "hkl_table.raw_k.weight" in state_dict:
        return "gammaA"
    if "hkl_table.linear_fano.weight" in state_dict:
        return "encoder_fano"
    raise KeyError("Cannot detect HKL table type.")


def extract_F(state_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract F and sigma_F from checkpoint. Returns (F_mean, sig_F)."""
    table_type = detect_table_type(state_dict)

    if table_type == "amplitude":
        raw_mu = state_dict["hkl_table.raw_mu.weight"].cpu().squeeze(-1)
        raw_sigma = state_dict["hkl_table.raw_sigma.weight"].cpu().squeeze(-1)
        mu = torch.exp(raw_mu).numpy()
        sigma = (F.softplus(raw_sigma) + 1e-6).numpy()
        return mu, sigma
    elif table_type == "gammaA":
        raw_k = state_dict["hkl_table.raw_k.weight"].cpu().squeeze(-1)
        raw_rate = state_dict["hkl_table.raw_rate.weight"].cpu().squeeze(-1)
        k = F.softplus(raw_k) + 0.1
        rate = F.softplus(raw_rate) + 1e-6
        I_mean = (k / rate).numpy()
        F_mean = np.sqrt(np.clip(I_mean, 0, None))
        I_var = (k / rate.pow(2)).numpy()
        sig_F = np.where(F_mean > 0, np.sqrt(I_var) / (2 * F_mean), 0.0)
        return F_mean, sig_F
    elif table_type == "encoder_fano":
        raw_mu = state_dict["hkl_table.raw_mu.weight"].cpu().squeeze(-1)
        mu = F.softplus(raw_mu) + 1e-6
        I_mean = mu.numpy()
        F_mean = np.sqrt(np.clip(I_mean, 0, None))
        sig_F = np.where(F_mean > 0, 0.5 * np.ones_like(F_mean), 0.0)
        return F_mean, sig_F
    else:
        raw_mu = state_dict["hkl_table.raw_mu.weight"].cpu().squeeze(-1)
        raw_fano = state_dict["hkl_table.raw_fano.weight"].cpu().squeeze(-1)
        mu = torch.exp(raw_mu)
        fano = F.softplus(raw_fano) + 1e-6
        rate = 1.0 / fano
        k = mu * rate + 0.1
        I_mean = (k / rate).numpy()
        F_mean = np.sqrt(np.clip(I_mean, 0, None))
        I_var = (k / rate.pow(2)).numpy()
        sig_F = np.where(F_mean > 0, np.sqrt(I_var) / (2 * F_mean), 0.0)
        return F_mean, sig_F


def build_friedel_pairs(
    id_to_hkl: torch.Tensor, spacegroup: gemmi.SpaceGroup
) -> list[tuple[int, int]]:
    """Find pairs of asu_ids that are Friedel mates."""
    n = len(id_to_hkl)
    hkl_arr = id_to_hkl.numpy().astype(np.int32)

    # Map each HKL to ASU form via hkl_to_asu
    asu_hkl, isym = rs.utils.hkl_to_asu(hkl_arr, spacegroup)

    # Group by ASU form
    asu_to_ids: dict[tuple[int, int, int], list[tuple[int, bool]]] = {}
    for aid in range(n):
        key = tuple(asu_hkl[aid])
        is_plus = (isym[aid] % 2 == 1)
        if key not in asu_to_ids:
            asu_to_ids[key] = []
        asu_to_ids[key].append((aid, is_plus))

    pairs = []
    for key, entries in asu_to_ids.items():
        plus_ids = [aid for aid, is_plus in entries if is_plus]
        minus_ids = [aid for aid, is_plus in entries if not is_plus]
        if plus_ids and minus_ids:
            pairs.append((plus_ids[0], minus_ids[0]))

    return pairs


def analyze_checkpoint(
    ckpt_path: Path,
    pairs: list[tuple[int, int]],
) -> dict:
    """Compute Bijvoet statistics for one checkpoint."""
    sd = torch.load(ckpt_path, weights_only=False, map_location="cpu")[
        "state_dict"
    ]
    F_mean, sig_F = extract_F(sd)

    plus_idx = np.array([p for p, m in pairs])
    minus_idx = np.array([m for p, m in pairs])

    Fp = F_mean[plus_idx]
    Fm = F_mean[minus_idx]

    dano = Fp - Fm
    Fmean_pair = (Fp + Fm) / 2.0
    dano_over_F = np.where(Fmean_pair > 1, dano / Fmean_pair, 0.0)

    return {
        "dano_mean": float(np.mean(dano)),
        "dano_std": float(np.std(dano)),
        "dano_abs_mean": float(np.mean(np.abs(dano))),
        "dano_abs_max": float(np.max(np.abs(dano))),
        "dano_over_F_std": float(np.std(dano_over_F[Fmean_pair > 1])),
        "n_pairs": len(pairs),
        "n_significant": int((np.abs(dano) > 2 * sig_F[plus_idx]).sum()),
        "F_mean": float(np.mean(Fmean_pair)),
        "Fp": Fp,
        "Fm": Fm,
        "dano": dano,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=str, default="bijvoet_diagnostic.png")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    cfg, ckpt_dir = load_run_info(run_dir)

    data_dir = Path(cfg["data_loader"]["args"]["data_dir"])
    id_to_hkl = torch.load(
        data_dir / "asu_id_to_hkl.pt", weights_only=False, map_location="cpu"
    )

    crystal = yaml.safe_load((data_dir / "crystal.yaml").read_text())
    sg_str = crystal.get("space_group", "P1").split("(")[0].strip()
    spacegroup = gemmi.SpaceGroup(sg_str)

    print(f"Building Friedel pairs from {len(id_to_hkl)} asu_ids...")
    pairs = build_friedel_pairs(id_to_hkl, spacegroup)
    print(f"Found {len(pairs)} Friedel pairs")

    # Process all checkpoints
    ckpts = sorted(ckpt_dir.glob("epoch=*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")

    print(f"Processing {len(ckpts)} checkpoints...")

    epochs = []
    stats_list = []

    for ckpt_path in ckpts:
        epoch = int(ckpt_path.stem.split("=")[1])
        stats = analyze_checkpoint(ckpt_path, pairs)
        epochs.append(epoch)
        stats_list.append(stats)
        print(
            f"  epoch {epoch:4d}: |DANO| mean={stats['dano_abs_mean']:.3f}, "
            f"std={stats['dano_std']:.3f}, DANO/F std={stats['dano_over_F_std']:.4f}, "
            f"n_signif={stats['n_significant']}"
        )

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # 1. |DANO| mean over epochs
    ax = axes[0, 0]
    ax.plot(epochs, [s["dano_abs_mean"] for s in stats_list], "o-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("|DANO| mean")
    ax.set_title("|DANO| mean vs epoch")
    ax.grid(True, alpha=0.3)

    # 2. DANO/F std over epochs
    ax = axes[0, 1]
    ax.plot(epochs, [s["dano_over_F_std"] for s in stats_list], "o-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("std(DANO / <F>)")
    ax.set_title("Relative Bijvoet difference vs epoch")
    ax.grid(True, alpha=0.3)

    # 3. Number of significant pairs
    ax = axes[0, 2]
    ax.plot(epochs, [s["n_significant"] for s in stats_list], "o-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("N pairs with |DANO| > 2sigma")
    ax.set_title("Significant Bijvoet pairs vs epoch")
    ax.grid(True, alpha=0.3)

    # 4. F(+) vs F(-) scatter for last checkpoint
    ax = axes[1, 0]
    last = stats_list[-1]
    ax.scatter(last["Fp"], last["Fm"], s=0.5, alpha=0.1, c="steelblue")
    lim = max(last["Fp"].max(), last["Fm"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=0.5)
    ax.set_xlabel("F(+)")
    ax.set_ylabel("F(-)")
    ax.set_title(f"F(+) vs F(-) - epoch {epochs[-1]}")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 5. DANO histogram for last checkpoint
    ax = axes[1, 1]
    dano = last["dano"]
    ax.hist(dano, bins=100, color="steelblue", alpha=0.7, range=(-20, 20))
    ax.axvline(0, color="r", linestyle="--", linewidth=0.5)
    ax.set_xlabel("DANO = F(+) - F(-)")
    ax.set_ylabel("Count")
    ax.set_title(f"DANO distribution - epoch {epochs[-1]}")
    ax.grid(True, alpha=0.3)

    # 6. DANO vs <F> for last checkpoint
    ax = axes[1, 2]
    Fmean_pair = (last["Fp"] + last["Fm"]) / 2
    ax.scatter(Fmean_pair, np.abs(last["dano"]), s=0.5, alpha=0.1, c="steelblue")
    ax.set_xlabel("<F>")
    ax.set_ylabel("|DANO|")
    ax.set_title(f"|DANO| vs <F> - epoch {epochs[-1]}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Bijvoet difference diagnostic - {len(pairs)} Friedel pairs, "
        f"{len(ckpts)} epochs",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
