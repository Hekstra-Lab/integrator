"""Score a trained run on metrics that do not need ground-truth intensities.

Rebuilds the model exactly as `integrator.train` did (resolved config, binning
labels, checkpoint), runs it in eval mode over every shoebox, and reports:

1. Spot shape on held-out bright spots: cosine similarity between the model's
   profile and the border-median-subtracted counts, on validation shoeboxes that are
   isolated (no masked pixel), free of harmonics, and in the top `--bright-frac` by
   the summation estimate.
2. Whether profile breadth tracks signal: Spearman correlation between the profile's
   effective pixel count `n_eff = 1 / sum(p^2)` and the summation I/sigma, plus the
   median `n_eff` per I/sigma decile.
3. Consistency of symmetry equivalents: an unweighted half-set correlation (CC1/2
   analogue). Each arm's intensities are first divided by their own mean in a
   wavelength x resolution cell, the same procedure for every arm, since Laue
   intensities carry a wavelength-dependent scale. Observations of each unique
   reflection (Laue class mmm for P212121, Friedel mates merged) are split into two
   halves by a seeded hash of `refl_ids`, identical for every run, and the halves
   are averaged without weights.

Every selection uses a model-independent quantity (summation estimate, masks,
harmonic flag), never the model's own output, so two arms are scored on the same
reflections.

Run:
    python evaluate_profile_run.py --run-dir RUN_DIR [--ckpt last] [--device cuda]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from integrator.utils import (
    apply_dataset_defaults,
    construct_data_loader,
    construct_integrator,
    inject_binning_labels,
    prepare_global_priors,
    prepare_per_bin_priors,
    resolve_config,
)

logging.basicConfig(level=logging.WARNING)


def load_run(run_dir: Path, ckpt: str, device: str):
    cfg = yaml.safe_load((run_dir / "config_log.yaml").read_text())
    cfg["data_loader"]["args"]["num_workers"] = 0
    cfg = resolve_config(apply_dataset_defaults(cfg))
    prepare_per_bin_priors(cfg)
    prepare_global_priors(cfg)
    dm = construct_data_loader(cfg)
    dm.setup()
    inject_binning_labels(dm, cfg)
    model = construct_integrator(cfg)
    ckpt_dir = run_dir / "files" / "checkpoints"
    path = ckpt_dir / ("last.ckpt" if ckpt == "last" else ckpt)
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=True)
    return cfg, dm, model.to(device).eval(), path


@torch.no_grad()
def collect(model, dm, device: str, batch_size: int, max_batches: int | None):
    ds = dm.full_dataset
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    keys = ("refl_ids", "H", "K", "L", "d", "wavelength", "n_harmonics", "image_num",
            "intensity.sum.value", "intensity.sum.variance")
    rows: dict[str, list] = {k: [] for k in (*keys, "qi_mean", "qi_sd", "n_eff", "cos_bright_ready")}
    profiles, sig = [], []
    for b, (counts, shoebox, mask, meta) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        meta_d = {k: v.to(device) for k, v in meta.items()}
        out = model(counts.to(device), shoebox.to(device), mask.to(device), meta_d)
        qi, qp = out["qi"], out["qp"]
        p = qp.mean_profile.reshape(counts.shape[0], -1).float().cpu()
        for k in keys:
            rows[k].append(meta[k].reshape(-1).float().cpu() if k in meta else torch.full((counts.shape[0],), float("nan")))
        rows["qi_mean"].append(qi.mean.reshape(-1).float().cpu())
        rows["qi_sd"].append(qi.variance.reshape(-1).sqrt().float().cpu())
        rows["n_eff"].append(1.0 / (p**2).sum(-1).clamp_min(1e-30))
        m = mask.reshape(counts.shape[0], -1).float()
        c = counts.reshape(counts.shape[0], -1).float()
        side = int(round(c.shape[1] ** 0.5))
        border = torch.ones(side, side, dtype=torch.bool)
        border[2:-2, 2:-2] = False
        bg = c[:, border.reshape(-1)].median(-1).values[:, None]
        s = (c - bg).clamp(min=0) * m
        rows["cos_bright_ready"].append((m.sum(-1) == m.shape[1]).float())
        profiles.append(p)
        sig.append(s)
    out = {k: torch.cat(v).numpy() for k, v in rows.items()}
    out["cos"] = (torch.cat(profiles) * torch.cat(sig)).sum(-1).div(
        torch.cat(profiles).norm(dim=-1) * torch.cat(sig).norm(dim=-1) + 1e-30
    ).numpy()
    return out


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def normalize_by_cell(i, wavelength, d, n_lambda=20, n_res=20):
    """Divide by the mean intensity in its wavelength x resolution cell."""
    s2 = 1.0 / (4 * d**2)
    lb = np.digitize(wavelength, np.quantile(wavelength, np.linspace(0, 1, n_lambda + 1)[1:-1]))
    rb = np.digitize(s2, np.quantile(s2, np.linspace(0, 1, n_res + 1)[1:-1]))
    cell = lb * n_res + rb
    means = np.bincount(cell, weights=i, minlength=n_lambda * n_res) / np.maximum(
        np.bincount(cell, minlength=n_lambda * n_res), 1
    )
    return i / np.where(means[cell] > 0, means[cell], np.nan), rb


def half_set_cc(i_norm, hkl, refl_ids, res_bin, n_res=20, seed=12345):
    """Unweighted half-set correlation over unique reflections (mmm, Friedel merged)."""
    key = np.abs(hkl).astype(np.int64)
    uid = (key[:, 0] * 1000 + key[:, 1]) * 1000 + key[:, 2]
    half = (np.random.default_rng(seed).permutation(int(refl_ids.max()) + 1)[refl_ids.astype(np.int64)] % 2)
    ok = np.isfinite(i_norm)
    out = {}
    for name, sel in [("all", ok)] + [(f"shell{r:02d}", ok & (res_bin == r)) for r in range(n_res)]:
        u, inv = np.unique(uid[sel], return_inverse=True)
        sums = np.zeros((2, len(u)))
        cnts = np.zeros((2, len(u)))
        for h in (0, 1):
            m = half[sel] == h
            np.add.at(sums[h], inv[m], i_norm[sel][m])
            np.add.at(cnts[h], inv[m], 1)
        both = (cnts > 0).all(0)
        a, b = sums[0, both] / cnts[0, both], sums[1, both] / cnts[1, both]
        out[name] = {"cc": float(np.corrcoef(a, b)[0, 1]) if both.sum() > 2 else float("nan"),
                     "n_unique": int(both.sum())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--bright-frac", type=float, default=0.02)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg, dm, model, ckpt_path = load_run(args.run_dir, args.ckpt, args.device)
    r = collect(model, dm, args.device, args.batch_size, args.max_batches)

    val_ids = set(np.loadtxt(args.run_dir / "files" / "val_labels.csv", skiprows=1, dtype=np.int64).tolist())
    is_val = np.array([int(i) in val_ids for i in r["refl_ids"]])
    snr = r["intensity.sum.value"] / np.sqrt(np.clip(r["intensity.sum.variance"], 1e-12, None))
    # `n_harmonics` is absent from datasets the current extractor writes --
    # assign() already keeps one harmonic per ray, so there is nothing to
    # exceed and `collect` fills the column with NaN. Comparing NaN <= 1 is
    # False, which emptied the candidate set and made the quantile below raise
    # an opaque IndexError on every arm.
    nh = r["n_harmonics"]
    clean = (r["cos_bright_ready"] > 0) & (np.isnan(nh) | (nh <= 1))
    cand = is_val & clean
    if not cand.any():
        raise SystemExit(
            f"no candidate reflections: of {len(is_val)} collected, "
            f"{int(is_val.sum())} are validation and "
            f"{int(clean.sum())} are clean (unmasked, <=1 harmonic). "
            "Check that val_labels.csv matches this run."
        )
    thresh = np.quantile(r["intensity.sum.value"][cand], 1 - args.bright_frac)
    bright = cand & (r["intensity.sum.value"] >= thresh)

    deciles = np.quantile(snr[is_val], np.linspace(0, 1, 11))
    dec = np.clip(np.digitize(snr[is_val], deciles[1:-1]), 0, 9)
    n_eff_by_decile = [
        float(np.median(r["n_eff"][is_val][dec == k])) if (dec == k).any() else float("nan")
        for k in range(10)
    ]

    i_norm, res_bin = normalize_by_cell(r["qi_mean"], r["wavelength"], r["d"])
    hkl = np.stack([r["H"], r["K"], r["L"]], -1)
    cc_model = half_set_cc(i_norm, hkl, r["refl_ids"], res_bin)
    s_norm, _ = normalize_by_cell(r["intensity.sum.value"], r["wavelength"], r["d"])
    cc_sum = half_set_cc(s_norm, hkl, r["refl_ids"], res_bin)

    qp_state = {k: v for k, v in model.surrogates["qp"].state_dict().items() if k.startswith("log_")}
    summary = {
        "run_dir": str(args.run_dir),
        "checkpoint": str(ckpt_path),
        "profile_surrogate": cfg["surrogates"]["qp"]["name"],
        "n_reflections": int(len(r["refl_ids"])),
        "n_val": int(is_val.sum()),
        "bright_spots": {"n": int(bright.sum()), "cos_median": float(np.median(r["cos"][bright])),
                         "cos_p10": float(np.percentile(r["cos"][bright], 10))},
        "neff_vs_snr": {"spearman_val": spearman(r["n_eff"][is_val], snr[is_val]),
                        "median_n_eff_by_snr_decile": n_eff_by_decile},
        "half_set_cc_model": cc_model,
        "half_set_cc_summation": cc_sum,
        "physics": {k: v.exp().tolist() if k != "log_spectrum" else v.tolist() for k, v in qp_state.items()},
    }
    if "log_mosaic" in qp_state:
        summary["physics"]["mosaic_deg"] = float(np.degrees(qp_state["log_mosaic"].exp().item()))
    qp = model.surrogates["qp"]
    if hasattr(qp, "size_tensor"):
        # widths along the principal axes, in the crystal frame
        d_eval = torch.linalg.eigvalsh(qp.size_tensor().detach().cpu())
        m_eval = torch.linalg.eigvalsh(qp.mosaic_tensor().detach().cpu())
        summary["physics"]["size_sd_invA"] = d_eval.sqrt().tolist()
        summary["physics"]["mosaic_sd_deg"] = np.degrees(m_eval.sqrt().numpy()).tolist()

    print(f"{summary['profile_surrogate']}  ({summary['n_reflections']} refl, ckpt {ckpt_path.name})")
    print(f"  bright val spots (n={summary['bright_spots']['n']}): cos median {summary['bright_spots']['cos_median']:.3f}, "
          f"p10 {summary['bright_spots']['cos_p10']:.3f}")
    print(f"  Spearman(n_eff, I/sigma) on val: {summary['neff_vs_snr']['spearman_val']:+.3f}; "
          f"median n_eff by I/sigma decile: {[round(v, 1) for v in n_eff_by_decile]}")
    print(f"  half-set CC (model):     all {cc_model['all']['cc']:.4f} over {cc_model['all']['n_unique']} unique")
    print(f"  half-set CC (crude 21x21 box sum, baseline): all {cc_sum['all']['cc']:.4f}")
    print("  per shell (low to high resolution), model / summation:")
    print("   " + " ".join(f"{cc_model[f'shell{k:02d}']['cc']:.3f}/{cc_sum[f'shell{k:02d}']['cc']:.3f}" for k in range(20)))
    if summary["physics"]:
        print(f"  physics: {summary['physics']}")
    out = args.out or (args.run_dir / f"eval_{ckpt_path.stem}.json")
    out.write_text(json.dumps(summary, indent=1))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
