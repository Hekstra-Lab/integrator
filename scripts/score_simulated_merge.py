"""Score a merging arm against a simulated dataset's known truth.

The point of simulating is that the quantities we care about stop being
inferred.  On real data an arm is judged through careless, phenix, an R-free
flag set and a merging-seed average -- four stages that each move the number by
as much as the effect under test, and none of which can say *which* part of the
model was wrong.  Here each part is measured on its own:

    scale            eps = rms(log s_hat - log s_true) after the gauge, which
                     is the quantity the Bijvoet signal needs below ~4%
    spectrum         CC(log s_hat, true G(lambda)) and the recovered swing
    per term         how much of each true term the fitted scale recovers,
                     including the crystal-frame absorption no lab-frame
                     scale can represent
    intensity        CC(I_h_hat, I_h_true) and the regression slope
    Bijvoet          the slope of (I+ - I-)_hat on (I+ - I-)_true, which is
                     the attenuation an anomalous map would suffer
    split            CC of the attributed signal counts against the true
                     s I p, which isolates the responsibility from the scale

An arm takes minutes, so a wrong turn costs minutes.

Usage
-----
    uv run python scripts/score_simulated_merge.py --run-dir RUN --epoch 2 \\
        --data /tmp/sim_laue
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml


def gauge_free_rms(log_hat: np.ndarray, log_true: np.ndarray) -> float:
    """rms of the log difference after removing the one free constant.

    The overall level of a per-observation scale is absorbed by `I_h`, so it
    carries no information and must not be counted as error.
    """
    delta = log_hat - log_true
    return float(np.sqrt(np.mean((delta - delta.mean()) ** 2)))


def regress(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope through the origin and the correlation, the two things wanted."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 8 or x.var() == 0:
        return float("nan"), float("nan")
    slope = float((x * y).sum() / (x * x).sum())
    return slope, float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--data", type=Path, required=True,
                    help="the simulated dataset directory (for metadata.pt)")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    from integrator.cli.pred import _load_checkpoint_tolerant, _match_profile_basis_dim
    from integrator.utils.factory_utils import (
        construct_data_loader,
        construct_integrator,
    )

    cfg = yaml.safe_load((args.run_dir / "config_log.yaml").read_text())
    ckpt_path = args.run_dir / "files" / "checkpoints" / f"epoch={args.epoch:04d}.ckpt"
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path}")
    ckpt = torch.load(ckpt_path.as_posix(), map_location="cpu")
    _match_profile_basis_dim(cfg, ckpt["state_dict"])
    model = construct_integrator(cfg, skip_warmstart=True)
    _load_checkpoint_tolerant(model, ckpt["state_dict"])
    model.eval()

    truth = torch.load((args.data / "metadata.pt").as_posix(), weights_only=False)
    arr = lambda k: truth[k].numpy().astype(np.float64)

    # ---- the scale, which needs no merge at all ------------------------
    cfg["data_loader"]["args"]["batch_size"] = args.batch_size
    cfg["data_loader"]["args"]["num_workers"] = 0
    loader = construct_data_loader(cfg)
    loader.setup()

    log_hat = np.zeros(len(truth["asu_id"]))
    refl = np.zeros(len(truth["asu_id"]), dtype=np.int64)
    collected: dict[str, list] = {}
    filled = 0
    with torch.no_grad():
        for batch in loader.predict_dataloader():
            _, _, _, metadata = batch
            s = model._get_scale(metadata, torch.device("cpu")).numpy()
            ids = metadata["refl_ids"].numpy().astype(np.int64)
            log_hat[filled:filled + len(s)] = np.log(np.maximum(s, 1e-30))
            refl[filled:filled + len(s)] = ids
            filled += len(s)
            for key in ("wavelength", "d", "xyzcal.px.0", "xyzcal.px.1",
                        "image_num", "s1.0", "s1.1", "s1.2"):
                if key in metadata:
                    collected.setdefault(key, []).append(
                        metadata[key].float().cpu()
                    )
    log_hat, refl = log_hat[:filled], refl[:filled]
    term_inputs = {}
    if {"wavelength", "d", "s1.0", "image_num"} <= set(collected):
        cat = {k: torch.cat(v) for k, v in collected.items()}
        term_inputs = dict(
            wavelength=cat["wavelength"],
            d=cat["d"],
            x=cat["xyzcal.px.0"],
            y=cat["xyzcal.px.1"],
            s1=torch.stack([cat[f"s1.{k}"] for k in range(3)], dim=-1),
            image=cat["image_num"],
        )
    # the loader filters and reorders rows, so align through refl_ids rather
    # than assuming the dump and the metadata share an order
    pos = np.searchsorted(arr("refl_ids"), refl)
    log_true = np.log(arr("true_scale"))[pos]
    print(f"aligned {filled} of {len(truth['asu_id'])} observations")

    eps = gauge_free_rms(log_hat, log_true)
    print("\n-- the scale --")
    print(f"eps = rms(log s_hat - log s_true), gauge removed   {eps:.4f} "
          f"({100 * eps:.1f}%; a 2-4% Bijvoet signal needs <~4%)")
    print(f"CC(log s_hat, log s_true)                          "
          f"{np.corrcoef(log_hat, log_true)[0, 1]:+.4f}")

    # Per-term recovery needs the MODEL's own decomposition. Correlating each
    # true term against the TOTAL fitted log scale measures the total -- which
    # the spectrum dominates -- so a polarisation pinned exactly at the truth
    # came back reading CC -0.16 instead of +1.00. `log_terms` exists for this.
    print("\n-- how much of each true term the scale recovered --")
    decomposed = getattr(model.scale_fn, "log_terms", None)
    fitted_terms: dict[str, np.ndarray] = {}
    if decomposed is not None and term_inputs:
        with torch.no_grad():
            parts = decomposed(**term_inputs)
        fitted_terms = {k: v.numpy().astype(np.float64) for k, v in parts.items()}
        print("  (against the scale's own term decomposition)")
    else:
        print("  (this scale has no term decomposition, so each row is the "
              "true term against the TOTAL fitted scale -- a weak diagnostic)")
    for term in ("true_spectrum", "true_polarisation", "true_absorption",
                 "true_decay", "true_image_gray"):
        if term not in truth:
            continue
        t = arr(term)[pos]
        name = term[5:]
        got = fitted_terms.get(name)
        f = got if got is not None else log_hat
        slope, cc = regress(t - t.mean(), f - f.mean())
        swing = float(np.exp(np.ptp(np.quantile(t, [0.01, 0.99]))))
        extra = "" if got is not None else "  [vs total]"
        print(f"  {name:14s} true swing {swing:6.2f}x   "
              f"CC {cc:+.4f}   slope {slope:+.3f}{extra}")

    # ---- the merged intensities ----------------------------------------
    print("\n-- the merged intensities --")
    # finalize_merge needs every HKL complete in one batch
    model.finalize_merge(loader.predict_dataloader(grouped=True))
    qi = model.get_merged_qi()
    i_hat = qi.mean.numpy().astype(np.float64)

    asu = arr("asu_id").astype(np.int64)
    i_true_by_asu = np.zeros(int(asu.max()) + 1)
    i_true_by_asu[asu] = arr("true_intensity")
    seen = np.zeros(int(asu.max()) + 1, dtype=bool)
    seen[asu] = True
    n = min(len(i_hat), len(i_true_by_asu))
    ok = seen[:n] & np.isfinite(i_hat[:n]) & (i_true_by_asu[:n] > 0)
    slope, cc = regress(i_true_by_asu[:n][ok], i_hat[:n][ok])
    print(f"{ok.sum()} reflections   CC(I_hat, I_true) {cc:+.4f}   "
          f"slope {slope:.4f}")
    lg = regress(np.log(i_true_by_asu[:n][ok]), np.log(np.maximum(i_hat[:n][ok], 1e-30)))
    print(f"in log: CC {lg[1]:+.4f}")

    # ---- the Bijvoet difference, which is what the scale is for --------
    pool = arr("nonanom_id").astype(np.int64)
    pool_of_asu = np.zeros(int(asu.max()) + 1, dtype=np.int64)
    pool_of_asu[asu] = pool
    centric_of_asu = np.zeros(int(asu.max()) + 1, dtype=bool)
    centric_of_asu[asu] = truth["centric"].numpy()
    d_hat: dict[int, list] = {}
    d_true: dict[int, list] = {}
    for a in np.flatnonzero(ok):
        if centric_of_asu[a]:
            continue
        d_hat.setdefault(int(pool_of_asu[a]), []).append(i_hat[a])
        d_true.setdefault(int(pool_of_asu[a]), []).append(i_true_by_asu[a])
    pairs = [(h, t) for h, t in zip(d_hat.values(), d_true.values()) if len(h) == 2]
    if pairs:
        dh = np.array([h[0] - h[1] for h, _ in pairs])
        dt = np.array([t[0] - t[1] for _, t in pairs])
        slope, cc = regress(dt, dh)
        print(f"\n-- the Bijvoet difference --")
        print(f"{len(pairs)} acentric pairs   attenuation (slope) {slope:+.4f}   "
              f"CC {cc:+.4f}")
        print(f"  true |dI|/I median {np.median(np.abs(dt) / np.array([np.mean(t) for _, t in pairs])):.4f}")
    else:
        print("\nno acentric Friedel pair had both mates merged")


if __name__ == "__main__":
    main()
