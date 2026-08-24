r"""Validate that the crystal-frame SH basis can reproduce DIALS's scale.

Run AFTER extract_crystal_frame_sh.py and a DIALS scaling run, BEFORE training
PhysicalScale. It answers the de-risk question directly: can a linear model in
PhysicalScale's exact parameterization

    log s = [scale(phi) + B(phi) s^2]  +  [absorption_sh . c_lm]      s^2 = 1/(4 d^2)
            \------- smooth bulk -------/   \---- fine absorption ----/

fit DIALS's per-reflection `inverse_scale_factor`? Because PhysicalScale is
linear in log-space, the OLS R^2 here IS the ceiling for what the trained model
can represent. The number that matters is the DETRENDED R^2: remove the smooth
bulk, then see how much of the residual fine scale the SH part explains. The old
MLP got r_detrended ~ 0 (it lacked the crystal-frame direction); if the SH part
gets it well up, the fine, anomalous-gating structure is in reach and the build
is de-risked. If it stays ~0, the coordinate/convention is wrong -- revisit the
extractor before training.

Alignment is by refl_ids when both files carry them, else by rounded xyzcal.px.

    python scripts/validate_sh_basis.py \
        --scaled-refl /n/.../scaled.refl \
        --metadata    /n/.../pytorch_data/metadata_sh.pt \
        --phi-degree 6 --decay-degree 2
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _cheb(t: np.ndarray, degree: int) -> np.ndarray:
    """Chebyshev design matrix T_0..T_degree at t in [-1, 1]. Returns (N, deg+1)."""
    cols = [np.ones_like(t), t]
    for k in range(2, degree + 1):
        cols.append(2.0 * t * cols[-1] - cols[-2])
    return np.stack(cols[: degree + 1], axis=-1)


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-30)


def _fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """OLS via lstsq; returns (prediction, R^2)."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    return yhat, _r2(y, yhat)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scaled-refl", required=True, help="DIALS scaled .refl")
    ap.add_argument("--metadata", required=True, help="metadata_sh.pt (has absorption_sh)")
    ap.add_argument("--phi-degree", type=int, default=6)
    ap.add_argument("--decay-degree", type=int, default=2)
    args = ap.parse_args()

    import reciprocalspaceship as rs

    meta = torch.load(args.metadata, weights_only=False)
    if "absorption_sh" not in meta:
        raise KeyError("metadata has no 'absorption_sh'; run extract_crystal_frame_sh.py")
    a = meta["absorption_sh"].numpy().astype(np.float64)
    d = meta["d"].numpy().astype(np.float64)
    phi = meta["xyzcal.mm.2"].numpy().astype(np.float64)
    px = np.stack(
        [meta["xyzcal.px.0"].numpy(), meta["xyzcal.px.1"].numpy(),
         meta["xyzcal.px.2"].numpy()], axis=-1,
    ).astype(np.float64)
    refl_ids = meta["refl_ids"].numpy() if "refl_ids" in meta else None
    n_meta = a.shape[0]

    ds = rs.io.read_dials_stills(
        args.scaled_refl, extra_cols=["inverse_scale_factor"]
    )
    if "inverse_scale_factor" not in ds:
        raise KeyError(
            "scaled .refl has no 'inverse_scale_factor' -- is this the dials.scale "
            "output? (need the scaled, not the integrated, reflections)"
        )
    inv = ds["inverse_scale_factor"].to_numpy().astype(np.float64)
    # read_dials_stills flattens vector columns: xyzcal.px -> xyzcal.px.{0,1,2}.
    for c in ("xyzcal.px.0", "xyzcal.px.1", "xyzcal.px.2"):
        if c not in ds:
            raise KeyError(f"scaled .refl missing {c!r} (columns: {list(ds)[:25]})")

    # --- align DIALS rows to the metadata rows ---
    if refl_ids is not None and "refl_ids" in ds:
        d_ids = ds["refl_ids"].to_numpy()
        pos = {int(r): i for i, r in enumerate(d_ids)}
        sel = np.array([pos.get(int(r), -1) for r in refl_ids])
        keep = sel >= 0
        idx_dials = sel[keep]
        idx_meta = np.where(keep)[0]
        how = "refl_ids"
    else:
        # rounded-pixel match (x, y, frame) -> metadata row. xyzcal.px is set at
        # integration and unchanged by scaling, so it matches exactly across the
        # two files; rounding only guards float formatting. (x,y,z) is ~unique.
        dpx = np.stack(
            [ds["xyzcal.px.0"].to_numpy(), ds["xyzcal.px.1"].to_numpy(),
             ds["xyzcal.px.2"].to_numpy()], axis=-1,
        ).astype(np.float64)
        key_meta = {
            (round(p[0], 1), round(p[1], 1), round(p[2], 1)): i
            for i, p in enumerate(px)
        }
        sel = np.array(
            [key_meta.get((round(p[0], 1), round(p[1], 1), round(p[2], 1)), -1)
             for p in dpx]
        )
        keep = sel >= 0
        idx_dials = np.where(keep)[0]
        idx_meta = sel[keep]
        how = "rounded xyzcal.px"
    logger.info(
        "Matched %d / %d metadata rows to DIALS (by %s)",
        len(idx_meta), n_meta, how,
    )
    if len(idx_meta) < 100:
        raise RuntimeError("Too few matches; check that the files correspond.")

    inv = inv[idx_dials]
    a, d, phi = a[idx_meta], d[idx_meta], phi[idx_meta]
    px = px[idx_meta]
    inv = np.clip(inv, 1e-6, None)
    y = np.log(inv)

    # --- design matrices mirroring PhysicalScale ---
    t = np.clip(2.0 * (phi - phi.min()) / (np.ptp(phi) + 1e-12) - 1.0, -1, 1)
    s_sq = 1.0 / (4.0 * np.clip(d, 1e-3, None) ** 2)
    cheb_scale = _cheb(t, args.phi_degree)                 # scale(phi)
    cheb_decay = _cheb(t, args.decay_degree) * s_sq[:, None]  # B(phi) s^2
    smooth = np.concatenate([cheb_scale, cheb_decay], axis=1)
    full = np.concatenate([smooth, a], axis=1)

    _, r2_smooth = _fit(smooth, y)
    _, r2_full = _fit(full, y)

    # detrended: residual after the smooth bulk, explained by the SH part
    resid_hat, _ = _fit(smooth, y)
    r = y - resid_hat
    a_aug = np.concatenate([np.ones((len(r), 1)), a], axis=1)
    _, r2_detrended = _fit(a_aug, r)

    # contrast: the old MLP's inputs (frame, x, y, d) on the same residual
    mlp_feats = np.concatenate(
        [np.ones((len(r), 1)), px / px.max(0, keepdims=True),
         s_sq[:, None] / s_sq.max()], axis=1,
    )
    _, r2_mlp_detrended = _fit(mlp_feats, r)

    print("\n=== SH-basis validation (log inverse_scale_factor) ===")
    print(f"  n matched                : {len(y)}")
    print(f"  R^2 smooth (scale+decay) : {r2_smooth:.4f}")
    print(f"  R^2 full  (+ SH absorp.) : {r2_full:.4f}   (PhysicalScale ceiling)")
    print(f"  R^2 detrended  -> SH     : {r2_detrended:.4f}   (<-- the key number)")
    print(f"  R^2 detrended  -> px/d   : {r2_mlp_detrended:.4f}   (MLP-style ceiling)")
    if r2_detrended > 0.3 and r2_detrended > 3 * max(r2_mlp_detrended, 1e-6):
        print("\n  => SH explains the fine scale the MLP could not. Build de-risked.")
    elif r2_detrended < 0.1:
        print("\n  => SH does NOT explain the fine scale. Check the extractor's "
              "rotation convention (axis/transpose) before training.")
    else:
        print("\n  => Partial. SH helps but not decisively; consider higher lmax.")


if __name__ == "__main__":
    main()
