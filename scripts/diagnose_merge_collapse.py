"""Quick diagnostic: is the merged intensity collapsed, and where?

Runs ONE grouped batch through the amortized-merging integrator and reports the
spread of the per-HKL merged intensity I_h = alpha_h/beta_h and its parts, next
to the DIALS intensity (the data). Localizes the collapse:

  - DIALS I has spread but I_h is flat  -> the merge is washing out structure.
  - alpha_h ~ beta_h (high corr) and both spread -> signal tracks exposure
    (the scale is absorbing the intensity) -> I_h = a/b flat.
  - alpha_h AND beta_h both flat        -> the per-obs potentials are constant.
  - log-CC(I_h, per-HKL DIALS) near 0   -> the merge isn't tracking the data.

Runs in eval AND train mode: if eval is flat but train is spread (or vice versa),
the bug is a train/finalize divergence (dropout, a buffer, eval-mode scale), not
the merge math.

Usage: uv run python scripts/diagnose_merge_collapse.py RUN_DIR [--checkpoint X]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _stats(name: str, x) -> None:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        print(f"  {name:20s} (empty)")
        return
    cv = float(x.std() / max(abs(x.mean()), 1e-12))
    print(
        f"  {name:20s} n={x.size:6d}  min={x.min():9.4g}  q25={np.percentile(x,25):9.4g}"
        f"  med={np.median(x):9.4g}  q75={np.percentile(x,75):9.4g}  max={x.max():9.4g}"
        f"  CV={cv:6.3f}"
    )


def _scatter_mean(vals: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
    num = torch.zeros(n, dtype=vals.dtype).scatter_add_(0, idx, vals)
    cnt = torch.zeros(n, dtype=vals.dtype).scatter_add_(0, idx, torch.ones_like(vals))
    return num / cnt.clamp(min=1.0)


def _load(run_dir: Path, checkpoint: Path | None):
    meta = yaml.safe_load((run_dir / "run_paths.yaml").read_text())
    cfg = yaml.safe_load(Path(meta["config"]).read_text())
    if checkpoint is None:
        ckpt_dir = Path(meta["wandb"]["log_dir"]) / "checkpoints"
        checkpoint = ckpt_dir / "last.ckpt"
        if not checkpoint.exists():
            cands = sorted(ckpt_dir.glob("epoch=*.ckpt")) or sorted(
                ckpt_dir.glob("*.ckpt")
            )
            checkpoint = cands[-1]
    log.info("config:     %s", meta["config"])
    log.info("checkpoint: %s", checkpoint)

    from integrator.utils.factory_utils import construct_integrator

    integ = construct_integrator(cfg, skip_warmstart=True)
    state = torch.load(checkpoint, weights_only=False, map_location="cpu")[
        "state_dict"
    ]
    ms = integ.state_dict()
    compat = {
        k: v for k, v in state.items() if k in ms and v.shape == ms[k].shape
    }
    integ.load_state_dict(compat, strict=False)
    return integ, cfg


def _forward_report(integ, batch, mode: str) -> None:
    counts, shoebox, mask, mt = batch
    integ.train(mode == "train")
    with torch.no_grad():
        out = integ(counts, shoebox, mask, mt)

    qih = out["qi_h"]
    alpha = qih.concentration.detach()
    beta = qih.rate.detach()
    Ih = qih.mean.detach()  # alpha / beta
    inverse = out["inverse"].detach()
    unique = out["unique_hkls"].detach()
    scale = out["scale"].detach()
    tau = out["tau_h"].detach()
    n_uni = len(unique)

    print(f"\n=== {mode.upper()} forward: {n_uni} HKLs, {counts.shape[0]} obs ===")
    _stats("scale (per obs)", scale.numpy())
    _stats("I_h = alpha/beta", Ih.numpy())
    _stats("alpha_h (shape)", alpha.numpy())
    _stats("beta_h (rate)", beta.numpy())
    _stats("sum d_alpha", (alpha - integ.alpha_W).numpy())
    _stats("sum d_beta", (beta - tau).numpy())

    # do alpha and beta move together (-> I_h flat)?
    a, b = alpha.numpy(), beta.numpy()
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if ok.sum() > 10:
        r_ab = float(np.corrcoef(np.log(a[ok]), np.log(b[ok]))[0, 1])
        print(f"  corr(log alpha_h, log beta_h) = {r_ab:.3f}  (->1 means I_h flat)")

    # does I_h track the DIALS data per HKL?
    dials_key = (
        "intensity.sum.value"
        if "intensity.sum.value" in mt
        else "intensity.prf.value"
    )
    if dials_key in mt:
        dials = mt[dials_key].float()
        dials_hkl = _scatter_mean(dials, inverse, n_uni)
        # De-scaled by the MODEL scale -> the model's own F^2 estimate from the
        # data. If raw DIALS is spread but this is FLAT, the scale has absorbed
        # F^2 (the gauge collapse): I_h should track this, not raw DIALS.
        descaled_hkl = _scatter_mean(dials / scale.clamp(min=1e-12), inverse, n_uni)
        _stats("DIALS/HKL (raw)", dials_hkl.numpy())
        _stats("DIALS/HKL (descaled)", descaled_hkl.numpy())
        m = (Ih > 0) & (dials_hkl > 0)
        if int(m.sum()) > 10:
            cc = float(
                np.corrcoef(
                    np.log(Ih[m].numpy()), np.log(dials_hkl[m].numpy())
                )[0, 1]
            )
            print(
                f"  log-CC(I_h, per-HKL DIALS[{dials_key}]) = {cc:.3f}  "
                f"(n={int(m.sum())})"
            )
        md = (Ih > 0) & (descaled_hkl > 0)
        if int(md.sum()) > 10:
            ccd = float(
                np.corrcoef(
                    np.log(Ih[md].numpy()), np.log(descaled_hkl[md].numpy())
                )[0, 1]
            )
            print(f"  log-CC(I_h, de-scaled DIALS)         = {ccd:.3f}")

    # --- background vs signal: is the bg absorbing per-reflection brightness? ---
    # Reproduces the wandb scatter (pred signal s*I_h vs total counts) WITH the
    # background it competes against. If pred-signal is flat but total counts are
    # spread AND the bg carries most of the prediction, the bg ate the signal.
    qbg = out["qbg"]
    bg_pp = qbg.mean.detach().float()  # per-obs background per pixel
    if bg_pp.ndim > 1:
        bg_pp = bg_pp.reshape(bg_pp.shape[0], -1).mean(-1)
    npix = mask.reshape(mask.shape[0], -1).float().sum(-1)
    tot_bg = bg_pp * npix
    tot_sig = scale * Ih[inverse]  # profile integrates to ~1
    tot_cnt = counts.reshape(counts.shape[0], -1).clamp(min=0).float().sum(-1)
    print("  -- prediction budget (per obs) --")
    _stats("total counts", tot_cnt.numpy())
    _stats("pred signal s*I_h", tot_sig.numpy())
    _stats("pred bg pp*npix", tot_bg.numpy())
    frac_bg = (tot_bg / (tot_bg + tot_sig).clamp(min=1e-12)).clamp(0, 1)
    print(f"  median bg fraction of prediction = {float(frac_bg.median()):.3f}")
    ok2 = (tot_sig > 0) & (tot_cnt > 0)
    if int(ok2.sum()) > 10:
        rsc = float(
            np.corrcoef(
                np.log(tot_sig[ok2].numpy()), np.log(tot_cnt[ok2].numpy())
            )[0, 1]
        )
        print(f"  log-CC(pred signal s*I_h, total counts) = {rsc:.3f}  (the wandb CC)")

    # --- profile peakedness: the decisive check ---
    # A uniform profile (max/mean ~ 1, top-9 mass ~ 9/npix) makes scale*I_h*prf
    # a FLAT field that mimics the Laue background, so the intensity fits the
    # background instead of the peak. A peaked profile concentrates mass on the
    # reflection (top-9 mass -> ~1).
    prf = out["qp"].mean_profile.detach().float()
    prf = prf / prf.sum(-1, keepdim=True).clamp(min=1e-12)
    npx = prf.shape[-1]
    _stats("profile max/mean", (prf.max(-1).values * npx).numpy())
    _stats("profile top9 mass", torch.topk(prf, min(9, npx), -1).values.sum(-1).numpy())
    pp = npix.clamp(min=1)
    print(
        f"  per-pixel: bg={float((tot_bg / pp).median()):.3f}  "
        f"signal={float((tot_sig / pp).median()):.3f}  "
        f"counts={float((tot_cnt / pp).median()):.3f}  "
        f"(uniform profile spreads signal to ~counts/pixel = background level)"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--checkpoint", type=Path, default=None)
    args = ap.parse_args()

    integ, cfg = _load(args.run_dir.resolve(), args.checkpoint)

    from integrator.utils.factory_utils import construct_data_loader

    dl = construct_data_loader(cfg)
    dl.setup()
    batch = next(iter(dl.train_dataloader()))

    counts_b, shoebox_b, mask_b, mt = batch

    # --- Empirical reflection shape from the DATA (model-independent) ---
    # Per-shoebox background = median of pixels (robust to the small peak), then
    # average the background-subtracted, masked boxes. A SHARP peak at the box
    # CENTER means the extraction centers the spot, so a centered peaked basis is
    # right. A diffuse or off-center average means the box is not centered on the
    # reflection -- a centered basis cannot fit it and the profile stays uniform.
    cnt = counts_b.reshape(counts_b.shape[0], -1).clamp(min=0).float()
    mk = mask_b.reshape(mask_b.shape[0], -1).float()
    bg_box = cnt.median(dim=-1, keepdim=True).values
    refl = ((cnt - bg_box).clamp(min=0) * mk).mean(0)
    refl = refl / refl.sum().clamp(min=1e-12)
    npx = refl.numel()
    side = int(round(npx**0.5))
    peak = int(refl.argmax())
    top9 = float(torch.topk(refl, min(9, npx)).values.sum())
    centered = divmod(peak, side) == (side // 2, side // 2)
    print("\n=== empirical reflection shape (DATA, bg-subtracted mean) ===")
    print(
        f"  peak (row,col) = {divmod(peak, side)}   box center = "
        f"{(side // 2, side // 2)}   {'CENTERED' if centered else '** OFF-CENTER **'}"
    )
    print(
        f"  top9 mass = {top9:.3f}  (uniform = {9 / npx:.3f})  -> "
        f"{'peaked' if top9 > 0.10 else '** DIFFUSE: box too big or spot off-center **'}"
    )
    # effective spot sigma (RMS radius) -> what hermite_basis_sigma should be.
    ys, xs = torch.meshgrid(
        torch.arange(side).float(), torch.arange(side).float(), indexing="ij"
    )
    ys, xs = ys.reshape(-1), xs.reshape(-1)
    cy, cx = float((refl * ys).sum()), float((refl * xs).sum())
    var = float((refl * ((ys - cy) ** 2 + (xs - cx) ** 2)).sum())
    sigma_eff = (0.5 * var) ** 0.5  # 2D: E[r^2] = 2 sigma^2
    print(
        f"  centroid=({cy:.1f},{cx:.1f})  effective sigma ~ {sigma_eff:.2f} px  "
        f"-> set hermite_basis_sigma near this (current basis sigma=3 spreads "
        f"~85% of mass off the central 9 px)"
    )
    # anscombe-standardized shoebox sanity: should be ~N(0,1) globally if the
    # anscombe_stats match this data; wildly off means the encoder sees garbage.
    sb = shoebox_b.reshape(shoebox_b.shape[0], -1).float()
    print(
        f"  standardized shoebox: mean={float(sb.mean()):.3f} std={float(sb.std()):.3f}"
        f"  (expect ~0 / ~1 if anscombe_stats are correct)"
    )

    dials_key = (
        "intensity.sum.value"
        if "intensity.sum.value" in mt
        else "intensity.prf.value"
    )
    print(f"\n=== DIALS intensity ({dials_key}, the DATA, per obs) ===")
    if dials_key in mt:
        _stats(dials_key, mt[dials_key].float().numpy())
    print("  (this is the spread the merge should reproduce in I_h)")

    _forward_report(integ, batch, "eval")
    _forward_report(integ, batch, "train")

    print(
        "\nReading: if DIALS I has spread but I_h is flat (low CV, log-CC~0) "
        "the merge is washing it out. If corr(alpha,beta)~1 the scale is "
        "absorbing the signal. If eval and train disagree it's a "
        "train/finalize divergence, not the merge math.\n"
    )


if __name__ == "__main__":
    main()
