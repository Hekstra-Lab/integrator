"""Score a profile on pixels it was not allowed to see.

Every other comparison in this project reaches the profile through the
merge: careless scales and merges, phenix refines, and an anomalous peak
or an R-free comes out the far end. Three stages sit between the profile
and the number, and the first of them weights each measurement by a sigma
that is a deterministic function of its own mean, so a better profile has
a narrow channel through which to show itself.

This is a direct test. The model asserts a rate per pixel,

    lambda_i = E[I] p_i + E[B],

so a profile that puts its mass where the photons are predicts counts
better than one that does not, and that can be scored WITHOUT merging
anything. A random share of the pixels in every shoebox is held out, the
model sees only the rest, and the held-out counts are scored under the
rate it predicts from the rest. It is a proper scoring rule, it is local
to one shoebox, and no scale, no R-free and no sigma enter it.

**The held-out pixels are hidden from the encoder, not only from the
fit.** `SVAEIntegrator` forms its encoder input as `shoebox * mask`
(`svae_integrator.py`), so zeroing the mask at a pixel removes it from
the profile encoder's view as well as from the conjugate sums that make
`q(I)` and `q(B)`. Without that the profile would have been computed from
the very counts it is then scored on, and the comparison would measure
how well each encoder memorises rather than how well it predicts. Held-out
pixels then look to the model exactly like the invalid pixels it was
trained to cope with, which is the closest thing to "absent" the
architecture has.

**The same pixels are held out for every arm**, drawn from a generator
seeded by the reflection id, so two arms are scored on identical pixels of
identical reflections. A per-batch draw would not do: the loaders agree
across arms but the comparison should not depend on that.

Reported per arm, overall and per resolution third:

- `logp_px`   mean Poisson log-likelihood per held-out pixel,
              `c log lambda - lambda - log c!`. Higher is better. The
              `log c!` term is the same for every arm and is kept only so
              the number is a log-likelihood rather than a ranking score.
- `chi2_px`   mean `(c - lambda)^2 / lambda` over held-out pixels. 1 is
              Poisson-consistent; above 1 the predicted rate is wrong in
              shape or scale, below 1 it is over-smooth.
- `bias_px`   mean `(c - lambda)` over held-out pixels, in counts. A
              profile too narrow leaves positive residual in the wings.
- `n_px`      held-out pixels scored.

Usage, through the generic reducer job:

    DATASET=hewl_1118 N_BATCHES=40 HOLDOUT=0.2 \\
      SCRIPT=$REPO/scripts/figures/dump_heldout_score.py \\
      OUT=$PWD/heldout_1118.csv \\
      sbatch scripts/figures/encoder_health.sbatch <run-dir>...

`REFL_IDS=<file>` restricts the scoring to named reflections, one id per
line, which is how to score only reflections that were held out of
training in EVERY arm -- the train/val split is drawn per run, so without
it one arm may be scored on reflections it trained on and another on
reflections it did not.
"""

from __future__ import annotations

import os
import sys

# Python puts a script's own directory first on sys.path, but this is also
# imported as a module by the tests, so the sibling import is made explicit.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATASET = os.environ.get("DATASET", "")
# tile size for the hold-out; 0 is the old scattered-pixel draw
BLOCK = int(os.environ.get("HOLDOUT_BLOCK", "0"))
# align the tiling to each reflection's own profile peak, so the bright
# end is sampled at all
PEAK_ALIGNED = os.environ.get("HOLDOUT_PEAK", "1") not in ("0", "false", "")
N_BATCHES = int(os.environ.get("N_BATCHES", "40"))
HOLDOUT = float(os.environ.get("HOLDOUT", "0.2"))
MIN_KEPT = int(os.environ.get("MIN_KEPT", "40"))

_ids_file = os.environ.get("REFL_IDS", "")


def wanted_ids():
    """The reflection ids to score, or None for every reflection seen."""
    if not _ids_file:
        return None
    import numpy as np

    if _ids_file.endswith(".npy"):
        return set(int(v) for v in np.load(_ids_file).ravel())
    with open(_ids_file) as fh:
        return set(int(line) for line in fh if line.strip())


def holdout_mask(refl_ids, n_px, frac, device, block: int = 0):
    """A boolean hold-out mask per reflection, keyed by its OWN id.

    Seeding from the id rather than from the batch makes the draw a
    property of the reflection, so every arm holds out the same pixels of
    the same reflection however the loaders order them.

    `block > 0` holds out contiguous `block x block` tiles instead of
    scattered pixels, and `scored_centres` then keeps only each tile's
    centre. Scattered pixels LEAK: the shoebox autocorrelation is 0.32 at
    lag 1, so a held-out pixel with four kept neighbours is partly
    predicted by them and its residual is too small, which biases every
    variance measured from it downwards. A 5x5 tile's centre has every
    neighbour out to lag 2 held out with it, and the correlation at lag 3
    is 0.0155.
    """
    import numpy as np
    import torch

    out = np.zeros((len(refl_ids), n_px), dtype=bool)
    side = int(round(n_px ** 0.5))
    for k, rid in enumerate(refl_ids):
        rng = np.random.default_rng(int(rid) & 0x7FFFFFFF)
        if block and side * side == n_px:
            nb = side // block
            take = rng.random(nb * nb) < frac
            grid = np.zeros((side, side), bool)
            for bi in range(nb):
                for bj in range(nb):
                    if take[bi * nb + bj]:
                        grid[bi * block:(bi + 1) * block,
                             bj * block:(bj + 1) * block] = True
            out[k] = grid.ravel()
        else:
            out[k] = rng.random(n_px) < frac
    return torch.from_numpy(out).to(device)


def centre_mask(n_px: int, block: int, device, peak=None):
    """The centre pixel of every `block x block` tile.

    `peak` shifts the tiling per reflection so that one centre lands on
    that pixel. Without it the centres sit on a fixed grid and the spot,
    which occupies a few pixels out of 625, almost never coincides with
    one: measured, the scored rates stopped at 15.7 ADU with 102 pixels
    in the top bin, so the whole bright end -- the only part of the
    variance function that was unmeasured -- went unsampled.
    """
    import numpy as np
    import torch

    side = int(round(n_px ** 0.5))
    half = block // 2
    if peak is None:
        grid = np.zeros((side, side), bool)
        for bi in range(side // block):
            for bj in range(side // block):
                grid[bi * block + half, bj * block + half] = True
        return torch.from_numpy(grid.ravel()).to(device)

    pr = (peak // side).long()
    pc = (peak % side).long()
    rows = torch.arange(side, device=peak.device)
    # a centre wherever (row - peak_row) is a multiple of `block`
    r_ok = ((rows[None, :] - pr[:, None]) % block == 0)
    c_ok = ((rows[None, :] - pc[:, None]) % block == 0)
    return (r_ok[:, :, None] & c_ok[:, None, :]).reshape(len(peak), -1)


def tile_mask_at(peak, n_px: int, block: int, frac: float, refl_ids):
    """Hold out tiles on a grid aligned to each reflection's own peak.

    The tile holding the peak is ALWAYS held out, so every reflection
    contributes one bright sample; the rest are drawn at `frac`.
    """
    import numpy as np
    import torch

    side = int(round(n_px ** 0.5))
    half = block // 2
    pr = (peak // side).detach().cpu().numpy()
    pc = (peak % side).detach().cpu().numpy()
    out = np.zeros((len(refl_ids), n_px), bool)
    for k, rid in enumerate(refl_ids):
        rng = np.random.default_rng(int(rid) & 0x7FFFFFFF)
        r0 = (pr[k] - half) % block
        c0 = (pc[k] - half) % block
        grid = np.zeros((side, side), bool)
        nb = side // block + 1
        for bi in range(-1, nb):
            for bj in range(-1, nb):
                r, c = r0 + bi * block, c0 + bj * block
                if r < -block or c < -block or r >= side or c >= side:
                    continue
                is_peak = (r <= pr[k] < r + block) and (c <= pc[k] < c + block)
                if is_peak or rng.random() < frac:
                    grid[max(r, 0):r + block, max(c, 0):c + block] = True
        out[k] = grid.ravel()
    return torch.from_numpy(out).to(peak.device)


def last_checkpoint(run: str):
    """The highest-epoch checkpoint under a run directory."""
    import glob
    import re

    found = glob.glob(f"{run}/**/checkpoints/epoch*.ckpt", recursive=True)
    if not found:
        found = glob.glob(f"{run}/**/*.ckpt", recursive=True)
    if not found:
        return None

    def epoch_of(path: str) -> int:
        m = re.search(r"epoch[=_]?(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else -1

    return sorted(found, key=epoch_of)[-1]


def score(model, loader, device, n_batches, want):
    """Mean held-out score per reflection, with resolution alongside."""
    import numpy as np
    import torch

    from dump_profile_maps import ensure_group_label

    logp, chi2, bias, npx, dd = [], [], [], [], []
    # per held-out PIXEL, for the two diagnostics that need the pixel and
    # not the reflection: the residual against the profile value it sits
    # on, and the squared residual against the rate
    px_p, px_lam, px_res, px_ldec = [], [], [], []
    # the per-reflection pedestal and where on the detector it came from,
    # so its structure can be seen rather than assumed
    ped, pos_x, pos_y, img_id = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            counts, shoebox, mask, metadata = batch
            counts = counts.to(device)
            shoebox = shoebox.to(device)
            mask = mask.to(device)
            metadata = {k: (v.to(device) if hasattr(v, "to") else v)
                        for k, v in metadata.items()}
            metadata, note = ensure_group_label(metadata, device)
            if note and i == 0:
                print(f"    {note}")

            rid = metadata.get("refl_ids")
            if rid is None:
                raise SystemExit(
                    "the loader carries no `refl_ids`; the hold-out has to "
                    "be keyed by the reflection so the arms agree"
                )
            rid_np = rid.detach().cpu().numpy().astype("int64")
            keep_rows = (np.isin(rid_np, list(want)) if want
                         else np.ones(len(rid_np), bool))
            if not keep_rows.any():
                continue

            m2 = mask.squeeze(-1) if mask.dim() == 3 else mask
            n_px = m2.shape[-1]
            pr_peak = prof.reshape(len(rid_np), -1).argmax(-1)
            if BLOCK and PEAK_ALIGNED:
                held = tile_mask_at(pr_peak, n_px, BLOCK, HOLDOUT, rid_np)
                ctr = centre_mask(n_px, BLOCK, device, peak=pr_peak)
            else:
                held = holdout_mask(rid_np, n_px, HOLDOUT, device, BLOCK)
                ctr = (centre_mask(n_px, BLOCK, device).unsqueeze(0)
                       if BLOCK else None)
            # only pixels that were valid to begin with can be held out
            held = held & m2.bool()
            # with tiles, only the centres are SCORED; the rest of the
            # tile is removed so the centre has no kept neighbour, but a
            # residual is taken from the centre alone
            scored = (held & ctr) if ctr is not None else held
            reduced = (m2.bool() & ~held).to(m2.dtype)
            if mask.dim() == 3:
                reduced_in = reduced.unsqueeze(-1)
            else:
                reduced_in = reduced

            out = model(counts, shoebox, reduced_in, metadata)
            fo = out["forward_out"] if "forward_out" in out else out
            prof = fo["qp_mean"] if "qp_mean" in fo else fo.get("profile")
            if prof is None:
                raise SystemExit("the forward returned no profile")
            i_mean = fo["qi_mean"].reshape(-1, 1)
            b_mean = fo["qbg_mean"].reshape(-1, 1)
            lam = (i_mean * prof.reshape(len(rid_np), -1)
                   + b_mean).clamp_min(1e-9)

            c = counts.reshape(len(rid_np), -1).float()
            h = scored.reshape(len(rid_np), -1)
            # rows with too little left to fit from are not scored: with
            # most of a box gone, `q(I)` is the prior and the comparison
            # would be between priors
            enough = reduced.sum(-1) >= MIN_KEPT
            rows = torch.from_numpy(keep_rows).to(device) & enough & (h.sum(-1) > 0)
            if not bool(rows.any()):
                continue

            ll = (c * lam.log() - lam - torch.lgamma(c + 1.0)) * h
            resid = (c - lam) * h
            pearson = ((c - lam) ** 2 / lam) * h
            n = h.sum(-1).clamp_min(1)

            sel = rows
            # thinned to keep the arrays small: the pixel diagnostics
            # want 10^6 points, not 10^8
            keep = sel.detach().cpu().numpy()
            pr = prof.reshape(len(rid_np), -1)
            # Binned by RANK WITHIN THE SHOEBOX, not by a global quantile.
            # A profile is peaked enough that over 90% of its pixels sit at
            # p/pmax ~ 0, so global quantile edges are all zero and every
            # pixel lands in one bin; and a shoebox is background enough
            # that global rate deciles span 10.1 to 11.6 ADU, a lever arm
            # too short to fit a line to. A within-shoebox decile puts 10%
            # of EVERY shoebox in each bin, so decile 9 is its peak and
            # decile 0 its background, whatever the reflection is worth.
            npx_row = pr.shape[-1]
            dec_p = (pr.argsort(-1).argsort(-1).float()
                     / npx_row * 10).floor().clamp(0, 9)
            dec_l = (lam.argsort(-1).argsort(-1).float()
                     / npx_row * 10).floor().clamp(0, 9)
            hb = h.detach().cpu().numpy()[keep]
            px_p.append(dec_p.detach().cpu().numpy()[keep][hb][::7])
            px_lam.append(lam.detach().cpu().numpy()[keep][hb][::7])
            px_ldec.append(dec_l.detach().cpu().numpy()[keep][hb][::7])
            px_res.append((c - lam).detach().cpu().numpy()[keep][hb][::7])

            # c0 = mean - var over the half of the valid pixels with the
            # lowest profile, which is this shoebox's own background. No
            # detector constant and no bias frame: for a background pixel
            # `mean = c0 + N` and `var = N`.
            pb = torch.where(reduced.bool(), pr,
                             torch.full_like(pr, float("inf")))
            thr = pb.quantile(0.5, dim=-1, keepdim=True)
            bsel = ((pb <= thr) & reduced.bool()).to(c.dtype)
            nb = bsel.sum(-1, keepdim=True).clamp_min(2.0)
            mu_b = (c * bsel).sum(-1, keepdim=True) / nb
            var_b = (((c - mu_b) ** 2) * bsel).sum(-1, keepdim=True) / (nb - 1)
            ped.append((mu_b - var_b).reshape(-1)[sel].detach().cpu().numpy())
            for key, store in (("xyzcal.px.0", pos_x), ("xyzcal.px.1", pos_y),
                               ("BATCH", img_id)):
                v = metadata.get(key)
                store.append(v.detach().float().cpu().numpy()[keep]
                             if v is not None
                             else np.full(int(sel.sum().item()), np.nan))

            logp.append((ll.sum(-1) / n)[sel].detach().cpu().numpy())
            chi2.append((pearson.sum(-1) / n)[sel].detach().cpu().numpy())
            bias.append((resid.sum(-1) / n)[sel].detach().cpu().numpy())
            npx.append(h.sum(-1)[sel].detach().cpu().numpy())
            d = metadata.get("d")
            dd.append(d.detach().float().cpu().numpy()[sel.cpu().numpy()]
                      if d is not None
                      else np.full(int(sel.sum().item()), np.nan))
    if not logp:
        return None
    return (tuple(np.concatenate(v) for v in (logp, chi2, bias, npx, dd))
            + (np.concatenate(px_p), np.concatenate(px_lam),
               np.concatenate(px_res), np.concatenate(px_ldec),
               np.concatenate(ped),
               np.concatenate(pos_x), np.concatenate(pos_y),
               np.concatenate(img_id)))


def pedestal_structure(ped, x, y, img, d, n_bins: int = 6) -> list[dict]:
    """Does the pedestal vary, and with what?

    A multi-panel detector reads each module through its own electronics,
    so the offset can differ by panel; it also drifts with temperature
    over a run, and it has no reason to depend on the reflection. Binning
    `c0` by detector x, detector y, image and resolution says which of
    those is true here. If it is flat in all four, one constant would do;
    if it steps with x or y, the panels differ; if it drifts with image,
    a per-image value is needed. The per-reflection estimator is correct
    under all of them, so this is for understanding and not for a fix.
    """
    import numpy as np

    out = []
    for name, v in (("detector x", x), ("detector y", y),
                    ("image", img), ("resolution", d)):
        if v is None or not np.isfinite(v).any():
            continue
        good = np.isfinite(v) & np.isfinite(ped)
        if good.sum() < 200:
            continue
        vv, pp = v[good], ped[good]
        edges = np.quantile(vv, np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-9
        idx = np.clip(np.digitize(vv, edges) - 1, 0, n_bins - 1)
        for k in range(n_bins):
            m = idx == k
            if m.sum() < 30:
                continue
            out.append({"against": name, "bin": k,
                        "value": float(np.median(vv[m])),
                        "n": int(m.sum()),
                        "pedestal": float(np.median(pp[m])),
                        "spread": float(np.std(pp[m]))})
    return out


def shape_of_misfit(pk, res, n_bins: int = 10) -> list[dict]:
    """The held-out residual against the profile value it sits on.

    This separates three explanations of a rate that does not fit. If the
    model's profile is too BROAD, the peak is under-predicted and the
    wings over-predicted, so the residual falls with `p`. If the INTENSITY
    is simply too low the residual is positive everywhere and scales with
    `p`. If the BACKGROUND is over-subtracted the residual is positive and
    flat, largest where `p` is smallest. A table of the residual by
    profile decile tells them apart; one number cannot.

    `pk` is the pixel's profile value divided by the largest in its own
    shoebox, so a decile means the same thing in a bright reflection and
    a weak one.
    """
    import numpy as np

    rows = []
    for k in range(10):
        m = pk == k
        if m.sum() < 50:
            continue
        rows.append({"bin": k, "profile_decile": k, "n": int(m.sum()),
                     "resid": float(np.mean(res[m])),
                     "resid_med": float(np.median(res[m]))})
    return rows


def noise_model(lam, res, ldec=None, n_bins: int = 12) -> dict:
    """Fit `Var(c) = a * rate + b` from the held-out residuals.

    Poisson is `a = 1, b = 0`. A detector that stores `g` units per photon
    with read noise `sigma_r` gives `a = g` and `b = sigma_r^2`, and
    `a < 1` is what a sub-Poisson chi-square means: the stored value is
    not a photon count.

    Fitted on the MEDIAN squared residual per rate decile rather than the
    mean, because the bright end carries rate misfit that is not noise and
    would otherwise set the slope. The median of a squared residual is
    0.4549 of its variance for a normal, so the bin medians are rescaled
    by that factor before the straight line is fitted.
    """
    import numpy as np

    # LOG-SPACED in the rate, not quantiles and not ranks. Quantile edges
    # put every bin at 10.5 ADU, because that is what nine pixels in ten
    # are, and a within-shoebox rank decile lumps the whole peak -- 20 to
    # 1000 ADU -- into one bin, where the median rate and the median
    # squared residual no longer describe the same pixels. Fixed log bins
    # give the bright end its own resolution, which is the only place the
    # slope is identifiable at all.
    lo = max(float(np.nanmin(lam)), 1e-3)
    hi = float(np.nanmax(lam))
    if not np.isfinite(hi) or hi <= lo * 1.01:
        return {"a": float("nan"), "b": float("nan"), "bins": []}
    edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(lam, edges) - 1, 0, n_bins - 1)
    x, y, rows = [], [], []
    for k in range(n_bins):
        m = idx == k
        if m.sum() < 50:
            continue
        lam_k = float(np.median(lam[m]))
        var_k = float(np.median(res[m] ** 2) / 0.4549)
        x.append(lam_k)
        y.append(var_k)
        rows.append({"bin": k, "rate": lam_k, "var": var_k,
                     "n": int(m.sum())})
    if len(x) < 3:
        return {"a": float("nan"), "b": float("nan"), "bins": rows}
    x, y = np.asarray(x), np.asarray(y)
    # each bin's median squared residual has relative error ~sqrt(2/n), so
    # the fit is weighted by sqrt(n); without it the sparse bright bins,
    # which carry the whole lever arm, are fitted as loosely as the
    # background ones that carry none
    wt = np.sqrt([r["n"] for r in rows])
    a, b = np.polyfit(x, y, 1, w=wt)
    return {"a": float(a), "b": float(b), "bins": rows}


def main() -> int:
    if not DATASET:
        raise SystemExit("set DATASET to the dataset these runs trained on")
    if len(sys.argv) < 3:
        raise SystemExit("usage: dump_heldout_score.py <out.csv> <run-dir>...")

    import numpy as np
    import torch
    import yaml

    from integrator.utils.factory_utils import (
        construct_data_loader,
        construct_integrator,
    )

    out_path, runs = sys.argv[1], sys.argv[2:]
    want = wanted_ids()
    if want:
        print(f"scoring {len(want)} named reflections from {_ids_file}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"holding out {HOLDOUT:.0%} of the valid pixels, keyed by refl_id")

    rows, misfit, noise, noise_bins, structure = [], [], [], [], []
    for run in runs:
        arm = os.path.basename(run.rstrip("/"))
        cfg_path = os.path.join(run, "config_log.yaml")
        ckpt = last_checkpoint(run)
        if not os.path.isfile(cfg_path) or not ckpt:
            print(f"  {arm}: no config_log.yaml or no checkpoint")
            continue
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh) or {}
        try:
            model = construct_integrator(cfg).to(device).eval()
            state = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state.get("state_dict", state), strict=False)
            dm = construct_data_loader(cfg)
            dm.setup("fit")
            loader = (dm.predict_dataloader()
                      if hasattr(dm, "predict_dataloader")
                      else dm.train_dataloader())
            got = score(model, loader, device, N_BATCHES, want)
        except Exception as exc:
            print(f"  {arm}: {type(exc).__name__}: {str(exc)[:110]}")
            continue
        if got is None:
            print(f"  {arm}: nothing scored")
            continue
        (lp, c2, bi, npx, d, pk, lam_px, res_px, ldec_px,
         ped_r, px_x, px_y, px_img) = got
        print(f"  {arm}: pedestal c0 median {np.median(ped_r):.2f} ADU, "
              f"spread {np.std(ped_r):.2f}")
        for r in pedestal_structure(ped_r, px_x, px_y, px_img, d):
            structure.append({"method": arm, **r})
        nm = noise_model(lam_px, res_px, ldec_px)
        print(f"  {arm}: noise model Var = {nm['a']:.3f} rate "
              f"+ {nm['b']:.2f}   (Poisson is 1.000 rate + 0)")
        for r in shape_of_misfit(pk, res_px):
            misfit.append({"method": arm, **r})
        noise.append({"method": arm, "a": nm["a"], "b": nm["b"]})
        for r in nm["bins"]:
            noise_bins.append({"method": arm, **r})
        thirds = (np.digitize(d, np.nanquantile(d, [1 / 3, 2 / 3]))
                  if np.isfinite(d).all() else np.zeros(len(d), int))
        # `d` is in angstrom, so the SMALLEST d is the highest resolution
        # and `digitize` puts it in band 0. Labelling band 0 `low_res`
        # inverted every resolution trend these tables report.
        for band, label in ((slice(None), "all"), (0, "high_res"),
                            (1, "mid_res"), (2, "low_res")):
            sel = (np.ones(len(lp), bool) if label == "all"
                   else thirds == band)
            if sel.sum() < 10:
                continue
            rows.append({
                "method": arm.replace("hewl1118_", "").replace(
                    "hewl9b7c_", ""),
                "dataset": DATASET,
                "band": label,
                "n_refl": int(sel.sum()),
                "n_px": float(npx[sel].mean()),
                "logp_px": float(lp[sel].mean()),
                "chi2_px": float(c2[sel].mean()),
                "bias_px": float(bi[sel].mean()),
            })
        print(f"  {arm}: {len(lp):,} reflections, "
              f"logp/px {lp.mean():.4f}, chi2/px {c2.mean():.3f}")

    if not rows:
        raise SystemExit("nothing scored")
    width = max(len(r["method"]) for r in rows) + 2
    print(f"\n{'arm':<{width}}{'band':<10}{'n_refl':>8}{'held px':>9}"
          f"{'logp/px':>10}{'chi2/px':>9}{'bias/px':>9}")
    for r in rows:
        print(f"{r['method']:<{width}}{r['band']:<10}{r['n_refl']:>8d}"
              f"{r['n_px']:>9.1f}{r['logp_px']:>10.4f}"
              f"{r['chi2_px']:>9.3f}{r['bias_px']:>9.3f}")

    if misfit:
        w2 = max(len(r["method"]) for r in misfit) + 2
        print(f"\nheld-out residual by profile decile "
              f"(p over the shoebox max)\n")
        print(f"{'arm':<{w2}}{'p decile':>10}{'n':>10}{'mean resid':>12}"
              f"{'median':>9}")
        for r in misfit:
            print(f"{r['method'].replace('hewl1118_', ''):<{w2}}"
                  f"{r['profile_decile']:>10d}{r['n']:>10d}"
                  f"{r['resid']:>12.3f}{r['resid_med']:>9.3f}")
    if noise_bins:
        w3 = max(len(r["method"]) for r in noise_bins) + 2
        print(f"\nvariance against rate, median squared residual per "
              f"rate decile\n")
        print(f"{'arm':<{w3}}{'rate':>10}{'var':>12}{'var/rate':>10}"
              f"{'n':>10}")
        for r in noise_bins:
            print(f"{r['method'].replace('hewl1118_', ''):<{w3}}"
                  f"{r['rate']:>10.3f}{r['var']:>12.3f}"
                  f"{r['var'] / max(r['rate'], 1e-9):>10.3f}{r['n']:>10d}")

    import csv

    def write(path, recs):
        if not recs:
            return
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0]))
            w.writeheader()
            w.writerows(recs)
        print(f"{path}: {len(recs)} rows")

    if structure:
        w4 = max(len(r["method"]) for r in structure) + 2
        print("\npedestal c0 against where it came from\n")
        print(f"{'arm':<{w4}}{'against':<13}{'value':>12}{'n':>9}"
              f"{'c0':>8}{'spread':>9}")
        for r in structure:
            print(f"{r['method'].replace('hewl1118_', ''):<{w4}}"
                  f"{r['against']:<13}{r['value']:>12.1f}{r['n']:>9d}"
                  f"{r['pedestal']:>8.2f}{r['spread']:>9.2f}")

    write(out_path, rows)
    base = out_path.rsplit(".csv", 1)[0]
    write(f"{base}_misfit.csv", misfit)
    write(f"{base}_noise.csv", noise_bins)
    write(f"{base}_pedestal.csv", structure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
