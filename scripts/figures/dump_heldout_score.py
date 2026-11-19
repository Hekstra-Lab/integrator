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


def holdout_mask(refl_ids, n_px, frac, device):
    """A boolean hold-out mask per reflection, keyed by its OWN id.

    Seeding from the id rather than from the batch makes the draw a
    property of the reflection, so every arm holds out the same pixels of
    the same reflection however the loaders order them.
    """
    import numpy as np
    import torch

    out = np.zeros((len(refl_ids), n_px), dtype=bool)
    for k, rid in enumerate(refl_ids):
        rng = np.random.default_rng(int(rid) & 0x7FFFFFFF)
        take = rng.random(n_px) < frac
        out[k] = take
    return torch.from_numpy(out).to(device)


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
            held = holdout_mask(rid_np, n_px, HOLDOUT, device)
            # only pixels that were valid to begin with can be held out
            held = held & m2.bool()
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
            h = held.reshape(len(rid_np), -1)
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
    return tuple(np.concatenate(v) for v in (logp, chi2, bias, npx, dd))


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

    rows = []
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
        lp, c2, bi, npx, d = got
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

    import csv

    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{out_path}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
