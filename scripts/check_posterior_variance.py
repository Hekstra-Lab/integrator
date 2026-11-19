#!/usr/bin/env python
"""Is the derived q(I) variance honest? Measured on a real run.

The SVAE's conjugate update pins

    Var_q[I] = E_q[I] / (tau + sum_valid pi)

which is Poisson-limited precision, as if every signal count were cleanly
attributed to the reflection. The actual Fisher information in the gate split
carries the responsibility SQUARED:

    I_exact ~ sum_p c_p g_p^2 / I^2        vs      I_conj ~ sum_p c_p g_p / I^2

so the ratio of variances is `sum_p c_p g_p^2 / sum_p c_p g_p`, which is 1
only when every gate is 0 or 1. Anywhere the gate is uncertain -- weak spots,
high background, which is most of a Laue shoebox -- the conjugate variance is
too small and the exported sigma too confident.

This computes that ratio per observation from a trained checkpoint, so the
size of the effect comes from the data rather than from a simulation, and
reports it against resolution and signal-to-noise. A merge weighted by
1/sigma^2 is only as good as this ratio is close to 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_dir", type=Path)
    p.add_argument("--batches", type=int, default=12)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--parquet", type=Path, default=None,
                   help="read an EXPORTED prediction parquet instead of "
                        "running the model, and report what each estimator's "
                        "variance carries beyond its own mean")
    return p.parse_args()


def exported(path: Path) -> None:
    """What the exported sigma carries, per estimator, from the parquet.

    The forward pass above asks whether the conjugate variance is honest.
    This asks a different question about the same run, on the columns that
    actually reach careless: given the intensity a reflection reported, does
    its sigma say anything further about it?

    It is the question the gate-architecture comparison turned on. The
    conjugate family pins `beta = beta_I + sum_i p_i = beta_I + 1` because
    the profile is normalised, so `var = I / (1 + tau)` exactly and
    `corr(log I, log var) = 1` by construction: the sigma is a relabelling
    of the mean and cannot tell careless which measurement is unreliable.
    The Rao-Blackwellized fit profiles the background out by Schur
    complement, so its variance depends on the shoebox as well.

    Reported per estimator: the correlation of the logs, and the PARTIAL
    correlation of log(var) with log(background) holding log(I) fixed --
    which is the part of the error bar that knows how much background the
    measurement was dug out of, and the part a sigma head would have to
    learn to reproduce.
    """
    import numpy as np
    import polars as pl

    df = pl.read_parquet(path)
    need = ("qi_mean", "qi_var", "i_rb", "var_rb", "qbg_mean", "d")
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} lacks {missing}; has {df.columns}")

    def logs(*names):
        cols = [df[n].to_numpy().astype(float) for n in names]
        good = np.ones(cols[0].shape, dtype=bool)
        for c in cols:
            good &= np.isfinite(c) & (c > 0)
        return [np.log(c[good]) for c in cols], good

    def partial(y, x, z):
        """corr(y, x) with z regressed out of both."""
        def resid(v):
            a = np.polyfit(z, v, 1)
            return v - (a[0] * z + a[1])
        ry, rx = resid(y), resid(x)
        if ry.std() < 1e-12 or rx.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(ry, rx)[0, 1])

    print(f"  {path}")
    print(f"  {df.height:,} reflections\n")
    print(f"  {'estimator':<10s} {'corr(logI, logvar)':>19s} "
          f"{'partial(logvar, logbg | logI)':>31s}")
    for name, i_col, v_col in (("posterior", "qi_mean", "qi_var"),
                               ("rb", "i_rb", "var_rb")):
        (li, lv, lb), _ = logs(i_col, v_col, "qbg_mean")
        r = float(np.corrcoef(li, lv)[0, 1])
        print(f"  {name:<10s} {r:19.5f} {partial(lv, lb, li):31.3f}")

    # by resolution third, because the shells are where the two estimators
    # diverged: the first shell agrees across arms and the last does not
    (li_r, lv_r, lb_r, ld), good = logs("i_rb", "var_rb", "qbg_mean", "d")
    (li_p, lv_p), _ = logs("qi_mean", "qi_var")
    cuts = np.quantile(ld, [1 / 3, 2 / 3])
    print(f"\n  {'d range (A)':<18s} {'n':>9s} "
          f"{'post corr':>10s} {'rb corr':>8s} {'rb partial bg':>14s}")
    for lo, hi, label in ((-np.inf, cuts[0], "high resolution"),
                          (cuts[0], cuts[1], "middle"),
                          (cuts[1], np.inf, "low resolution")):
        sel = (ld > lo) & (ld <= hi)
        if sel.sum() < 100:
            continue
        d_lo, d_hi = np.exp(ld[sel].min()), np.exp(ld[sel].max())
        rp = float(np.corrcoef(li_p[sel], lv_p[sel])[0, 1])
        rr = float(np.corrcoef(li_r[sel], lv_r[sel])[0, 1])
        pb = partial(lv_r[sel], lb_r[sel], li_r[sel])
        print(f"  {label:<18s} {int(sel.sum()):9,d} {rp:10.5f} "
              f"{rr:8.3f} {pb:14.3f}   ({d_lo:.2f}-{d_hi:.2f})")


def main() -> None:
    a = parse_args()
    if a.parquet is not None:
        exported(a.parquet)
        return
    import yaml

    from integrator.cli.utils.run_layout import resolve_run
    from integrator.utils import construct_integrator

    cfg_path, log_dir = resolve_run(a.run_dir)
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    ckpts = sorted((log_dir / "checkpoints").glob("epoch=*.ckpt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {log_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = construct_integrator(cfg).to(device).eval()
    state = torch.load(ckpts[-1].as_posix(), map_location=device)
    model.load_state_dict(state["state_dict"])
    print(f"  {ckpts[-1].name} on {device}")

    # the same construction `predict` uses: `inject_binning_labels` is what
    # puts `group_label` in the metadata, and the integrator's forward needs it
    from integrator.utils.factory_utils import construct_data_loader
    from integrator.utils.prepare_priors import inject_binning_labels

    dm = construct_data_loader(cfg)
    dm.setup()
    inject_binning_labels(dm, cfg)
    loader = dm.val_dataloader()
    if loader is None or len(loader.dataset) == 0:
        loader = dm.predict_dataloader()

    ratios, snr, dd, neff = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= a.batches:
                break
            batch = [
                b.to(device) if torch.is_tensor(b) else
                {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in b.items()} if isinstance(b, dict) else b
                for b in batch
            ]
            out = model(*batch)["forward_out"]
            counts = out["counts"].float()
            mask = out["mask"].float()
            if mask.dim() == 3:
                mask = mask.squeeze(-1)
            g = out.get("signal_prob")
            if g is None:
                raise SystemExit(
                    "this run has no `signal_prob`; the ratio is only defined "
                    "for the gated (SVAE) integrators"
                )
            g = g.float()
            if g.dim() == 3:
                g = g.mean(1)
            cm = counts * mask
            num = (cm * g.pow(2)).sum(-1)
            den = (cm * g).sum(-1).clamp_min(1e-9)
            ratios.append((num / den).cpu().numpy())
            qi = out["qi_mean"].reshape(-1).cpu().numpy()
            qv = out["qi_var"].reshape(-1).cpu().numpy()
            snr.append(qi / np.sqrt(np.clip(qv, 1e-12, None)))
            md = batch[3] if len(batch) > 3 else {}
            dd.append(md["d"].reshape(-1).cpu().numpy())
            if "n_eff" in out:
                neff.append(out["n_eff"].reshape(-1).cpu().numpy())

    r = np.concatenate(ratios)
    s = np.concatenate(snr)
    d = np.concatenate(dd)
    print(f"\n  {len(r):,} observations\n")
    print("  var_conj / var_exact = sum c g^2 / sum c g")
    print("  (1 = honest; below 1 = the exported sigma is too small)\n")
    q = np.nanquantile(r, [0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"  quantiles 5/25/50/75/95: "
          f"{np.array2string(q, precision=3)}")
    print(f"  implied sd understatement, median: {1/np.sqrt(np.median(r)):.2f}x")

    print(f"\n  {'I/sigma quintile':>22s} {'n':>8s} {'ratio':>8s} {'sd x':>7s}")
    qs = np.nanquantile(s, np.linspace(0, 1, 6))
    for i in range(5):
        m = (s >= qs[i]) & (s <= qs[i + 1])
        if m.sum() < 20:
            continue
        med = np.median(r[m])
        print(f"  {qs[i]:9.1f}-{qs[i+1]:9.1f} {m.sum():8,d} "
              f"{med:8.3f} {1/np.sqrt(med):7.2f}")

    print(f"\n  {'d shell (A)':>22s} {'n':>8s} {'ratio':>8s} {'sd x':>7s}")
    s2 = 1.0 / np.clip(d, 1e-6, None) ** 2
    de = np.nanquantile(s2, np.linspace(0, 1, 7))
    for i in range(6):
        m = (s2 >= de[i]) & (s2 <= de[i + 1])
        if m.sum() < 20:
            continue
        med = np.median(r[m])
        lo, hi = 1 / np.sqrt(de[i + 1]), 1 / np.sqrt(max(de[i], 1e-12))
        print(f"  {hi:10.2f}-{lo:9.2f} {m.sum():8,d} "
              f"{med:8.3f} {1/np.sqrt(med):7.2f}")

    if a.out_dir is None:
        return
    a.out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ok = np.isfinite(r) & np.isfinite(s)
    ax[0].hexbin(np.clip(s[ok], 0, 30), r[ok], gridsize=80, bins="log",
                 cmap="magma")
    ax[0].axhline(1.0, color="w", ls="--", lw=1)
    ax[0].set_xlabel("I/sigma as exported")
    ax[0].set_ylabel("var_conj / var_exact")
    ax[0].set_title("1.0 = the reported sigma is honest", fontsize=9)
    ax[1].hist(1 / np.sqrt(np.clip(r[ok], 1e-6, None)), bins=80,
               range=(1, 4), color="steelblue")
    ax[1].set_xlabel("sd understatement factor")
    ax[1].set_ylabel("observations")
    ax[1].set_title("how much too confident", fontsize=9)
    fig.tight_layout()
    out = a.out_dir / "fig_posterior_variance.png"
    fig.savefig(out, dpi=a.dpi, bbox_inches="tight")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
