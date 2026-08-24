"""Probe: can geometry predict DIALS's per-observation scale?

Answers the question "is the MLP scale not expressive enough, or is it just hard
to optimize to a viable solution?" by SUPERVISED-regressing DIALS's per-obs scale
(`inverse_scale_factor` from a scaled.refl) onto feature sets of increasing
richness and networks of increasing capacity:

  - feature sets: the current MLPScale inputs [frame, x, y, lp, d]; + the
    diffracted-beam direction s1 (directional/absorption info the MLP lacks).
  - capacities: linear, 2-layer MLP (= MLPScale), 4-layer MLP.

Reading the table:
  * High R^2 already with [frame,x,y,lp,d] + 2-layer  -> the network CAN represent
    DIALS's scale from its current inputs; the model's 29-vs-32 gap is then an
    OPTIMIZATION issue (the ELBO isn't driving s_i to that scale), not expressivity.
  * Low R^2 that jumps when s1 is added            -> a missing-INPUT / coordinate
    (absorption-geometry) gap; the MLP needs directional features.
  * 4-layer >> 2-layer                              -> capacity helps (use a bigger net).
  * linear ~ MLP                                    -> capacity isn't the issue; inputs are.

NOTE: s1 here is the LAB-frame diffracted beam. True absorption lives in the
CRYSTAL frame (s1 rotated by the per-image orientation U); a proper test of that
needs the .expt (scan-varying U via dxtbx). This probe still shows whether
directional info helps at all.

Usage:
    uv run python scripts/regress_dials_scale.py SCALED.refl [--max-obs N]
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import reciprocalspaceship as rs
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def read_scaled_refl(path: str) -> rs.DataSet:
    """Read a DIALS scaled.refl, tolerating columns rs can't parse."""
    want = ["xyzcal.px", "s1", "lp", "d", "inverse_scale_factor"]
    for cols in (want, ["xyzcal.px", "lp", "d", "inverse_scale_factor"]):
        try:
            ds = rs.io.read_dials_stills(path, extra_cols=cols)
            logger.info("Read %s: %d rows, columns=%s", path, len(ds),
                        sorted(ds.columns))
            return ds
        except Exception as e:  # noqa: BLE001
            logger.warning("read with %s failed: %s", cols, e)
    raise RuntimeError(f"Could not read {path}")


def _col(ds, name):
    if name not in ds.columns:
        raise KeyError(
            f"Column {name!r} not in scaled.refl (have {sorted(ds.columns)}). "
            "Did dials.scale write it? (need inverse_scale_factor)"
        )
    return ds[name].to_numpy().astype(np.float64)


def build_features(ds) -> tuple[dict, np.ndarray]:
    """Return {feature_set_name: (N, F) array} and the target log-scale (N,)."""
    scale = _col(ds, "inverse_scale_factor")
    frame = _col(ds, "xyzcal.px.2")
    x = _col(ds, "xyzcal.px.0")
    y = _col(ds, "xyzcal.px.1")
    lp = _col(ds, "lp")
    d = _col(ds, "d")

    good = np.isfinite(scale) & (scale > 0) & np.isfinite(d) & (d > 0)
    target = np.log(scale[good])

    base = np.stack([frame, x, y, lp, d], axis=1)[good]
    sets = {"frame_only": frame[good][:, None], "mlp_inputs": base}
    if all(f"s1.{i}" in ds.columns for i in range(3)):
        s1 = np.stack([_col(ds, f"s1.{i}") for i in range(3)], axis=1)[good]
        sets["mlp_inputs+s1"] = np.concatenate([base, s1], axis=1)
    else:
        logger.warning("s1 not present; skipping the +s1 feature set")
    logger.info("Target log(scale): N=%d mean=%.3f std=%.3f",
                len(target), target.mean(), target.std())
    return sets, target


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, n_layers):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_and_score(
    X: np.ndarray, y: np.ndarray, hidden: int, n_layers: int,
    device, seed: int = 0, epochs: int = 400,
) -> float:
    """Train on 80%, return held-out R^2 (fraction of log-scale variance)."""
    g = torch.Generator().manual_seed(seed)
    n = len(y)
    perm = torch.randperm(n, generator=g).numpy()
    n_val = max(1, n // 5)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    mu, sd = Xt[tr_idx].mean(0), Xt[tr_idx].std(0).clamp(min=1e-6)
    Xt = (Xt - mu) / sd
    ym, ys = yt[tr_idx].mean(), yt[tr_idx].std().clamp(min=1e-6)
    yn = (yt - ym) / ys

    Xt, yn, yt = Xt.to(device), yn.to(device), yt.to(device)
    tr = torch.tensor(tr_idx, device=device)
    va = torch.tensor(val_idx, device=device)

    model = MLP(X.shape[1], hidden, n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bs = min(65536, len(tr))
    for _ in range(epochs):
        model.train()
        b = tr[torch.randperm(len(tr), device=device)[:bs]]
        opt.zero_grad()
        loss = ((model(Xt[b]) - yn[b]) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xt[va]) * ys + ym  # back to log-scale units
        tgt = yt[va]
        ss_res = ((pred - tgt) ** 2).sum()
        ss_tot = ((tgt - tgt.mean()) ** 2).sum().clamp(min=1e-12)
        return float(1.0 - ss_res / ss_tot)


def main():
    ap = argparse.ArgumentParser(
        description="Regress DIALS per-obs scale on geometry to test whether the "
        "MLP scale is expressivity- or optimization-limited."
    )
    ap.add_argument("scaled_refl", help="DIALS scaled.refl (has inverse_scale_factor)")
    ap.add_argument("--max-obs", type=int, default=600000,
                    help="Subsample to this many observations for speed")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = read_scaled_refl(args.scaled_refl)
    sets, target = build_features(ds)

    if len(target) > args.max_obs:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(target), args.max_obs, replace=False)
        target = target[sel]
        sets = {k: v[sel] for k, v in sets.items()}
        logger.info("Subsampled to %d observations", args.max_obs)

    arches = [("linear", 0, 0), ("mlp-2x64", 64, 2), ("mlp-4x128", 128, 4)]
    rows = []
    for fname, feats in sets.items():
        for aname, hidden, nl in arches:
            r2 = fit_and_score(feats, target, hidden, nl, device)
            rows.append((fname, aname, feats.shape[1], r2))
            logger.info("%-16s | %-10s | F=%d | R2=%.4f",
                        fname, aname, feats.shape[1], r2)

    print("\n=== R^2 predicting log(DIALS inverse_scale_factor) ===")
    print(f"{'features':<16} {'arch':<10} {'F':>3}  {'R^2':>7}")
    for fname, aname, f, r2 in rows:
        print(f"{fname:<16} {aname:<10} {f:>3}  {r2:>7.4f}")
    print(
        "\nHigh R^2 with mlp_inputs+mlp-2x64 -> expressive enough; gap is "
        "OPTIMIZATION.\nLow until +s1 -> missing directional/absorption inputs.\n"
        "4x128 >> 2x64 -> capacity helps. linear ~ mlp -> inputs, not capacity."
    )


if __name__ == "__main__":
    main()
