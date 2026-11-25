"""Check of the SBC pipeline itself: PIT values under the exact posterior of one latent given the true values of the others.

On a `sbc_simulate.py` prior dataset, for each reflection j with the true profile
p_j (rebuilt from the saved eta_j by the generating checkpoint's surrogate) and
the true background b_j, the exact posterior of the intensity is

    pi(I | c_j, p_j, b_j)  proportional to  Gamma(I; c_I, r_I) prod_k Poisson(c_jk; I p_jk + b_j)

with the Gamma the prior the simulation drew from (`i_conc`, `i_rate` of
`sbc_truth.npz`). It is one-dimensional, so its CDF at I_true is computed on a
grid (`--grid` evenly spaced points over +-12 likelihood sd about a data estimate, floored above 0)
without any approximation beyond the grid. Likewise for b given the true I and
p, with prior Gamma(b_conc, b_rate). If the simulation, the stored truths, the
priors and the PIT code are right, both PIT histograms are uniform; any
departure is a defect of the pipeline, not of a model.

usage: python sbc_exact_check.py --ckpt GENERATOR_CKPT --config GENERATOR_CONFIG --data SBC_DIR --out DIR
           [--n 40000] [--grid 1500] [--device cuda]
Writes DIR/sbc_exact_pit.csv (per reflection) and prints the summary and 20-bin histograms.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sbc_ranks import summarise  # noqa: E402
from sbc_simulate import eta_prior_mean, profile_at  # noqa: E402


def grid_cdf(x_true, lo, hi, logpost_fn, G):
    """CDF at x_true of a 1-D density given up to a constant, on G evenly spaced points in [lo, hi] per row."""
    t = torch.linspace(0, 1, G, device=lo.device, dtype=torch.float64)
    xs = lo[:, None] + (hi - lo)[:, None] * t[None]  # (B, G)
    lp = logpost_fn(xs)
    w = torch.exp(lp - lp.max(1, keepdim=True).values)
    cdf = torch.cumsum(w, 1) / w.sum(1, keepdim=True)
    idx = torch.searchsorted(xs, x_true[:, None]).clamp(0, G - 1)
    return cdf.gather(1, idx)[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--grid", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    from integrator.io import load_data
    from integrator.utils.factory_utils import construct_integrator

    cfg = yaml.safe_load(a.config.read_text())
    model = construct_integrator(cfg)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    qp = model.surrogates["qp"].eval().to(a.device)

    t = np.load(a.data / "sbc_truth.npz")
    meta = {k: np.asarray(v) for k, v in load_data(a.data / "metadata.npy").items()}
    counts = np.load(a.data / "counts.npy", mmap_mode="r")
    masks = np.load(a.data / "masks.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(len(t["I"]), size=min(a.n, len(t["I"])), replace=False))
    out = {"row": [], "I_true": [], "u_I_exact": [], "u_b_exact": []}
    dev = a.device
    with torch.no_grad():
        for i in range(0, len(rows), a.batch):
            r = rows[i:i + a.batch]
            md = {k: torch.as_tensor(v[r].astype(np.float32), device=dev) for k, v in meta.items()}
            m = torch.as_tensor(np.asarray(masks[r]), device=dev).double()
            c = torch.as_tensor(np.asarray(counts[r]), device=dev).double()
            eta = torch.as_tensor(t["eta"][r], device=dev).float()
            phi, fxy, _ = eta_prior_mean(qp, md, m.float())
            p = profile_at(qp, phi, fxy, eta, m.float()).double()
            I0 = torch.as_tensor(t["I"][r], device=dev).double()
            b0 = torch.as_tensor(t["b"][r], device=dev).double()
            ic, ir = (torch.as_tensor(t[k][r], device=dev).double() for k in ("i_conc", "i_rate"))
            bc, br = (torch.as_tensor(t[k][r], device=dev).double() for k in ("b_conc", "b_rate"))
            npx = m.sum(1)

            def lp_I(xs):  # xs (B, G)
                lam = xs[:, :, None] * p[:, None, :] + b0[:, None, None]
                ll = (m[:, None, :] * (c[:, None, :] * torch.log(lam) - lam)).sum(-1)
                return ll + (ic[:, None] - 1) * torch.log(xs) - ir[:, None] * xs

            def lp_b(xs):
                lam = I0[:, None, None] * p[:, None, :] + xs[:, :, None]
                ll = (m[:, None, :] * (c[:, None, :] * torch.log(lam) - lam)).sum(-1)
                return ll + (bc[:, None] - 1) * torch.log(xs) - br[:, None] * xs

            # grids: +-12 sd of the likelihood about a data estimate (not about the truth), floored above 0
            s2 = (p * p * m).sum(1)
            i_hat = ((p * (c - b0[:, None]) * m).sum(1) / s2).clamp_min(0)
            sd_i = torch.sqrt((i_hat + b0 / s2.clamp_min(1e-12) + 1.0))
            uI = grid_cdf(I0, (i_hat - 12 * sd_i).clamp_min(1e-6), i_hat + 12 * sd_i + 5, lp_I, a.grid)
            b_hat = ((c - I0[:, None] * p) * m).sum(1) / npx
            sd_b = torch.sqrt(b_hat.clamp_min(0.5) / npx)
            ub = grid_cdf(b0, (b_hat - 12 * sd_b).clamp_min(1e-6), b_hat + 12 * sd_b, lp_b, a.grid)
            for k, v in (("row", torch.as_tensor(r)), ("I_true", I0), ("u_I_exact", uI), ("u_b_exact", ub)):
                out[k].append(v.cpu().numpy())
    df = pl.DataFrame({k: np.concatenate(v) for k, v in out.items()})
    df.write_csv(a.out / "sbc_exact_pit.csv")
    for q in ("u_I_exact", "u_b_exact"):
        u = df[q].to_numpy()
        h, _ = np.histogram(u, bins=20, range=(0, 1))
        print(q, summarise(u), "histogram / (n/20):", np.round(h / (len(u) / 20), 2).tolist())


if __name__ == "__main__":
    main()
