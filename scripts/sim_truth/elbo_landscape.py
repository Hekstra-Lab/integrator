"""The ELBO and each of its terms along profile-shape directions, at the parameters that generated an SBC dataset.

On a `sbc_simulate.py` prior dataset drawn from a checkpoint, that checkpoint
holds the true values of every global parameter, including the profile's
metadata prior theta0 + f(m). This script moves theta0 along one direction at
a time, everything else fixed,

    width   +Delta on every along and across raw width (all components)
    along   +Delta on the along raw widths only
    w0      +Delta on the logit weight of component 0 (the narrowest)

and at each Delta evaluates, on the same reflections and with the same random
numbers (common seeds per batch), the per-reflection means of the loss's own
terms -- Poisson negative log-likelihood, KL of q(I), KL of q(b), profile KL --
and their sum, the negative ELBO. Beside them: the median n_eff = 1 / sum p^2 of
the mean profile and the relative bias of i_rb against the true I in bins of
I_true. If the negative ELBO is lowest away from Delta = 0, the objective
itself prefers a profile other than the true one, and the term whose minimum
moves is the one responsible; if it is lowest at 0, training's narrowing is an
optimisation or amortisation effect.

The amortized parts (encoders, q(eta) heads, gate) were trained at Delta = 0,
so this is the slope of the ELBO along theta0 with them held, not a retraining.

usage: python elbo_landscape.py --ckpt CKPT --config SBC_CONFIG --data SBC_DIR --out DIR
           [--n-batches 60] [--deltas -0.4,-0.3,...] [--device cuda] [--integrator-args rb_profile=mean]
Writes DIR/elbo_landscape.csv (direction, delta, term, value) and prints the table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml

I_BINS = [(10, 30), (30, 100), (100, 300), (1000, 3000)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-batches", type=int, default=60)
    ap.add_argument("--deltas", default="-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--integrator-args", default="rb_profile=mean")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    from integrator.utils import construct_data_loader, construct_integrator, inject_binning_labels

    cfg = yaml.safe_load(a.config.read_text())
    cfg["data_loader"]["args"]["data_dir"] = str(a.data)
    cfg["data_loader"]["args"]["shoebox_file_names"]["data_dir"] = str(a.data)
    for kv in filter(None, a.integrator_args.split(",")):
        k, v = kv.split("=", 1)
        cfg["integrator"]["args"][k] = yaml.safe_load(v)
    cfg["integrator"]["args"]["predict_keys"] = ["refl_ids", "i_rb", "var_rb", "qp_mean"]
    dl = construct_data_loader(cfg)
    dl.setup()
    inject_binning_labels(dl, cfg)
    model = construct_integrator(cfg)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    model = model.eval().to(a.device)
    qp = model.surrogates["qp"]
    K = qp.K

    batches = []
    for bi, (counts, shoebox, mask, md) in enumerate(dl.predict_dataloader()):
        if bi >= a.n_batches:
            break
        batches.append((counts.to(a.device), shoebox.to(a.device), mask.to(a.device),
                        {k: v.to(a.device) if torch.is_tensor(v) else v for k, v in md.items()}))
    t = np.load(a.data / "sbc_truth.npz")
    row_of = {int(r): i for i, r in enumerate(t["refl_ids"])}
    I_true = torch.as_tensor(t["I"])

    theta0 = qp.theta0.detach().clone()
    directions = {"width": list(range(0, 2 * K)), "along": list(range(0, K)), "w0": [2 * K]}
    deltas = [float(x) for x in a.deltas.split(",")]
    rows = []
    for name, idx in directions.items():
        for dlt in deltas:
            with torch.no_grad():
                qp.theta0.copy_(theta0)
                qp.theta0[idx] += dlt
                acc = {"neg_ll": 0.0, "kl_i": 0.0, "kl_bg": 0.0, "kl_prf": 0.0}
                n, neffs, rel, Iv = 0, [], [], []
                for bi, (counts, shoebox, mask, md) in enumerate(batches):
                    torch.manual_seed(1000 + bi)
                    out = model(counts, shoebox, mask, md)
                    ld = model.loss(rate=out["forward_out"]["rates"], counts=out["forward_out"]["counts"],
                                    qp=out["qp"], qi=out["qi"], qbg=out["qbg"], mask=out["forward_out"]["mask"],
                                    group_labels=md["group_label"].long(), metadata=md,
                                    qbg_plane=out.get("qbg_plane"), coset_mode=model.coset_mode)
                    B = counts.shape[0]
                    for key, lk in (("neg_ll", "neg_ll_mean"), ("kl_i", "kl_i_mean"), ("kl_bg", "kl_bg_mean"),
                                    ("kl_prf", "kl_prf_mean")):
                        acc[key] += float(ld[lk]) * B
                    n += B
                    torch.manual_seed(1000 + bi)
                    pr = model.predict_step((counts, shoebox, mask, md), bi)
                    p = pr["qp_mean"].reshape(B, -1)
                    neffs.append((1 / p.pow(2).sum(-1)).cpu())
                    It = I_true[[row_of[int(r)] for r in pr["refl_ids"].cpu()]]
                    rel.append(((pr["i_rb"].cpu() - It) / It.clamp_min(1e-6)))
                    Iv.append(It)
            vals = {k: v / n for k, v in acc.items()}
            vals["neg_elbo"] = sum(vals.values())
            vals["median_n_eff"] = float(torch.cat(neffs).median())
            rel_c, I_c = torch.cat(rel), torch.cat(Iv)
            for lo, hi in I_BINS:
                s = (I_c >= lo) & (I_c < hi)
                vals[f"i_rb_bias_{lo}_{hi}"] = float(rel_c[s].mean()) if s.any() else float("nan")
            rows += [dict(direction=name, delta=dlt, term=k, value=v) for k, v in vals.items()]
            print(name, dlt, {k: round(v, 4) for k, v in vals.items()}, flush=True)
    with torch.no_grad():
        qp.theta0.copy_(theta0)
    df = pl.DataFrame(rows)
    df.write_csv(a.out / "elbo_landscape.csv")
    base = df.filter(pl.col("delta") == 0.0).select("direction", "term", pl.col("value").alias("v0"))
    rel_df = df.join(base, on=["direction", "term"]).with_columns(
        d=pl.when(pl.col("term").str.starts_with("i_rb") | (pl.col("term") == "median_n_eff"))
        .then(pl.col("value")).otherwise(pl.col("value") - pl.col("v0")))
    with pl.Config(tbl_rows=200, tbl_cols=20, float_precision=4):
        for name in directions:
            print(name, "(ELBO terms relative to delta 0; n_eff and i_rb bias absolute)")
            print(rel_df.filter(direction=name).pivot(on="term", index="delta", values="d").sort("delta"))


if __name__ == "__main__":
    main()
