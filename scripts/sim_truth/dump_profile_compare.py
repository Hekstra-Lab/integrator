"""True profiles beside each model's prior profile and Laplace-mode profile, for fig_sim_profile_compare.py.

For the reflections of a `dump_profile_truth_examples.py --sample N` npz (their
true profiles and shoebox counts), and for each model given as NAME=CKPT:CONFIG:

    prior    the prior profile: the mean over 16 draws eta ~ N(theta0 + f(m), sigma^2),
             what a reflection too faint to move its shape is integrated with
    mode     the profile at the joint Laplace mode of (eta, I, b) given the reflection's
             own counts (`integrator.model.posterior.collapsed.laplace_eta`, from the prior
             mean), what the Laplace + collapsed export uses at its centre

Writes one npz with `true_profile`, `counts` (N, 25, 25), `i_true`, `d`, `phi`
(the streak azimuth), `refl_ids`, and per model `<name>_prior`, `<name>_mode`
(N, 25, 25, float32). Derived quantities (means over cells, differences, kappa)
are left to the figure.

usage: python dump_profile_compare.py --npz SAMPLE.npz --data SHOEBOX_DIR --out OUT.npz
           --model NAME=CKPT:CONFIG [--model ...] [--device cuda]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from integrator.model.posterior.collapsed import log_lik, laplace_eta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", action="append", required=True, help="NAME=CKPT:CONFIG")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    from integrator.io import load_data
    from integrator.utils.factory_utils import construct_integrator

    z = np.load(a.npz, allow_pickle=True)
    meta = {k: np.asarray(v) for k, v in load_data(a.data / "metadata.npy").items()}
    row_of = {int(r): i for i, r in enumerate(meta["refl_ids"])}
    rows = np.array([row_of[int(r)] for r in z["refl_ids"]])
    counts = np.load(a.data / "counts.npy", mmap_mode="r")
    masks = np.load(a.data / "masks.npy", mmap_mode="r")
    gl = a.data / "group_labels_30.npy"
    glab = np.asarray(load_data(gl)) if gl.exists() else np.zeros(len(meta["d"]), dtype=np.int64)
    N, D = len(rows), torch.float64
    out = {k: z[k] for k in ("true_profile", "i_true", "d", "refl_ids")}
    order = np.argsort(rows)
    inv = np.argsort(order)
    out["counts"] = np.asarray(counts[rows[order]])[inv].reshape(N, 25, 25).astype(np.float32)
    dev = a.device
    for spec in a.model:
        name, rest = spec.split("=", 1)
        ck, cfgp = rest.split(":", 1)
        cfg = yaml.safe_load(Path(cfgp).read_text())
        cfg["integrator"]["args"].pop("init_from_checkpoint", None)
        la = cfg["loss"]["args"]
        for key in ("spectrum_init_from", "eb_init_from"):
            if la.get(key) and not Path(la[key]).exists():
                la[key] = None
        model = construct_integrator(cfg)
        model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["state_dict"], strict=True)
        model = model.eval().to(dev)
        qp = model.surrogates["qp"].double()
        for prm in qp.parameters():
            prm.requires_grad_(False)
        pri, mod, phis = [], [], []
        for i in range(0, N, a.batch):
            r = rows[i:i + a.batch]
            o = np.argsort(r)
            iv = np.argsort(o)
            md = {k: torch.as_tensor(v[r].astype(np.float64), device=dev) for k, v in meta.items()}
            md["group_label"] = torch.as_tensor(glab[r], device=dev).long()
            c = torch.as_tensor(np.asarray(counts[r[o]])[iv], device=dev).to(D).reshape(len(r), -1)
            m = torch.as_tensor(np.asarray(masks[r[o]])[iv], device=dev).to(D).reshape(len(r), -1)
            phi, fxy, em = qp.eta_prior(md, m)
            sig = qp.eta_sd.to(D)
            g = torch.Generator(device=dev).manual_seed(0)
            with torch.no_grad():
                p_prior = torch.stack([qp.profile_from_eta(phi, fxy, em + sig * torch.randn(em.shape, generator=g, device=dev, dtype=D), m)
                                       for _ in range(16)]).mean(0)
            md32 = {k: (v.float() if v.is_floating_point() else v) for k, v in md.items()}  # the loss is float32
            a_I, r_I = model.loss.intensity_prior_params(md32, dev)
            a_b, r_b = model._background_prior(md32, dev)
            a_I, r_I, a_b, r_b = (torch.as_tensor(x, dtype=D, device=dev).expand(len(r)) for x in (a_I, r_I, a_b, r_b))
            NE = em.shape[1]
            b0 = (c * m).sum(1) / m.sum(1)
            i0 = ((c - b0[:, None]) * m).sum(1).clamp_min(1.0)

            def nlp(th):
                eta, I, b = th[:, :NE], th[:, NE], th[:, NE + 1]
                p = qp.profile_from_eta(phi, fxy, eta, m)
                return (-log_lik(c, m, I, b, p) + r_I * I - (a_I - 1) * torch.log(I) + r_b * b - (a_b - 1) * torch.log(b)
                        + 0.5 * (((eta - em) / sig) ** 2).sum(-1))

            mode, _, _ = laplace_eta(nlp, torch.cat([em, i0[:, None], b0[:, None].clamp_min(0.05)], 1), steps=40,
                                     positive=(NE, NE + 1))
            with torch.no_grad():
                p_mode = qp.profile_from_eta(phi, fxy, mode[:, :NE], m)
            pri.append(p_prior.cpu().numpy())
            mod.append(p_mode.cpu().numpy())
            phis.append(phi.cpu().numpy())
        out[f"{name}_prior"] = np.concatenate(pri).reshape(N, 25, 25).astype(np.float32)
        out[f"{name}_mode"] = np.concatenate(mod).reshape(N, 25, 25).astype(np.float32)
        out["phi"] = np.concatenate(phis)
        print(f"{name}: done", flush=True)
    out["models"] = np.array([s.split("=", 1)[0] for s in a.model])
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
