"""Is the profile encoder alive? Dead-unit fractions over training.

`ProfileEncoder.output_activation` defaults to `"relu"`, and its own
docstring records the failure that default allows: "on sim_geom_both every
one of the 32 pre-activations was negative on every shoebox, so the profile
latent received no gradient and could not recover." A dead encoder is not a
crash. Training continues, the ELBO falls, the basis still moves -- the
DECODER is a free parameter and keeps learning -- and the profile simply
stops depending on the shoebox. Every reflection then gets the same shape.

So the basis diagnostics cannot answer this. `basis_snapshots.npz` is the
decoder; this is the encoder, and the two fail independently.

Three numbers per checkpoint, measured on real shoeboxes rather than
inferred from the config:

- **dead_fraction** -- output units that are zero on EVERY shoebox in the
  sample. These are the units that cannot recover: ReLU has no gradient at
  zero, so a unit whose pre-activation is negative everywhere is finished.
- **zero_fraction** -- the mean fraction of the embedding that is zero on a
  given shoebox. Sparsity rather than death; high is not necessarily wrong.
- **preact_mean/max** -- the pre-activation before the ReLU, which says
  whether a dead unit is marginally or hopelessly negative.

The pre-activation is read with a forward hook on the encoder's final
linear layer, so nothing in the model is modified and the measurement is of
the model as trained.

With GEOM_OUT=<csv> and a `geometric_basis_profile` surrogate, the last
checkpoint of each run is also taken apart on the same shoeboxes, through a
forward hook on the surrogate (so the inputs are the model's own): the norm
of the pixel latent `h` (`latent_gain * loc_head(x)`) against the geometry
shift `m(s)` (`shift_head`), the posterior width `scale`, the learned prior
width `sigma(s)`, the geometric log extents (along, across the streak), the
extent gate `s / (s + k)` from the shoebox's signal estimate, the gated pixel
residual on the extents, and the profile's `n_eff = 1 / sum p^2`. Rows are
quartiles of the signal estimate, plus `all`, so the question "does the
profile depart from geometry as the shoebox gets brighter" is read directly.

Usage:
    DATASET=pdz2_1116 python dump_encoder_health.py <out.csv> <run-dir>...
    GEOM_OUT=geom.csv ...  # the geometric-surrogate breakdown above
    N_SHOEBOXES=4096 ... # sample size, default 2048
    STRIDE=5 ...         # every 5th checkpoint, default every one
"""

from __future__ import annotations

import csv
import glob
import os
import re
import sys

DATASET = os.environ.get("DATASET", "")
N_SHOEBOXES = int(os.environ.get("N_SHOEBOXES", "2048"))
STRIDE = int(os.environ.get("STRIDE", "1"))
COLUMNS = ("method", "dataset", "epoch", "activation", "n_units",
           "dead_fraction", "zero_fraction", "preact_mean", "preact_max",
           "n_shoeboxes")


def epoch_of(path: str) -> int | None:
    m = re.search(r"epoch=(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def checkpoints(run_dir: str) -> list[tuple[int, str]]:
    found = []
    for p in sorted(glob.glob(f"{run_dir}/files/checkpoints/epoch=*.ckpt")):
        e = epoch_of(p)
        if e is not None:
            found.append((e, p))
    return found[::STRIDE] if STRIDE > 1 else found


def ensure_group_label(metadata, device):
    """Supply `group_label` when the installed data module did not.

    The repository's `POLY_DS_COLS` lists it, but the environment's
    installed package is older and the PDZ2 metadata carries `radial_bin`
    instead, so the forward died on `KeyError: 'group_label'`. It selects
    the per-group prior, so a wrong value would move the PRIOR -- not the
    encoder, the profile shape or the gate, which is what is being
    measured here. `radial_bin` is used where present and zeros
    otherwise, and the substitution is printed rather than done quietly.
    """
    import torch

    if "group_label" in metadata:
        return metadata, ""
    ref = next((v for v in metadata.values() if hasattr(v, "shape")), None)
    if ref is None:
        return metadata, ""
    if "radial_bin" in metadata:
        metadata["group_label"] = metadata["radial_bin"].long()
        return metadata, "group_label <- radial_bin"
    metadata["group_label"] = torch.zeros(
        ref.shape[0], dtype=torch.long, device=device)
    return metadata, "group_label <- zeros (prior groups collapsed)"


def profile_encoder(model):
    """The encoder whose output the profile surrogate consumes."""
    encoders = getattr(model, "encoders", None)
    if encoders is None:
        return None
    for key in ("profile", "profile_encoder", "qp"):
        if key in encoders:
            return encoders[key]
    return None


def measure(model, batch, device) -> dict | None:
    """Dead and zero fractions of the profile embedding, on real shoeboxes.

    The MODEL's own forward is run, not the encoder's. The encoder is not
    handed the raw counts: the integrator masks the shoebox, reshapes it,
    and may widen it with coordinate channels before passing it on, so
    calling the encoder directly would measure a tensor the model never
    builds. Hooks on the final linear and on the encoder itself read the
    pre- and post-activation out of the real path.
    """
    import torch

    enc = profile_encoder(model)
    if enc is None:
        return None
    final = getattr(enc, "fc", None)
    if final is None:
        return None

    pre: list = []
    post_out: list = []
    handles = [
        final.register_forward_hook(
            lambda _m, _i, out: pre.append(out.detach())),
        enc.register_forward_hook(
            lambda _m, _i, out: post_out.append(out.detach())),
    ]
    try:
        with torch.no_grad():
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                counts, shoebox, mask, metadata = batch
                metadata, note = ensure_group_label(dict(metadata), device)
                if note:
                    print(f"    {note}")
                model(counts, shoebox, mask, metadata)
            elif isinstance(batch, (list, tuple)):
                model(*batch)
            else:
                model(batch)
    except Exception as exc:  # the step signature differs by integrator
        for h in handles:
            h.remove()
        print(f"    forward failed: {type(exc).__name__}: {exc}")
        return None
    finally:
        for h in handles:
            h.remove()
    if not pre or not post_out:
        return None

    z = pre[0].reshape(pre[0].shape[0], -1).float()
    post = post_out[0].reshape(post_out[0].shape[0], -1).float()
    # a unit that is zero on every shoebox in the sample: ReLU has no
    # gradient at zero, so this one cannot come back
    dead = (post.abs() <= 0).all(dim=0)
    return {
        "activation": getattr(enc, "output_activation", "?"),
        "n_units": int(post.shape[1]),
        "dead_fraction": round(float(dead.float().mean()), 6),
        "zero_fraction": round(float((post.abs() <= 0).float().mean()), 6),
        "preact_mean": round(float(z.mean()), 6),
        "preact_max": round(float(z.max()), 6),
        "n_shoeboxes": int(post.shape[0]),
    }


GEOM_OUT = os.environ.get("GEOM_OUT", "")


def geometric_breakdown(model, batch, device, arm: str) -> list[dict]:
    """Internals of a `geometric_basis_profile` surrogate on real shoeboxes."""
    import torch

    from integrator.model.distributions.profile_geometry import geometry_features

    qp = model.surrogates["qp"] if hasattr(model, "surrogates") else None
    if qp is None or not hasattr(qp, "extent_head"):
        return []
    seen: dict = {}

    def hook(mod, args, kwargs, out):
        x = args[0]
        md = kwargs.get("metadata")
        loc = mod.latent_gain * mod.loc_head(x)
        scale = mod.latent_gain * torch.nn.functional.softplus(mod.scale_head(x)) + 1e-6
        feats, _ = geometry_features(md, mod.beam_center, mod.lambda_min, mod.lambda_max)
        feats = feats.to(x.dtype).to(x.device)
        seen["h"] = loc.norm(dim=-1)
        seen["scale"] = scale.mean(dim=-1)
        seen["m"] = (mod.shift_head(feats).norm(dim=-1) if mod.shift_head is not None
                     else torch.zeros_like(seen["h"]))
        if mod.prior_scale_head is not None:
            ls = mod.prior_scale_head(feats).clamp(-mod.max_log_prior_scale, mod.max_log_prior_scale)
            seen["sigma"] = (math.log(mod.prior_scale) + ls).exp().mean(dim=-1)
        if mod.extent_head is not None:
            le = mod.extent_head(feats)
            seen["ext_par"], seen["ext_perp"] = le[:, 0], le[:, 1]
            sig = kwargs.get("shape_signal")
            if mod.extent_residual_head is not None:
                g = mod._extent_gate(sig, loc).reshape(-1)
                r = mod.extent_residual_head(x)
                seen["gate"] = g
                seen["res_par"], seen["res_perp"] = g * r[:, 0], g * r[:, 1]
            if sig is not None:
                seen["signal"] = sig.reshape(-1).to(x.dtype)
        seen["n_eff"] = 1.0 / (out.mean_profile.reshape(out.mean_profile.shape[0], -1) ** 2).sum(-1)

    import math

    h = qp.register_forward_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            counts, shoebox, mask, metadata = batch
            metadata, _ = ensure_group_label(dict(metadata), device)
            model(counts, shoebox, mask, metadata)
    finally:
        h.remove()
    if not seen:
        return []
    n = seen["h"].shape[0]
    sig = seen.get("signal")
    groups = [("all", torch.ones(n, dtype=torch.bool, device=seen["h"].device))]
    if sig is not None:
        q = torch.quantile(sig.float(), torch.tensor([0.25, 0.5, 0.75], device=sig.device))
        edges = [-float("inf"), *q.tolist(), float("inf")]
        groups += [(f"signal_q{i + 1}", (sig > edges[i]) & (sig <= edges[i + 1])) for i in range(4)]
    rows = []
    for name, sel in groups:
        row = {"method": arm, "dataset": DATASET, "group": name, "n": int(sel.sum())}
        for k, v in seen.items():
            row[f"{k}_median"] = round(float(v[sel].float().median()), 5)
        rows.append(row)
    return rows


def one_batch(cfg, device):
    """One real batch, in whatever shape the data module yields it.

    Passed to the model unchanged rather than unpacked: the batch layout is
    the data module's business and differs between the monochromatic and
    polychromatic modules.
    """
    from integrator.utils.factory_utils import construct_data_loader

    dm = construct_data_loader(cfg)
    dm.setup("fit")
    for batch in dm.train_dataloader():
        if isinstance(batch, (list, tuple)):
            return [b.to(device) if hasattr(b, "to") else b for b in batch]
        return batch.to(device) if hasattr(batch, "to") else batch
    return None


def main() -> int:
    if not DATASET:
        raise SystemExit(
            "set DATASET to the dataset these runs trained on, e.g.\n"
            "    DATASET=pdz2_1116 python dump_encoder_health.py out.csv <run>"
        )
    if len(sys.argv) < 3:
        raise SystemExit(__doc__.strip().splitlines()[-1])

    import torch
    import yaml

    from integrator.utils.factory_utils import construct_integrator

    def read_config_log(run_dir: str) -> dict:
        """The run's own config, read here rather than imported.

        `integrator.utils.hparams` is new and the cluster environment's
        installed package predates it. This script has to run against THAT
        package, because the checkpoints it loads were trained with it and
        a model built from newer code is not the model that was trained.
        """
        path = os.path.join(run_dir, "config_log.yaml")
        if not os.path.isfile(path):
            return {}
        with open(path) as fh:
            return yaml.safe_load(fh) or {}

    out_path, runs = sys.argv[1], sys.argv[2:]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for run in runs:
        arm = os.path.basename(run.rstrip("/"))
        cfg = read_config_log(run)
        if not cfg:
            print(f"  {arm}: no config_log.yaml")
            continue
        ckpts = checkpoints(run)
        if not ckpts:
            print(f"  {arm}: no checkpoints naming an epoch")
            continue
        batch = one_batch(cfg, device)
        if batch is None:
            print(f"  {arm}: no batch from its data_dir")
            continue
        model = construct_integrator(cfg).to(device).eval()
        for epoch, path in ckpts:
            state = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(state.get("state_dict", state), strict=False)
            got = measure(model, batch, device)
            if got is None:
                print(f"  {arm}: no profile encoder with a final linear")
                break
            rows.append({"method": arm, "dataset": DATASET, "epoch": epoch,
                         **got})
        if GEOM_OUT and ckpts:
            got = geometric_breakdown(model, batch, device, arm)
            for r in got:
                print("    " + ", ".join(f"{k} {v}" for k, v in r.items() if k not in ("method", "dataset")))
            if got:
                new = not os.path.exists(GEOM_OUT)
                with open(GEOM_OUT, "a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(got[0]))
                    if new:
                        w.writeheader()
                    w.writerows(got)
        if rows and rows[-1]["method"] == arm:
            last = rows[-1]
            print(f"  {arm}: activation {last['activation']}, "
                  f"{last['n_units']} units, dead "
                  f"{last['dead_fraction']:.1%} at epoch {last['epoch']} "
                  f"(pre-activation mean {last['preact_mean']:+.3f})")

    if not rows:
        raise SystemExit("nothing measured")
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        w.writeheader()
        w.writerows(rows)
    print(f"{out_path}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
