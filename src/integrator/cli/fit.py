"""Integrate a dataset by direct profile fitting, with no encoder and no training.

Reads the same YAML a training run uses, takes only the `surrogates.qp` block and
the data loader from it, fits the profile's global shape parameters by maximum
likelihood, then solves every reflection's intensity and background in closed
form. It writes the same `pred.parquet` schema `integrator.cli.predict` writes,
so the downstream chain -- `write_mtz_from_preds`, careless, `phenix` -- is
unchanged.

The config's encoders, `qi`, `qbg`, loss and trainer blocks are ignored. Nothing
is loaded from a checkpoint and nothing is written to one.

    python -m integrator.cli.fit --config run/config_log.yaml \
        --out-dir $S/direct/physical --write-mtz

`--resident` replaces the data module with `integrator.fit.resident`: counts and
masks go to the device once and the metadata is read straight from the reference
file. On hewl_1118 the streaming path spends 151 s building the data module and
61.7 s on the pass, against 73.7 s of actual estimator -- 12.0 s to fit the four
globals and 61.7 s for the closed-form solve over 2,664,858 reflections -- and
the pass itself is bound by the dataloader and the host-to-device copy rather
than by arithmetic. Row order is file order on both paths and the predict
dataloader does not shuffle, so the two give the same reflections in the same
order; `min_valid_pixels` is read from the config so they filter alike.

See `integrator.fit.direct` for the estimator and
`docs/rb_export_2026-11-12.md` for why sigma is taken from the Fisher
information rather than from a variational posterior.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("integrator.cli.fit")

# carried through to the parquet when the loader supplies them; everything the
# MTZ writer and the downstream diagnostics ask for
_METADATA_COLUMNS = (
    "refl_ids",
    "is_test",
    "d",
    "wavelength",
    "H",
    "K",
    "L",
    "intensity.sum.value",
    "intensity.sum.variance",
    "background.sum.value",
    "background.sum.variance",
)

# what `--resident` holds on the device: the parquet's columns plus the geometry
# `PhysicalProfileSurrogate._profile` reads. `d` and `wavelength` serve both.
# Every one of these is float32 in `metadata.pt`, so holding them costs
# 4 bytes x n_refl each and the parquet schema comes out identical to the
# streaming path's.
_RESIDENT_KEYS = _METADATA_COLUMNS + (
    "xyzcal.px.0",
    "xyzcal.px.1",
    "image_num",
)

# parquet names for the fitted centre, in the order the surrogate returns them.
# A still has the first two; a rotation series adds the rocking axis.
_SHIFT_NAMES = ("mu_radial", "mu_tangential", "mu_frame")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Direct profile-fit integration: no encoder, no SGD."
    )
    p.add_argument("--config", required=True, help="training YAML to read")
    p.add_argument("--out-dir", required=True, help="where pred.parquet goes")
    p.add_argument(
        "--bg-basis",
        default="flat",
        choices=("flat", "plane"),
        help="background model inside the shoebox (default: flat, as the "
        "current model has)",
    )
    p.add_argument(
        "--fit-batches",
        type=int,
        default=8,
        help="batches held for the global shape fit (default: 8)",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=60,
        help="L-BFGS iterations for the global fit (default: 60)",
    )
    p.add_argument(
        "--inner-iter",
        type=int,
        default=5,
        help="IRLS passes per reflection (default: 5)",
    )
    p.add_argument(
        "--globals",
        default=None,
        help="JSON file of previously fitted globals; skips the fit",
    )
    p.add_argument(
        "--no-fit",
        action="store_true",
        help="use the config's initial globals without fitting them",
    )
    p.add_argument(
        "--recenter",
        action="store_true",
        help="fit each reflection's centre as well as its intensity, instead "
        "of pinning the profile to the middle of the shoebox. Adds the two "
        "centre-derivative columns to the same solve, so the cost is a few "
        "extra passes and no extra fitted parameters.",
    )
    p.add_argument(
        "--recenter-iter",
        type=int,
        default=3,
        help="outer passes for --recenter; the expansion is local, so one pass "
        "is a single linearized step (default: 3)",
    )
    p.add_argument(
        "--no-recenter-shrink",
        action="store_true",
        help="turn off the empirical-Bayes shrinkage of the shift. On by "
        "default, because an unshrunk centre from a weak reflection is two "
        "parameters out of a handful of photons.",
    )
    p.add_argument(
        "--max-shift",
        type=float,
        default=3.0,
        help="clamp on the fitted shift, in pixels (default: 3.0)",
    )
    p.add_argument(
        "--recenter-bins",
        type=int,
        default=0,
        help="group the shrinkage scale by image, in this many equal-count "
        "bins of image_num; 0 fits one scale per direction for the whole "
        "dataset (default: 0)",
    )
    p.add_argument(
        "--resident",
        action="store_true",
        help="hold counts and masks on the device and read the metadata "
        "straight from the reference file, instead of building the data "
        "module and streaming through its dataloader. Costs counts + masks in "
        "device memory (8.3 GB on hewl_1118) and skips the standardized "
        "shoebox tensor, which this estimator never reads.",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=65536,
        help="reflections per device chunk under --resident (default: 65536)",
    )
    p.add_argument("--write-mtz", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _shift_bins(metadata, n_bins):
    """Equal-count bins of `image_num`, for the shift shrinkage scale.

    Centroid error drifts with the crystal rather than with the reflection, so
    grouping by image is the motivated choice when the dataset is large enough
    to support a scale per group. Returns `None` for one global scale, which is
    also what happens when the loader carries no `image_num`.
    """
    import torch

    if int(n_bins) <= 1 or "image_num" not in metadata:
        return None
    img = metadata["image_num"].reshape(-1).float()
    q = torch.quantile(
        img, torch.linspace(0, 1, int(n_bins) + 1, device=img.device)
    )
    return torch.bucketize(img, q[1:-1].contiguous()).clamp_(0, int(n_bins) - 1)


def _to_device(batch, device):
    import torch

    counts, shoebox, mask, metadata = batch
    md = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in metadata.items()
    }
    return counts.to(device), mask.to(device), md


def main(argv=None):
    import numpy as np
    import polars as pl
    import torch
    import yaml

    from integrator.fit import (
        background_basis,
        fit_globals,
        solve_profile_fit,
        solve_profile_fit_recentered,
    )
    from integrator.registry import REGISTRY
    from integrator.utils import construct_data_loader, inject_binning_labels

    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    torch.set_float32_matmul_precision("high")

    config = yaml.safe_load(Path(args.config).read_text())
    qp_cfg = config["surrogates"]["qp"]
    if qp_cfg["name"] != "physical_profile":
        raise SystemExit(
            f"--config must use surrogates.qp.name: physical_profile; this one "
            f"is {qp_cfg['name']!r}. The direct fit needs a profile that is a "
            "function of geometry and a few globals, not of an encoder output."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    surrogate = REGISTRY["surrogates"]["physical_profile"](
        **dict(qp_cfg.get("args", {}))
    ).to(device)
    shape = tuple(surrogate.profile_shape)
    logger.info(
        "profile %s, %d fitted globals",
        shape,
        sum(p.numel() for p in surrogate.parameters() if p.requires_grad),
    )

    dl_args = config["data_loader"]["args"]
    batch_size = int(dl_args.get("batch_size") or 512)

    # `batches()` is the full pass; `head_batches(n)` is the sample the global
    # shape fit sees. Under --resident both read slices of the device tensors,
    # and the head slices are `batch_size` wide so the fit sees the same number
    # of reflections it would have streamed.
    if args.resident:
        from integrator.fit.resident import ResidentShoeboxes

        names = dl_args.get("shoebox_file_names") or {}
        boxes = ResidentShoeboxes(
            names.get("data_dir", dl_args["data_dir"]),
            device,
            chunk=args.chunk,
            min_valid_pixels=int(dl_args.get("min_valid_pixels") or 0),
            counts_file=names.get("counts", "counts.npy"),
            masks_file=names.get("masks", "masks.npy"),
            reference_file=names.get("reference", "metadata.pt"),
            metadata_keys=_RESIDENT_KEYS,
        )
        missing = [k for k in _RESIDENT_KEYS if k not in boxes.metadata]
        if missing:
            logger.warning(
                "reference file has no %s; those parquet columns will be absent",
                ", ".join(missing),
            )

        def batches():
            for _idx, counts, mask, md in boxes():
                yield counts, mask, md

        def head_batches(n):
            for i in range(int(n)):
                lo, hi = i * batch_size, min((i + 1) * batch_size, len(boxes))
                if lo >= hi:
                    return
                yield (
                    boxes.counts[lo:hi],
                    boxes.mask[lo:hi],
                    {k: v[lo:hi] for k, v in boxes.metadata.items()},
                )
    else:
        data_loader = construct_data_loader(config)
        data_loader.setup()
        inject_binning_labels(data_loader, config)
        loader = data_loader.predict_dataloader()

        def batches():
            for batch in loader:
                yield _to_device(batch, device)

        def head_batches(n):
            for i, batch in enumerate(loader):
                if i >= int(n):
                    return
                yield _to_device(batch, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- globals
    t0 = time.time()
    if args.globals:
        vals = json.loads(Path(args.globals).read_text())
        with torch.no_grad():
            for name in ("a", "b", "c", "z"):
                getattr(surrogate, f"log_{name}").fill_(float(np.log(vals[name])))
        logger.info("globals loaded from %s: %s", args.globals, vals)
        fit_info = {**vals, "source": args.globals}
    elif args.no_fit:
        fit_info = {
            n: float(getattr(surrogate, f"log_{n}").exp()) for n in "abcz"
        }
        logger.info("globals left at their config values: %s", fit_info)
    else:
        held = list(head_batches(args.fit_batches))
        logger.info(
            "fitting 4 globals on %d reflections from %d batches",
            sum(c.shape[0] for c, _, _ in held),
            len(held),
        )
        fit_info = fit_globals(
            surrogate,
            [(c, m, md) for c, m, md in held],
            bg_kind=args.bg_basis,
            max_iter=args.max_iter,
            n_iter_inner=args.inner_iter,
        )
        del held
        logger.info(
            "globals: a=%.2f b=%.3f c=%.3f z=%.3f  loglik %.4f -> %.4f (%d evals)",
            fit_info["a"],
            fit_info["b"],
            fit_info["c"],
            fit_info["z"],
            fit_info["loglik_before"],
            fit_info["loglik_after"],
            fit_info["n_eval"],
        )
    t_fit = time.time() - t0

    # ------------------------------------------------------------- full pass
    t0 = time.time()
    bg_cols = background_basis(shape, args.bg_basis, device, torch.float32)
    # The clamp has to be per direction. On 9b7c the shoebox is 3 x 21 x 21, so
    # the rocking axis has one frame of room either side against ten pixels in
    # the detector plane; a single scalar would let the profile walk out of the
    # box along the short axis. Half the box extent is the limit past which the
    # profile's mass is mostly outside the data.
    n_dir = surrogate.n_shift_directions
    shift_cap = [args.max_shift] * n_dir
    if n_dir > 2:
        shift_cap[2] = min(args.max_shift, max((shape[0] - 1) / 2.0, 0.5))
    if args.recenter:
        logger.info(
            "recentring %d directions, clamp %s px, %d outer passes, shrink %s",
            n_dir,
            [round(c, 2) for c in shift_cap],
            args.recenter_iter,
            not args.no_recenter_shrink,
        )
    chunks: list[dict] = []
    n_refl = 0
    with torch.no_grad():
        for counts, mask, md in batches():
            if args.recenter:
                res, mu = solve_profile_fit_recentered(
                    counts,
                    mask,
                    lambda off, md=md, mask=mask: surrogate.profile_and_grad(
                        md, mask, off
                    ),
                    bg_cols,
                    n_iter=args.inner_iter,
                    n_outer=args.recenter_iter,
                    shrink=not args.no_recenter_shrink,
                    max_shift=shift_cap,
                    bin_index=_shift_bins(md, args.recenter_bins),
                    n_bins=max(1, args.recenter_bins),
                )
            else:
                profile = surrogate._profile(md, mask)
                res = solve_profile_fit(
                    counts, mask, profile, bg_cols, n_iter=args.inner_iter
                )
                mu = None
            row = {
                "qi_mean": res.intensity,
                "qi_var": res.variance,
                "qbg_mean": res.background,
                "qbg_var": res.background_variance,
                "i_rb": res.intensity,
                "var_rb": res.variance,
                "n_eff": res.n_eff,
            }
            if mu is not None:
                # written out so the shifts can be looked at against image and
                # resolution, which is how a geometry problem is told apart
                # from a profile one. A rotation series carries a third
                # direction, the rocking axis.
                for j, name in enumerate(_SHIFT_NAMES[: mu.shape[1]]):
                    row[name] = mu[:, j]
            out = {k: v.detach().cpu().numpy().ravel() for k, v in row.items()}
            for col in _METADATA_COLUMNS:
                if col in md:
                    out[col] = md[col].detach().cpu().numpy().ravel()
            chunks.append(out)
            n_refl += res.intensity.shape[0]
    t_pass = time.time() - t0

    frame = pl.DataFrame(
        {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    ).with_columns(pl.lit(0).alias("epoch"))
    pred_path = out_dir / "pred.parquet"
    frame.write_parquet(pred_path)
    logger.info("wrote %s (%d reflections)", pred_path, frame.height)

    i = frame["qi_mean"].to_numpy()
    s = np.sqrt(frame["qi_var"].to_numpy())
    good = np.isfinite(i) & np.isfinite(s) & (s > 0)
    logger.info(
        "direct fit: median I %.2f  median sigma %.2f  IQR log sigma %.2f  "
        "negative I %.1f%%",
        float(np.median(i[good])),
        float(np.median(s[good])),
        float(np.subtract(*np.percentile(np.log(s[good]), [75, 25]))),
        100.0 * float((i < 0).mean()),
    )
    logger.info(
        "timing: globals %.1f s, full pass %.1f s, %d reflections",
        t_fit,
        t_pass,
        n_refl,
    )
    (out_dir / "globals.json").write_text(json.dumps(fit_info, indent=2))

    if args.write_mtz:
        from integrator.cli.predict import _reference_file
        from integrator.io import get_pred_files, write_mtz_from_preds

        data_dir = Path(dl_args["data_dir"])
        pred_data = get_pred_files(
            ckpt_dir=out_dir, filetype="parquet", extra_columns=("i_rb", "var_rb")
        )
        mtz_path = out_dir / "preds_epoch_0000.mtz"
        write_mtz_from_preds(
            pred_data=pred_data,
            metadata_path=data_dir / _reference_file(config),
            data_dir=data_dir,
            out_path=mtz_path,
            intensity_key="i_rb",
            variance_key="var_rb",
            # a monochromatic dataset has no wavelength column; the surrogate
            # already needed the scalar to form sin(theta), so reuse it
            wavelength=qp_cfg.get("args", {}).get("wavelength"),
        )
        logger.info("wrote %s", mtz_path)


if __name__ == "__main__":
    main()
