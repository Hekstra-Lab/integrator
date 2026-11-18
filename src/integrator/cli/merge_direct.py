"""Merge a dataset from pixels to structure factors, with no training step.

Reads the same YAML a training run uses, takes the profile's four global shape
parameters from `integrator.cli.fit` (or fits them here), then runs the block
sweep in `integrator.fit.sweep`: responsibilities, backgrounds, per-image
scales, per-pair intensities and Bijvoet differences, per-shell Wilson scales,
all in closed form. Writes a MERGED MTZ carrying `I(+)/SigI(+)`, `I(-)/SigI(-)`
and the French-Wilson-free amplitudes derived from them, so it can go straight
into `phenix.refine`.

There is no intermediate per-reflection intensity and no second program: the
sigma is the observed-data Fisher information of the merged quantity itself.

    python -m integrator.cli.merge_direct --config run/config_log.yaml \
        --globals $S/fulldirect/physical/epoch_0000/globals.json \
        --out-dir $S/fullmerge/physical --sweeps 15

See `docs/rb_export_2026-11-12.md` for why sigma is taken from the information
matrix rather than from a variational posterior.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("integrator.cli.merge_direct")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Closed-form pixels-to-F merge.")
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--globals", default=None, help="globals.json from integrator.cli.fit"
    )
    p.add_argument("--sweeps", type=int, default=15)
    p.add_argument("--bins", type=int, default=20, help="resolution shells")
    p.add_argument(
        "--tau-scale",
        type=float,
        default=None,
        help="PC prior: the anomalous signal exceeds this FRACTION of the "
        "intensity with probability --tau-prob. Omitted gives the empirical "
        "Bayes estimate with no prior on the scale.",
    )
    p.add_argument("--tau-prob", type=float, default=0.01)
    p.add_argument("--dmin", type=float, default=None)
    p.add_argument(
        "--cheb",
        nargs=2,
        type=int,
        default=None,
        metavar=("DEG_IMAGE", "DEG_S2"),
        help="smooth Chebyshev scale in (image, s^2) instead of one free "
        "scalar per image. Columns constant in image and varying in s^2 are "
        "dropped as gauge.",
    )
    p.add_argument(
        "--lambda-bins",
        type=int,
        default=0,
        help="wavelength bins for G(lambda). An HKL is measured at many "
        "wavelengths, so the spectrum is a per-observation factor the "
        "per-image scale cannot represent. 0 leaves it out.",
    )
    p.add_argument(
        "--anchor",
        action="store_true",
        help="measure the fake anomalous difference on centric reflections, "
        "where the truth is zero. Costs one extra responsibility pass.",
    )
    p.add_argument(
        "--resident",
        action="store_true",
        help="hold the shoeboxes on the device. Removes the host-to-device "
        "copy from every sweep pass and the separate indexing pass, at the "
        "cost of counts + masks in device memory (8.3 GB on hewl_1118).",
    )
    p.add_argument("--chunk", type=int, default=65536)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    import numpy as np
    import reciprocalspaceship as rs
    import torch
    import yaml

    from integrator.fit.merge import scale_design
    from integrator.fit.sweep import (
        equal_count_bins,
        friedel_pseudo_mate,
        pair_index,
        run_sweep,
    )
    from integrator.io import read_dataset_spec
    from integrator.registry import REGISTRY
    from integrator.utils import construct_data_loader

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
            "--config must use surrogates.qp.name: physical_profile; the sweep "
            "needs a profile that is a function of geometry and a few globals."
        )
    data_dir = Path(config["data_loader"]["args"]["data_dir"])
    spec = read_dataset_spec(data_dir) or {}
    crystal = spec["crystal"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surrogate = REGISTRY["surrogates"]["physical_profile"](
        **dict(qp_cfg.get("args", {}))
    ).to(device)
    if args.globals:
        vals = json.loads(Path(args.globals).read_text())
        with torch.no_grad():
            for name in ("a", "b", "c", "z"):
                getattr(surrogate, f"log_{name}").fill_(float(np.log(vals[name])))
        logger.info("profile globals from %s: %s", args.globals, vals)
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad_(False)

    t0 = time.time()
    if args.resident:
        from integrator.fit.resident import ResidentShoeboxes

        dl_args = config["data_loader"]["args"]
        names = dl_args.get("shoebox_file_names") or {}
        boxes = ResidentShoeboxes(
            data_dir,
            device,
            chunk=args.chunk,
            min_valid_pixels=int(dl_args.get("min_valid_pixels") or 0),
            counts_file=names.get("counts", "counts.npy"),
            masks_file=names.get("masks", "masks.npy"),
            reference_file=names.get("reference", "metadata.pt"),
        )
        md_all = boxes.metadata
        hkl = (
            torch.stack([md_all["H"], md_all["K"], md_all["L"]], -1)
            .round()
            .int()
            .cpu()
            .numpy()
        )
        dd = md_all["d"].cpu()
        img = md_all["image_num"].cpu().long()
        lam = md_all["wavelength"].cpu()
        n_obs = len(boxes)
        batches = boxes
        logger.info(
            "resident load: %d observations in %.1f s", n_obs, time.time() - t0
        )
        expected = spec.get("n_reflections")
        if expected and int(expected) != n_obs:
            logger.warning(
                "resident holds %d observations but dataset.yaml says %d; the "
                "streaming path may be filtering differently",
                n_obs, int(expected),
            )
    else:
        dm = construct_data_loader(config)
        dm.setup()

        # one pass to collect the per-observation indices the sweep needs
        rows, hkl, dd, img, lam = [], [], [], [], []
        for counts, _shoebox, _mask, md in dm.predict_dataloader():
            del counts
            rows.append(md["refl_ids"].reshape(-1).long())
            hkl.append(
                torch.stack([md["H"], md["K"], md["L"]], -1)
                .reshape(-1, 3)
                .round()
                .int()
            )
            dd.append(md["d"].reshape(-1).float())
            img.append(md["image_num"].reshape(-1).long())
            lam.append(md["wavelength"].reshape(-1).float())
        rows = torch.cat(rows)
        hkl = torch.cat(hkl).numpy()
        dd = torch.cat(dd)
        img = torch.cat(img)
        lam = torch.cat(lam)
        n_obs = int(rows.numel())
        logger.info("indexed %d observations in %.1f s", n_obs, time.time() - t0)

        # dense lookup from reflection id to a row of the per-observation state.
        # A Python dict here would cost one lookup per observation per pass,
        # which is 43 million of them over a 16-pass sweep; this is a gather.
        pos_of_reflid = torch.full((int(rows.max()) + 1,), -1, dtype=torch.long)
        pos_of_reflid[rows] = torch.arange(n_obs)

        def batches():
            for counts, _shoebox, mask, md in dm.predict_dataloader():
                idx = pos_of_reflid[md["refl_ids"].reshape(-1).long()]
                yield idx, counts, mask, {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in md.items()
                }

    import gemmi

    # dataset.yaml writes the name with a trailing "(No. 96)", which gemmi
    # will not parse, and carries the number separately. Prefer the number.
    if crystal.get("space_group_number"):
        sg = gemmi.find_spacegroup_by_number(int(crystal["space_group_number"]))
    else:
        sg = gemmi.SpaceGroup(
            str(crystal.get("space_group", "P 1")).split("(")[0].strip()
        )
    logger.info("space group %s (%d)", sg.hm, sg.number)
    pair, mate, centric_pair, n_pair = pair_index(hkl, sg)
    pseudo = friedel_pseudo_mate(hkl, sg) if args.anchor else None
    groups, group = torch.unique(img, return_inverse=True)
    bins_obs = equal_count_bins(dd, args.bins)
    lambda_bin = (
        equal_count_bins(lam, args.lambda_bins) if args.lambda_bins > 0 else None
    )
    cheb = None
    if args.cheb:
        u = img.to(torch.float32)
        u = (u - u.min()) / (u.max() - u.min()).clamp_min(1.0)
        s2 = 1.0 / (4.0 * dd.to(torch.float32).clamp_min(1e-6) ** 2)
        cheb, terms = scale_design(u, s2, args.cheb[0], args.cheb[1])
        logger.info(
            "Chebyshev scale: degrees %d x %d, %d identifiable columns "
            "(dropped %d gauge)",
            args.cheb[0], args.cheb[1], cheb.shape[1],
            (args.cheb[0] + 1) * (args.cheb[1] + 1) - cheb.shape[1],
        )
    # a pair's shell is the shell of its observations; they agree by construction
    bin_pair = torch.zeros(n_pair, dtype=torch.long)
    bin_pair[pair] = bins_obs
    logger.info(
        "%d Miller pairs, %d images, %d shells; %d centric",
        n_pair, groups.numel(), args.bins, int(centric_pair.sum()),
    )

    def profile_fn(md, mask):
        return surrogate._profile(md, mask)

    t0 = time.time()
    res = run_sweep(
        batches,
        profile_fn=profile_fn,
        pair=pair,
        mate=mate,
        group=group,
        bin_index=bin_pair,
        centric_pair=centric_pair,
        n_pair=n_pair,
        n_groups=int(groups.numel()),
        n_bins=args.bins,
        n_obs=n_obs,
        n_sweeps=args.sweeps,
        lambda_bin=lambda_bin,
        n_lambda=args.lambda_bins,
        cheb_design=cheb,
        pseudo_mate=pseudo,
        tau_scale=args.tau_scale,
        tau_prob=args.tau_prob,
        device=device,
    )
    logger.info("sweep: %d passes in %.1f s", args.sweeps, time.time() - t0)
    if res.state.spectrum is not None:
        g = res.state.spectrum.detach().cpu()
        logger.info(
            "G(lambda) over %d bins, %.3f A to %.3f A: %s",
            args.lambda_bins, float(lam.min()), float(lam.max()),
            " ".join(f"{float(x):.3f}" for x in g),
        )

    # the sweep is the expensive part; checkpoint it before anything that can
    # fail downstream, so a writer bug does not cost twenty minutes of passes
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "merged": res.state.merged.detach().cpu(),
            "delta": res.state.delta.detach().cpu(),
            "delta_shrunk": res.delta_shrunk.detach().cpu(),
            "var_merged": res.var_merged.detach().cpu(),
            "var_delta": res.var_delta.detach().cpu(),
            "scale": res.state.scale.detach().cpu(),
            "spectrum": (
                None if res.state.spectrum is None
                else res.state.spectrum.detach().cpu()
            ),
            "sigma_d": res.state.sigma_d.detach().cpu(),
            "tau": res.tau.detach().cpu(),
            "pair": pair,
            "mate": mate,
            "centric_pair": centric_pair,
            "bin_pair": bin_pair,
        },
        out_dir / "sweep_state.pt",
    )
    logger.info("checkpointed the sweep to %s", out_dir / "sweep_state.pt")

    j = res.state.merged.detach().cpu().numpy().astype("float64")
    delta = res.delta_shrunk.detach().cpu().numpy().astype("float64")
    vj = res.var_merged.detach().cpu().numpy().astype("float64")
    vd = res.var_delta.detach().cpu().numpy().astype("float64")
    tau = res.tau.detach().cpu().numpy()
    logger.info(
        "per-shell anomalous fraction tau: %s",
        " ".join(f"{t:.4f}" for t in tau),
    )

    # canonical Miller index of each pair, taken from its first observation
    pair_np = pair.numpy()
    first = np.zeros(n_pair, dtype=np.int64)
    first[pair_np[::-1]] = np.arange(len(pair_np))[::-1]
    from integrator.utils.asu import canonical_hkl

    pooled = canonical_hkl(hkl, sg, anomalous=False)[first]

    ip, im = j + 0.5 * delta, j - 0.5 * delta
    var_p = vj + 0.25 * vd
    var_m = var_p
    keep = np.isfinite(var_p) & (var_p > 0) & np.isfinite(j) & (j > 0)
    if args.dmin is not None:
        # take each pair's resolution from its own first observation rather
        # than recomputing it: `gemmi.Miller` does not exist in the Python API,
        # and the per-observation `d` is already loaded and exact.
        res_pair = dd.numpy().astype("float64")[first]
        keep &= res_pair >= args.dmin

    ds = rs.DataSet(
        {
            "H": pooled[keep, 0], "K": pooled[keep, 1], "L": pooled[keep, 2],
            "I(+)": ip[keep], "SIGI(+)": np.sqrt(var_p[keep]),
            "I(-)": im[keep], "SIGI(-)": np.sqrt(var_m[keep]),
        },
        cell=gemmi.UnitCell(*crystal["cell"]),
        spacegroup=sg,
        merged=True,
    ).infer_mtz_dtypes()
    ds.set_index(["H", "K", "L"], inplace=True)

    mtz_path = out_dir / "merged.mtz"
    ds.write_mtz(str(mtz_path))
    logger.info("wrote %s (%d Miller pairs)", mtz_path, int(keep.sum()))

    (out_dir / "sweep.json").write_text(
        json.dumps(
            {
                "n_pair": int(n_pair),
                "n_kept": int(keep.sum()),
                "n_images": int(groups.numel()),
                "sweeps": args.sweeps,
                "tau": [float(t) for t in tau],
                "cheb_coeff": (
                    [float(x) for x in res.state.cheb_coeff.detach().cpu()]
                    if res.state.cheb_coeff is not None
                    else None
                ),
                "spectrum": (
                    [float(x) for x in res.state.spectrum.detach().cpu()]
                    if res.state.spectrum is not None
                    else None
                ),
                "anchor": (
                    {
                        "n": res.anchor["n"],
                        "n_usable": res.anchor["n_usable"],
                        "z_sd": res.anchor["z_sd"],
                        "tau_centric": [float(t) for t in res.anchor["tau"]],
                        "fraction_of_tau": res.anchor.get("fraction_of_tau"),
                    }
                    if res.anchor is not None
                    else None
                ),
                "history": res.state.history,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
