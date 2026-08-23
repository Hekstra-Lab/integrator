"""Build the training figures for a run by replaying its checkpoints.

Works on runs that are already finished: every epoch checkpoint is loaded
in turn, a fixed set of tracked shoeboxes is pushed through it, and the
profile decoder is snapshotted, producing exactly the dumps that the
`--figures` training callbacks would have written. The final checkpoint
also supplies the latent-space and model-check figures.

Usage:
    uv run python scripts/make_training_figures.py --run-dir <run-dir>

    # a subset of the work, on a specific set of checkpoints
    uv run python scripts/make_training_figures.py \
        --config config_log.yaml --ckpt <dir-of-ckpts> \
        --stages tracked,basis --stride 2 --no-animate

    # re-render figures from dumps that already exist
    uv run python scripts/make_training_figures.py \
        --run-dir <run-dir> --render-only
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import torch

from integrator.reporting.figure_data import (
    BasisRecorder,
    LatentRecorder,
    TrackedRecorder,
    dials_snr,
    select_tracked,
)
from integrator.reporting.plot_jobs import (
    render_basis,
    render_checks,
    render_explain,
    render_latent,
    render_tracked,
)

logger = logging.getLogger(__name__)

ALL_STAGES = ("tracked", "basis", "latent", "checks", "explain")
EPOCH_RE = re.compile(r"epoch[=_](\d+)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Render training figures from a run's checkpoints"
    )
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="A .ckpt file or a directory of them (overrides --run-dir)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Where dumps and figures go (default: <output_root>/figures)",
    )
    p.add_argument(
        "--stages",
        type=str,
        default=",".join(ALL_STAGES),
        help=f"Comma-separated subset of {ALL_STAGES}",
    )
    p.add_argument("--n-per-regime", type=int, default=4)
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every n-th checkpoint (the last one is always kept)",
    )
    p.add_argument(
        "--latent-points",
        type=int,
        default=20000,
        help="Cap on reflections used for the latent-space figures",
    )
    p.add_argument(
        "--check-pixels",
        type=int,
        default=2_000_000,
        help="Cap on pixels kept for the posterior predictive check",
    )
    p.add_argument("--n-clusters", type=int, default=6)
    p.add_argument(
        "--formats",
        type=str,
        default="png,pdf",
        help="Comma-separated figure formats",
    )
    p.add_argument("--no-animate", action="store_true")
    p.add_argument(
        "--render-only",
        action="store_true",
        help="Skip inference and re-render from existing dumps",
    )
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def resolve_sources(args) -> tuple[dict, list[Path], Path]:
    """Return `(config, checkpoints, out_dir)` from --run-dir and overrides."""
    import yaml

    from integrator.utils import apply_dataset_defaults, load_config

    meta: dict = {}
    if args.run_dir:
        meta = yaml.safe_load(
            (Path(args.run_dir) / "run_paths.yaml").read_text()
        )
    config_path = args.config or meta.get("config")
    if not config_path:
        raise SystemExit("provide --config or --run-dir")
    config = apply_dataset_defaults(load_config(config_path))

    if args.ckpt:
        path = Path(args.ckpt)
        checkpoints = (
            [c for c in sorted(path.glob("**/*.ckpt")) if c.name != "last.ckpt"]
            if path.is_dir()
            else [path]
        )
    elif meta:
        log_dir = Path(meta.get("log_dir") or meta["wandb"]["log_dir"])
        checkpoints = sorted(log_dir.glob("**/epoch*.ckpt"))
    else:
        raise SystemExit("provide --ckpt or --run-dir")

    checkpoints = sorted(checkpoints, key=epoch_of)
    if args.stride > 1 and len(checkpoints) > 2:
        kept = checkpoints[:: args.stride]
        if checkpoints[-1] not in kept:
            kept.append(checkpoints[-1])
        checkpoints = kept

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif meta.get("figures_dir"):
        out_dir = Path(meta["figures_dir"])
    elif meta.get("output_root"):
        out_dir = Path(meta["output_root"]) / "figures"
    elif checkpoints:
        out_dir = checkpoints[0].parent.parent / "figures"
    else:
        out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    return config, checkpoints, out_dir


def epoch_of(path: Path) -> int:
    """Epoch number parsed from a checkpoint filename, else -1."""
    match = EPOCH_RE.search(path.name)
    return int(match.group(1)) if match else -1


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def build_tracked_batch(dataset, out_dir: Path, n_per_regime: int):
    """Collate the tracked mini-batch, reusing an existing selection if present."""
    import json

    from torch.utils.data._utils.collate import default_collate

    reference = {k: np.asarray(v) for k, v in dataset.reference.items()}
    selection_path = out_dir / "tracked_selection.json"
    if selection_path.exists():
        stored = json.loads(selection_path.read_text())
        selection = {
            "index": np.asarray(stored["index"], dtype=int),
            "refl_id": np.asarray(stored["refl_id"], dtype=int),
            "regime": list(stored["regime"]),
            "snr": np.asarray(stored["snr"], dtype=float),
            "intensity": np.asarray(stored["intensity"], dtype=float),
            "sigma": np.asarray(stored["sigma"], dtype=float),
        }
        logger.info("reusing tracked selection from %s", selection_path.name)
    else:
        counts = None
        if dials_snr(reference) is None:
            counts = np.asarray(dataset.counts, dtype=np.float32)
        selection = select_tracked(
            reference, counts=counts, n_per_regime=n_per_regime
        )
    batch = default_collate(
        [dataset[int(i)] for i in selection["index"]]
    )
    return selection, batch


def replay_checkpoints(
    config, checkpoints, out_dir: Path, args, dataset, device
):
    """Push the tracked batch through every checkpoint and snapshot the basis."""
    from integrator.utils import construct_integrator

    stages = set(args.stages.split(","))
    selection, batch = build_tracked_batch(
        dataset, out_dir, args.n_per_regime
    )
    integrator = construct_integrator(config).to(device).eval()
    shape = integrator.shoebox_shape

    tracked = (
        TrackedRecorder(
            selection,
            batch[0].float().numpy(),
            batch[2].float().numpy(),
            shape,
        )
        if "tracked" in stages or "explain" in stages
        else None
    )
    basis = BasisRecorder(shape) if "basis" in stages else None
    gpu_batch = _to_device(batch, device)

    for ckpt in checkpoints:
        epoch = epoch_of(ckpt)
        state = torch.load(ckpt.as_posix(), map_location=device)
        integrator.load_state_dict(state["state_dict"])
        integrator.eval()
        logger.info("epoch %d: %s", epoch, ckpt.name)

        if tracked is not None:
            torch.manual_seed(0)
            with torch.no_grad():
                out = integrator(*gpu_batch)["forward_out"]
            tracked.record(epoch, out)
        if basis is not None:
            decoder = getattr(integrator.surrogates["qp"], "decoder", None)
            if decoder is None:
                logger.warning("no learned-basis decoder; skipping basis")
                basis = None
            else:
                basis.record(
                    epoch, decoder.weight.detach(), decoder.bias.detach()
                )

    if tracked is not None:
        tracked.save(out_dir)
    if basis is not None:
        basis.save(out_dir)
    return integrator


def collect_latents_and_checks(
    integrator, data_loader, out_dir: Path, args, device
):
    """One pass over the validation split for the latent and check figures."""
    stages = set(args.stages.split(","))
    want_latent = "latent" in stages
    want_checks = "checks" in stages
    if not (want_latent or want_checks):
        return

    loader = data_loader.val_dataloader()
    if loader is None or len(loader.dataset) == 0:
        loader = data_loader.predict_dataloader()

    recorder = LatentRecorder(max_points=args.latent_points)
    z_parts, rate_parts = [], []
    model_i, dials_i, dials_sigma = [], [], []
    n_pixels = 0

    integrator.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            out = integrator(*batch)["forward_out"]
            metadata = batch[3] if len(batch) > 3 else {}
            if want_latent:
                recorder.add(out, metadata)
            if want_checks:
                rate = out["rates"].mean(1)
                counts = out["counts"].float()
                mask = out["mask"].float()
                if mask.dim() == 3:
                    mask = mask.squeeze(-1)
                sel = mask > 0
                pearson = (counts - rate) / rate.clamp(min=1e-3).sqrt()
                z_parts.append(pearson[sel].cpu().numpy())
                rate_parts.append(rate[sel].cpu().numpy())
                n_pixels += int(sel.sum())
                model_i.append(out["qi_mean"].cpu().numpy().ravel())
                for key, sink in (
                    ("intensity.prf.value", dials_i),
                    ("intensity.prf.variance", dials_sigma),
                ):
                    if key in metadata:
                        sink.append(metadata[key].cpu().numpy().ravel())
            done_latent = (
                not want_latent or recorder.n_points >= args.latent_points
            )
            done_checks = not want_checks or n_pixels >= args.check_pixels
            if done_latent and done_checks:
                break

    if want_latent:
        path = recorder.save(out_dir, epoch=9999)
        if path is not None:
            final = out_dir / "latents_final.parquet"
            path.rename(final)
            logger.info("latents -> %s (%d points)", final.name, recorder.n_points)
            # plot_jobs discovers latents by the epoch glob; keep both names.
            import shutil

            shutil.copyfile(final, out_dir / "latents_epoch_9999.parquet")

    if want_checks and z_parts:
        payload = {
            "z": np.concatenate(z_parts)[: args.check_pixels],
            "rate": np.concatenate(rate_parts)[: args.check_pixels],
        }
        if model_i:
            payload["model_i"] = np.concatenate(model_i)
        if dials_i:
            payload["dials_i"] = np.concatenate(dials_i)
        if dials_sigma:
            payload["dials_sigma"] = np.sqrt(
                np.clip(np.concatenate(dials_sigma), 0, None)
            )
        np.savez_compressed(out_dir / "checks.npz", **payload)
        logger.info("checks -> checks.npz (%d pixels)", len(payload["z"]))


def render(out_dir: Path, args, integrator=None):
    """Render every figure family whose dumps are present."""
    stages = set(args.stages.split(","))
    formats = tuple(args.formats.split(","))
    animate = not args.no_animate
    weight = bias = shape = None
    if integrator is not None:
        decoder = getattr(integrator.surrogates["qp"], "decoder", None)
        if decoder is not None:
            weight = decoder.weight.detach().cpu().numpy()
            bias = decoder.bias.detach().cpu().numpy()
            shape = integrator.shoebox_shape

    jobs = {
        "tracked": lambda: render_tracked(
            out_dir, animate=animate, formats=formats
        ),
        "basis": lambda: render_basis(
            out_dir, animate=animate, formats=formats
        ),
        "explain": lambda: render_explain(out_dir, formats=formats),
        "checks": lambda: render_checks(out_dir, formats=formats),
        "latent": lambda: render_latent(
            out_dir,
            weight=weight,
            bias=bias,
            shape=shape,
            n_clusters=args.n_clusters,
            formats=formats,
        ),
    }
    written = []
    for name, job in jobs.items():
        if name not in stages:
            continue
        try:
            written += job()
        except Exception as exc:  # noqa: BLE001 - one family must not stop the rest
            logger.warning("%s figures failed: %s", name, exc, exc_info=True)
    return written


def main():
    args = parse_args()
    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][
            min(args.verbose, 2)
        ],
        format="%(levelname)s | %(message)s",
    )
    torch.set_float32_matmul_precision("high")

    config, checkpoints, out_dir = resolve_sources(args)
    logger.info("%d checkpoints -> %s", len(checkpoints), out_dir)

    integrator = None
    if not args.render_only:
        from integrator.utils import (
            construct_data_loader,
            inject_binning_labels,
        )

        data_loader = construct_data_loader(config)
        data_loader.setup()
        inject_binning_labels(data_loader, config)
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        integrator = replay_checkpoints(
            config,
            checkpoints,
            out_dir,
            args,
            data_loader.full_dataset,
            device,
        )
        collect_latents_and_checks(
            integrator, data_loader, out_dir, args, device
        )

    written = render(out_dir, args, integrator)
    print(f"\n{len(written)} files written to {out_dir}")


if __name__ == "__main__":
    main()
