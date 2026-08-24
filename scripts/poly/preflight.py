"""Check a Laue run's inputs before spending a GPU allocation on it.

Validates, in order: the training config resolves and the model builds, the
dataset directory is complete and self-consistent, the metadata carries the
columns the MTZ writer and careless need, and the downstream tools and
reference files named in the pipeline config exist.

Nothing here loads the shoebox arrays, so it runs in seconds on a login node.

Usage:
    python scripts/poly/preflight.py --config configs/poly/hewl1118_poly.yaml
    python scripts/poly/preflight.py --config <cfg> \
        --pipeline-cfg scripts/poly/poly_pipeline_cfg.yaml
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# columns the MTZ writer reads straight out of the metadata file
MTZ_KEYS = ("H", "K", "L", "xyzcal.px.0", "xyzcal.px.1")
# a per-reflection wavelength only exists, and is only needed, for Laue data
POLY_ONLY_KEYS = ("wavelength",)

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "


class Report:
    """Collects check results and decides the exit status."""

    def __init__(self):
        self.failed = 0
        self.warned = 0

    def ok(self, what, detail=""):
        print(f"[{OK}] {what}{(' — ' + detail) if detail else ''}")

    def warn(self, what, detail=""):
        self.warned += 1
        print(f"[{WARN}] {what}{(' — ' + detail) if detail else ''}")

    def fail(self, what, detail=""):
        self.failed += 1
        print(f"[{BAD}] {what}{(' — ' + detail) if detail else ''}")

    def exists(self, label, path, required=True):
        if path is None:
            (self.fail if required else self.warn)(label, "not set")
            return False
        if Path(path).exists():
            self.ok(label, str(path))
            return True
        (self.fail if required else self.warn)(label, f"missing: {path}")
        return False


def parse_args():
    p = argparse.ArgumentParser(description="Pre-flight a Laue training run")
    p.add_argument("--config", required=True, help="Training config YAML")
    p.add_argument(
        "--pipeline-cfg",
        default=None,
        help="poly_pipeline_cfg.yaml, to also check the downstream tools",
    )
    p.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip loading the metadata file (it can be ~0.5 GB)",
    )
    return p.parse_args()


def check_config(report, config_path):
    """Resolve the config and build the model from it."""
    from integrator.utils import (
        apply_dataset_defaults,
        construct_integrator,
        load_config,
    )

    cfg = apply_dataset_defaults(load_config(config_path))
    try:
        model = construct_integrator(cfg)
    except Exception as exc:  # noqa: BLE001 - the point is to report it
        report.fail("config builds a model", f"{type(exc).__name__}: {exc}")
        return cfg, None
    n = sum(p.numel() for p in model.parameters())
    report.ok(
        "config builds a model",
        f"{type(model).__name__}, {n / 1e6:.2f}M params, "
        f"shoebox {model.shoebox_shape}",
    )
    return cfg, model


def mode_of(cfg) -> str:
    """`polychromatic` or `monochromatic`, from the config's mode or its loss."""
    mode = cfg.get("mode")
    if mode:
        return str(mode)
    loss = str(cfg.get("loss", {}).get("name", ""))
    return "polychromatic" if loss.startswith("poly") else "monochromatic"


def check_dataset(report, cfg, skip_metadata):
    """Dataset directory, manifest, array sizes, and metadata columns.

    The requirements differ by mode: Laue data carries a per-reflection
    wavelength and needs image_num so careless can scale per image, while
    rotation data has neither and is scaled by DIALS from the experiment
    list.
    """
    from integrator.io import load_metadata, read_dataset_spec

    dl_args = cfg.get("data_loader", {}).get("args", {})
    data_dir = Path(dl_args.get("data_dir", ""))
    if not report.exists("data_dir", data_dir):
        return

    spec = read_dataset_spec(data_dir)
    if spec is None:
        report.fail(
            "dataset.yaml",
            f"absent in {data_dir}; the data module raises without it",
        )
        return
    report.ok("dataset.yaml", f"{spec['n_reflections']:,} reflections")

    mode = mode_of(cfg)
    is_poly = mode == "polychromatic"
    if bool(spec.get("polychromatic")) != is_poly:
        report.fail(
            "mode vs dataset",
            f"config is {mode} but dataset.yaml says polychromatic="
            f"{bool(spec.get('polychromatic'))}",
        )
    else:
        report.ok("mode", f"{mode}, matching dataset.yaml")
    if "crystal" not in spec:
        report.fail("dataset.yaml crystal block", "the MTZ writer needs it")

    files = spec.get("files", {})
    geom = spec["geometry"]
    n_px = geom["d"] * geom["h"] * geom["w"]
    for key, itemsize in (("counts", None), ("masks", None)):
        path = data_dir / files.get(key, f"{key}.npy")
        if not report.exists(f"files.{key}", path):
            continue
        size = path.stat().st_size
        if path.suffix == ".npy":
            per_refl = (size - 128) / spec["n_reflections"]
            if per_refl % n_px:
                report.warn(
                    f"files.{key} size",
                    f"{per_refl:.1f} B/reflection is not a multiple of "
                    f"{n_px} pixels; n_reflections may be wrong",
                )
            else:
                report.ok(
                    f"files.{key} size",
                    f"{size / 1e9:.2f} GB, {per_refl / n_px:.0f} B/pixel",
                )
        _ = itemsize

    meta_path = data_dir / files.get("reference", "metadata.npy")
    if not report.exists("files.reference", meta_path) or skip_metadata:
        return

    meta = load_metadata(meta_path)
    wanted = MTZ_KEYS + (POLY_ONLY_KEYS if is_poly else ())
    missing = [k for k in wanted if k not in meta]
    if missing:
        report.fail("metadata columns for the MTZ", f"missing {missing}")
    else:
        report.ok("metadata columns for the MTZ", ", ".join(wanted))

    if "image_num" in meta:
        n_images = len(set(meta["image_num"].tolist()))
        report.ok("metadata image_num", f"{n_images} images -> BATCH")
    elif is_poly:
        report.fail(
            "metadata image_num",
            "absent: every BATCH becomes 1 and careless cannot scale "
            "per image. Fix with scripts/add_image_num_to_metadata.py",
        )
    else:
        report.ok(
            "metadata image_num",
            "absent, which is fine: DIALS scales from the experiment list",
        )

    if is_poly and "wavelength" in meta:
        lam = meta["wavelength"]
        lo, hi = float(lam.min()), float(lam.max())
        report.ok("wavelength range", f"{lo:.4f} – {hi:.4f} Å")
        loss_args = cfg.get("loss", {}).get("args", {})
        cfg_lo = loss_args.get("lambda_min")
        cfg_hi = loss_args.get("lambda_max")
        if cfg_lo is not None and (lo < cfg_lo or hi > cfg_hi):
            report.warn(
                "loss lambda range",
                f"config [{cfg_lo}, {cfg_hi}] does not cover the data "
                f"[{lo:.4f}, {hi:.4f}]; G(λ) extrapolates at the edges",
            )
    if "is_test" not in meta:
        report.warn("metadata is_test", "no held-out split flag in the data")


def check_pipeline(report, pipeline_path):
    """Downstream tools and the reference files the pipeline config names."""
    import sys as _sys

    import yaml

    _sys.path.insert(0, str(Path(__file__).parent))
    from careless_configs import (
        DESCRIPTIONS,
        FRIEDEL_SPLIT_CONFIGS,
        METADATA_KEYS,
    )

    cfg = yaml.safe_load(Path(pipeline_path).read_text()) or {}

    careless = cfg.get("careless", {})
    config = int(careless.get("config", 4))
    if config in FRIEDEL_SPLIT_CONFIGS:
        report.fail(
            f"careless config {config}",
            "Friedel-split configs are not implemented in run_pipeline.py; "
            "use refltorch/scripts/laue_output/careless_scale.sh",
        )
    elif config in DESCRIPTIONS:
        report.ok(f"careless config {config}", DESCRIPTIONS[config])
    else:
        report.fail("careless config", f"unknown: {config}")
    report.ok("careless metadata keys", METADATA_KEYS)

    phenix = cfg.get("phenix", {})
    report.exists("phenix env", phenix.get("env"))
    report.exists("phenix eff1", phenix.get("eff1"))
    report.exists("phenix eff2", phenix.get("eff2"))
    report.exists("phenix starting model", phenix.get("pdb"))

    peaks = cfg.get("peaks", {})
    report.exists("peak-height script", peaks.get("script"))
    report.exists("peak-height reference model", peaks.get("pdb"))

    for tool in ("careless", "phenix.refine"):
        where = shutil.which(tool)
        if where:
            report.ok(f"{tool} on PATH", where)
        else:
            report.warn(
                f"{tool} on PATH",
                "not visible from this shell; the driver activates its env",
            )

    for label, key in (
        ("reference peaks", "peaks"),
        ("reference refine log", "refine_log"),
    ):
        value = (cfg.get("reference") or {}).get(key)
        if value:
            report.exists(label, value, required=False)


def main():
    args = parse_args()
    report = Report()

    print(f"\n== training config: {args.config}")
    cfg, _ = check_config(report, args.config)

    print("\n== dataset")
    check_dataset(report, cfg, args.skip_metadata)

    if args.pipeline_cfg:
        print(f"\n== pipeline: {args.pipeline_cfg}")
        check_pipeline(report, args.pipeline_cfg)

    print(
        f"\n{report.failed} failure(s), {report.warned} warning(s). "
        f"CUDA visible: {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
    )
    sys.exit(1 if report.failed else 0)


if __name__ == "__main__":
    main()
