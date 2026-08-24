"""Drive a trained Laue run through predict -> careless -> phenix -> peaks.

Each step prints the exact shell it runs, and `--dry-run` prints without
executing, so the pipeline doubles as documentation of the commands.

Usage:
    python scripts/poly/run_pipeline.py --run-dir <run-dir> --dry-run
    python scripts/poly/run_pipeline.py --run-dir <run-dir> \
        --cfg scripts/poly/poly_pipeline_cfg.yaml
    python scripts/poly/run_pipeline.py --run-dir <run-dir> \
        --steps careless,phenix,peaks      # predictions already written
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ALL_STEPS = ("predict", "careless", "phenix", "peaks")
DEFAULT_CFG = Path(__file__).with_name("poly_pipeline_cfg.yaml")
# micromamba hook; override with MAMBA_SH for a different install
MAMBA_SH = os.environ.get(
    "MAMBA_SH",
    "/n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh",
)


def parse_args():
    p = argparse.ArgumentParser(description="Laue post-training pipeline")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    p.add_argument(
        "--steps",
        default=",".join(ALL_STEPS),
        help=f"Comma-separated subset of {ALL_STEPS}",
    )
    p.add_argument(
        "--config",
        type=int,
        default=None,
        help="careless config 1-6; overrides the pipeline config",
    )
    p.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch to process (default: the highest one predicted)",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def run(cmd: str, cwd: Path | None = None, dry: bool = False) -> None:
    """Echo a bash command, then run it unless this is a dry run."""
    where = f"  (cwd {cwd})" if cwd else ""
    print(f"\n$ {cmd}{where}", flush=True)
    if dry:
        return
    proc = subprocess.run(
        ["bash", "-lc", cmd], cwd=str(cwd) if cwd else None, check=False
    )
    if proc.returncode:
        raise SystemExit(f"step failed with exit code {proc.returncode}")


def via_python(cmd: str) -> str:
    """Run a console script through python, ignoring its shebang line.

    The `crls` environment predates the storage migration: its 52 console
    scripts still start with `#!/n/hekstra_lab/...`, an interpreter path that
    no longer exists, so executing them directly fails with "bad interpreter"
    (exit 126). Resolving the script and handing it to the active python
    sidesteps that without modifying the environment.
    """
    prog, _, rest = cmd.partition(" ")
    return f'python "$(command -v {prog})" {rest}'


def in_env(cmd: str, env: str | None) -> str:
    """Wrap a command in its environment: a phenix_env.sh, or a mamba env."""
    if not env:
        return cmd
    if str(env).endswith(".sh"):
        return f"source {env} && {cmd}"
    return f"source {MAMBA_SH} && micromamba activate {env} && {cmd}"


def predictions_dir(run_dir: Path) -> Path:
    """Read predictions_dir out of the run's run_paths.yaml."""
    meta = yaml.safe_load((run_dir / "run_paths.yaml").read_text())
    return Path(
        meta.get("predictions_dir")
        or Path(meta["output_root"]) / "predictions"
    )


def find_mtz(
    pred_dir: Path, epoch: int | None, allow_missing: bool = False
) -> Path:
    """The predicted unmerged MTZ for `epoch`, or the highest epoch present.

    With `allow_missing` (a dry run, or a run still training) the expected
    path is returned instead of raising, so the rest of the chain can still
    be printed.
    """
    mtzs = sorted(pred_dir.glob("epoch_*/preds_epoch_*.mtz"))
    if not mtzs:
        if allow_missing:
            e = 0 if epoch is None else int(epoch)
            return pred_dir / f"epoch_{e:04d}" / f"preds_epoch_{e:04d}.mtz"
        raise SystemExit(
            f"no preds_epoch_*.mtz under {pred_dir}; run the predict step "
            "with write_mtz enabled"
        )
    if epoch is None:
        return mtzs[-1]
    match = [m for m in mtzs if re.search(rf"epoch_0*{epoch}\b", m.name)]
    if not match:
        raise SystemExit(f"no MTZ for epoch {epoch} under {pred_dir}")
    return match[0]


def step_predict(cfg, run_dir, dry):
    """integrator.predict over the run's checkpoints, writing MTZs."""
    pcfg = cfg.get("predict", {})
    flags = "--write-mtz" if pcfg.get("write_mtz", True) else ""
    cmd = f"integrator.predict -v --run-dir {run_dir} {flags}".strip()
    run(in_env(cmd, pcfg.get("env")), dry=dry)


def step_careless(cfg, mtz: Path, dry) -> Path:
    """Scale and merge the unmerged MTZ with careless.

    Writes into `<mtz parent>/<out_dir>/config<N>_*`, matching the layout the
    refltorch scaling scripts produce, so downstream tooling finds it.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from careless_configs import METADATA_KEYS, flags

    ccfg = cfg.get("careless", {})
    config = int(ccfg.get("config", 4))
    scaling_dir = mtz.parent / ccfg.get("out_dir", "scaling")
    if not dry:
        scaling_dir.mkdir(parents=True, exist_ok=True)
    out_base = scaling_dir / f"config{config}"

    args = flags(config, ccfg.get("dmin", 1.5), ccfg.get("seed"))
    args += [str(a) for a in ccfg.get("extra_args", [])]
    cmd = (
        f"careless poly {' '.join(args)} "
        f'"{METADATA_KEYS}" {mtz} {out_base}'
    )
    run(in_env(via_python(cmd), ccfg.get("env", "crls")), dry=dry)
    return Path(f"{out_base}_0.mtz")


def step_phenix(cfg, merged: Path, work_dir: Path, dry) -> Path:
    """Two-pass phenix.refine, matching the Laue refinement recipe.

    Pass 1 refines the reference model against the merged data; pass 2 picks
    up pass 1's data and model and reuses its R-free flags.
    """
    pcfg = cfg.get("phenix", {})
    for key in ("env", "eff1", "eff2", "pdb"):
        if not pcfg.get(key):
            raise SystemExit(f"phenix.{key} is not set in the pipeline config")
    prefix = pcfg.get("prefix", "refined")
    refine1 = work_dir / "refine1"
    refine2 = work_dir / "refine2"
    if not dry:
        refine1.mkdir(parents=True, exist_ok=True)
        refine2.mkdir(parents=True, exist_ok=True)

    pass1 = (
        f'phenix.refine "{Path(pcfg["eff1"]).resolve()}" '
        f'refinement.input.xray_data.file_name="{merged.resolve()}" '
        f'refinement.input.pdb.file_name="{Path(pcfg["pdb"]).resolve()}" '
        f"refinement.output.prefix={prefix} --overwrite"
    )
    run(in_env(pass1, pcfg["env"]), cwd=refine1, dry=dry)

    data = refine1 / f"{prefix}_data.mtz"
    model = refine1 / f"{prefix}_1.pdb"
    pass2 = (
        f'phenix.refine "{Path(pcfg["eff2"]).resolve()}" '
        f'refinement.input.xray_data.file_name="{data}" '
        f'refinement.input.pdb.file_name="{model}" '
        f'refinement.input.xray_data.r_free_flags.file_name="{data}" '
        f"refinement.output.prefix={prefix} --overwrite"
    )
    run(in_env(pass2, pcfg["env"]), cwd=refine2, dry=dry)
    return refine2


def refined_output(refine_dir: Path, prefix: str, suffix: str) -> Path:
    """The highest-serial phenix output in `refine_dir`, e.g. refined_2.pdb.

    phenix.refine increments the serial from the input model, so the second
    pass reads `refined_1.pdb` and writes `refined_2.pdb`. Globbing for the
    highest serial keeps this correct however many passes run, instead of
    hard-coding one.
    """
    files = list(refine_dir.glob(f"{prefix}_[0-9]*{suffix}"))
    if not files:
        return refine_dir / f"{prefix}_2{suffix}"  # expected name, dry runs

    def serial(path: Path) -> int:
        stem = path.name[len(prefix) + 1 : -len(suffix)]
        return int(stem) if stem.isdigit() else -1

    return max(files, key=serial)


def step_peaks(cfg, refine2: Path, merged: Path, dry) -> None:
    """Anomalous peaks from the refined model, plus CCanom on the xval data.

    Two peak measures, because they answer different questions and only one
    of them feeds the plotting suite:

    - `rs.find_peaks` writes the one-row-per-site table that `plot_peaks.py`
      reads (`seqid`, `peakz`), so it is the default.
    - `anomalous_peak_heights.py` samples the ANOM map at the model's S and I
      sites and averages over symmetry mates. It writes a two-row transposed
      CSV, which is what the earlier laue-dials runs used, so it is kept for
      comparability under a separate name.
    """
    pcfg = cfg.get("peaks", {})
    method = pcfg.get("method", "find_peaks")
    prefix = cfg.get("phenix", {}).get("prefix", "refined")
    mtz = refined_output(refine2, prefix, ".mtz")
    pdb = refined_output(refine2, prefix, ".pdb")
    out_dir = refine2.parent

    if method in ("find_peaks", "both"):
        cmd = (
            f"rs.find_peaks {mtz} {pdb} "
            f"-f {pcfg.get('f', 'ANOM')} -p {pcfg.get('phi', 'PHANOM')} "
            f"-z {pcfg.get('z', 5.0)} -o {out_dir / 'peaks.csv'}"
        )
        run(in_env(cmd, pcfg.get("env")), cwd=refine2, dry=dry)

    if method in ("peak_heights", "both"):
        script = pcfg.get("script")
        if not script:
            raise SystemExit("peaks.script is needed for the peak_heights method")
        cmd = (
            f"python {script} {mtz} {Path(pcfg['pdb']).resolve()} "
            f"{pcfg.get('elements', '[S,I]')} {out_dir / 'peak_heights.csv'}"
        )
        run(in_env(cmd, pcfg.get("env")), dry=dry)

    if pcfg.get("ccanom", True):
        xval = merged.with_name(merged.name.replace("_0.mtz", "_xval_0.mtz"))
        ccanom = (
            f"careless.ccanom {xval} "
            f"-o {out_dir / 'ccanom.csv'} -i {out_dir / 'ccanom.png'}"
        )
        run(
            in_env(via_python(ccanom), pcfg.get("ccanom_env", "crls")),
            dry=dry,
        )


def main():
    args = parse_args()
    cfg = load_cfg(args.cfg)
    if args.config is not None:
        cfg.setdefault("careless", {})["config"] = args.config
    run_dir = args.run_dir.resolve()
    steps = [s for s in args.steps.split(",") if s]
    unknown = set(steps) - set(ALL_STEPS)
    if unknown:
        raise SystemExit(f"unknown steps: {sorted(unknown)}")

    print(f"run dir  : {run_dir}")
    print(f"steps    : {', '.join(steps)}")
    print(f"careless : config {cfg.get('careless', {}).get('config', 4)}")

    if "predict" in steps:
        step_predict(cfg, run_dir, args.dry_run)

    merged = refine2 = None
    if {"careless", "phenix", "peaks"} & set(steps):
        mtz = find_mtz(
            predictions_dir(run_dir), args.epoch, allow_missing=args.dry_run
        )
        print(f"unmerged : {mtz}")
        if args.dry_run and not mtz.exists():
            print("           (does not exist yet; predict writes it)")
        ccfg = cfg.get("careless", {})
        config = int(ccfg.get("config", 4))
        scaling_dir = mtz.parent / ccfg.get("out_dir", "scaling")
        merged = scaling_dir / f"config{config}_0.mtz"
        work_dir = scaling_dir / f"config{config}_refine"
        if "careless" in steps:
            merged = step_careless(cfg, mtz, args.dry_run)
    if "phenix" in steps:
        refine2 = step_phenix(cfg, merged, work_dir, args.dry_run)
    if "peaks" in steps:
        step_peaks(
            cfg, refine2 or work_dir / "refine2", merged, args.dry_run
        )

    print("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
