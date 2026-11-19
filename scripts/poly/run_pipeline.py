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

from integrator.utils import checkpoints, phenix

ALL_STEPS = ("predict", "careless", "stats", "phenix", "peaks")

# `careless.<tool>` -> which of careless's three outputs it reads, how many
# resolution bins it defaults to, and whether it takes `-m/--method`.
#
# The three outputs are not interchangeable. `<base>_xval_0.mtz` carries
# `repeat` and `half`, `<base>_predictions_0.mtz` carries `test`, `Iobs` and
# `Ipred`, and `<base>_0.mtz` is the merged result with `I(+)/SigI(+)`;
# handing a tool the wrong one raises a KeyError on the column it wanted.
# Verified against careless 0.4's own parsers, not inferred from the names.
CARELESS_STATS = {
    "cchalf": {"input": "xval", "bins": 10, "method": True},
    "ccanom": {"input": "xval", "bins": 10, "method": True},
    "rsplit": {"input": "xval", "bins": 10, "method": False},
    "ccpred": {"input": "predictions", "bins": 10, "method": True},
    # isigi reads the merged file through --intensity-key, which defaults to
    # None and then raises `KeyError: None`: an anomalous merge has I(+)/I(-)
    # and no plain `I`, so the key has to be named.
    "isigi": {"input": "merged", "bins": 20, "method": False, "keyed": True},
    "completeness": {"input": "merged", "bins": 10, "method": False},
}

# Columns in a careless.stats CSV that say which row this is rather than
# measuring anything. Everything else that parses as a number is a statistic.
_STATS_ID_COLS = {
    "", "index", "bin", "Bin", "file", "Spacegroup", "spacegroup", "cell",
    "Resolution Range (Å)", "Resolution Range", "dHKL", "repeat", "half",
    "test",
}
# the column naming the resolution shell, in the order careless might use
_STATS_SHELL_COLS = ("Resolution Range (Å)", "Resolution Range", "dHKL")
# a second grouping careless splits on: Train/Test for ccpred, the
# crossvalidation repeat for cchalf and friends
_STATS_GROUP_COLS = ("test", "repeat", "half")
DEFAULT_CFG = Path(__file__).with_name("poly_pipeline_cfg.yaml")
# micromamba hook; override with MAMBA_SH for a different install
MAMBA_SH = os.environ.get(
    "MAMBA_SH",
    "/n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh",
)


def parse_args():
    p = argparse.ArgumentParser(description="Laue post-training pipeline")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    p.add_argument(
        "--out-base",
        type=Path,
        default=None,
        help=(
            "Run `stats` directly on a careless output base, e.g. "
            "<dir>/merged_e35, instead of resolving it from --run-dir. For "
            "chains that scaled outside this pipeline; PDZ2 is the case it "
            "was added for. Implies --steps stats."
        ),
    )
    p.add_argument(
        "--method",
        default=None,
        help="Override the `method` key column written into the long table",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="Override the `dataset` key column written into the long table",
    )
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
    """Wrap a command in its environment.

    Three forms: a phenix install directory, a `phenix_env.sh` to source, or a
    mamba env to activate. A phenix 2.0 install has to be the first form --
    its own env script points at the pre-migration prefix and leaves nothing
    on PATH (see `integrator.utils.phenix`).
    """
    if not env:
        return cmd
    if str(env).endswith(".sh"):
        return f"source {env} && {cmd}"
    if (Path(env) / "bin" / "phenix.refine").exists() or "phenix" in Path(env).name:
        return f"{phenix.shell_prelude(install=env)} && {cmd}"
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


epoch_of = checkpoints.epoch_of


def checkpoints_for(run_dir: Path, scope) -> list[Path] | None:
    """The checkpoints `scope` names, or None to let predict walk them all.

    The selection itself lives in `integrator.utils.checkpoints`, because the
    monochromatic chain asks the same question and the two answers must not
    drift apart. `None` is kept here as the "every one" signal, which is what
    `integrator.predict --run-dir` does when given no `--ckpt`.
    """
    if scope == "all":
        return None
    try:
        got = checkpoints.select(run_dir, scope)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    return got or None


def checkpoint_for(run_dir: Path, scope) -> Path | None:
    """The single checkpoint `scope` names, or None for "every one"."""
    got = checkpoints_for(run_dir, scope)
    if got is None:
        return None
    if len(got) > 1:
        raise SystemExit(
            f"scope {scope!r} names {len(got)} checkpoints; this step takes one"
        )
    return got[0]


# The MODULE, not the console script. `integrator.predict` is not on PATH
# in every environment that has the package -- the pipeline died with
# `bash: integrator.predict: command not found` -- and where it is, its
# shebang may be the pre-migration interpreter, which is what `via_python`
# exists to work around. `python -m` needs neither: it resolves through the
# active environment's own interpreter and the installed package.
PREDICT_CMD = "python -m integrator.cli.predict"


def step_predict(cfg, run_dir, dry):
    """integrator.predict over the selected checkpoints, writing MTZs."""
    pcfg = cfg.get("predict", {})
    base = ["--write-mtz"] if pcfg.get("write_mtz", True) else []
    try:
        ckpts = checkpoints_for(run_dir, pcfg.get("epoch", "last"))
    except (OSError, KeyError):
        ckpts = None  # dry run against a run that has not trained yet
    if ckpts is None:
        # no --ckpt: predict walks every checkpoint itself
        cmd = f"{PREDICT_CMD} -v --run-dir {run_dir} {' '.join(base)}"
        run(in_env(cmd.strip(), pcfg.get("env")), dry=dry)
        return
    print(f"predict  : {len(ckpts)} checkpoint(s): "
          f"{', '.join(str(epoch_of(c)) for c in ckpts)}")
    for ckpt in ckpts:
        flags = base + ["--ckpt", str(ckpt)]
        cmd = f"{PREDICT_CMD} -v --run-dir {run_dir} {' '.join(flags)}"
        run(in_env(cmd.strip(), pcfg.get("env")), dry=dry)


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

    # the canonical per-shell statistics, so this run can share a figure with
    # a DIALS-scaled monochromatic one
    if ccfg.get("merging_stats", True):
        emitter = (
            Path(__file__).resolve().parent / "emit_merging_stats.py"
        )
        stats_cmd = (
            f"python {emitter} --scaling-dir {scaling_dir} "
            f"--config {config} --unmerged {mtz}"
        )
        run(in_env(stats_cmd, cfg.get("predict", {}).get("env")), dry=dry)

    return Path(f"{out_base}_0.mtz")


def step_stats(cfg, out_base: Path, epoch, dry) -> None:
    """Every `careless.stats` tool, each to its own CSV, then one long table.

    These are the numbers careless can report and DIALS cannot be asked for,
    so they are the polychromatic counterpart of the per-shell merging table:
    CChalf, CCanom, CCpred, Rsplit, I/sigma and completeness by resolution.
    Each tool writes its own CSV with `-o`; `careless_stats.csv` beside them
    is all of it concatenated with the `method`, `dataset`, `epoch` and
    `correlation` key columns the figures group on.
    """
    scfg = cfg.get("stats", {})
    tools = [t for t in (scfg.get("tools") or list(CARELESS_STATS))]
    unknown = set(tools) - set(CARELESS_STATS)
    if unknown:
        raise SystemExit(f"unknown careless.stats tools: {sorted(unknown)}")

    out_dir = out_base.parent / scfg.get("out_dir", "stats")
    if not dry:
        out_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "merged": Path(f"{out_base}_0.mtz"),
        "xval": Path(f"{out_base}_xval_0.mtz"),
        "predictions": Path(f"{out_base}_predictions_0.mtz"),
    }
    # Which correlation `careless.stats` computes, for the tools that take
    # `-m`. CChalf, CCanom and CCpred are correlations and Pearson and
    # Spearman do not rank arms the same way -- Pearson follows the bright
    # end, Spearman is insensitive to the scale of the intensities -- so a
    # single one is a choice that has to be declared. `methods` runs the
    # tool once per entry; absent, the behaviour is exactly as before, one
    # run using `method`, so existing configs are unchanged.
    correlations = list(scfg.get("methods") or [scfg.get("method") or ""])
    written = []
    for tool in tools:
        spec = CARELESS_STATS[tool]
        src = sources[spec["input"]]
        if not dry and not src.exists():
            print(f"stats    : {tool} skipped, no {src.name}")
            continue
        # a tool with no `-m` runs once whatever `methods` says
        for corr in (correlations if spec["method"] else [""]):
            _run_one_stat(tool, spec, src, out_dir, scfg, corr,
                          len(correlations) > 1, written, dry)

    tidy = out_dir / "careless_stats.csv"
    if dry:
        print(f"stats    : would tidy {len(written)} tables into {tidy}")
        return
    _tidy_careless_stats(written, tidy, cfg, epoch)


def _run_one_stat(tool, spec, src, out_dir, scfg, corr, suffixed,
                  written, dry) -> None:
    """One `careless.<tool>` invocation, for one correlation method.

    The output file carries the method in its name only when more than one
    is being run, so a single-method run writes the same `<tool>.csv` it
    always has and nothing downstream has to know about this.
    """
    csv_path = out_dir / (f"{tool}_{corr}.csv" if suffixed and corr
                          else f"{tool}.csv")
    flags = [f"-b {scfg.get('bins', spec['bins'])}"]
    if spec["method"] and corr:
        flags.append(f"-m {corr}")
    if spec.get("keyed"):
        flags += [
            f"--intensity-key '{scfg.get('intensity_key', 'I(+)')}'",
            f"--uncertainty-key '{scfg.get('uncertainty_key', 'SigI(+)')}'",
        ]
    flags += [str(a) for a in scfg.get("extra_args", [])]
    cmd = (
        f"careless.{tool} {' '.join(flags)} -o {csv_path} {src}"
    )
    run(in_env(via_python(cmd), scfg.get("env", "crls")), dry=dry)
    written.append((tool, csv_path, corr))


def _tidy_careless_stats(written, tidy: Path, cfg, epoch) -> None:
    """Concatenate the per-tool CSVs into one long-format table.

    Each tool writes its own columns, so the value column is found rather
    than named: everything that is not a bin label, a resolution or an index
    is melted into `statistic`/`value`. That way a tool added to careless
    later needs no change here.
    """
    import csv as _csv

    keys = cfg.get("keys", {})
    method = keys.get("method", "integrator")
    dataset = keys.get("dataset", "hewl_1118")
    rows = []
    for tool, path, corr in written:
        if not path.is_file():
            continue
        with open(path, newline="") as fh:
            for i, rec in enumerate(_csv.DictReader(fh)):
                shell = next(
                    (rec[c] for c in _STATS_SHELL_COLS if rec.get(c)), ""
                )
                group = next(
                    (rec[c] for c in _STATS_GROUP_COLS if rec.get(c) not in
                     (None, "")), ""
                )
                try:
                    shell_bin = int(float(rec.get("bin", i)))
                except (TypeError, ValueError):
                    shell_bin = i
                for col, val in rec.items():
                    if col in _STATS_ID_COLS or val in ("", None):
                        continue
                    try:
                        v = float(val)
                    except (TypeError, ValueError):
                        continue
                    rows.append({
                        "method": method,
                        "dataset": dataset,
                        "epoch": epoch if epoch is not None else "",
                        "tool": tool,
                        # which correlation careless computed, empty for a
                        # tool that does not take `-m`. Not folded into
                        # `statistic`, which already names the column within
                        # a tool's own CSV, and not into `method`, which is
                        # the arm.
                        "correlation": corr,
                        "statistic": col,
                        "bin": shell_bin + 1,
                        "shell": shell,
                        "group": group,
                        "value": v,
                    })
    if not rows:
        print(f"stats    : nothing to tidy into {tidy}")
        return
    with open(tidy, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"stats    : {tidy} ({len(rows)} rows from {len(written)} tools)")


def step_phenix(cfg, merged: Path, work_dir: Path, dry) -> Path:
    """Two-pass phenix.refine, matching the Laue refinement recipe.

    Pass 1 refines the reference model against the merged data; pass 2 picks
    up pass 1's data and model and reuses its R-free flags.

    With `phenix.rfree` set, both passes instead read that one flag set, which
    is what makes two runs comparable: the generating form holds out a
    different subset of every MTZ, worth about 0.008 in $R_\\mathrm{free}$ on
    weak Laue data.
    """
    pcfg = cfg.get("phenix", {})
    for key in ("env", "eff1", "eff2", "pdb"):
        if not pcfg.get(key):
            raise SystemExit(f"phenix.{key} is not set in the pipeline config")
    rfree = pcfg.get("rfree")
    prefix = pcfg.get("prefix", "refined")
    refine1 = work_dir / "refine1"
    refine2 = work_dir / "refine2"
    if not dry:
        refine1.mkdir(parents=True, exist_ok=True)
        refine2.mkdir(parents=True, exist_ok=True)

    # phenix 2.0 effs carry their own files, because the labels that select a
    # column live in the same data_manager block as the file they select from
    # and a command-line file= opens a fresh block instead. A 1.20 eff names
    # no files, so those still take the override form.
    subs1 = {"MTZFILE": merged, "MODELFILE": pcfg["pdb"]}
    if rfree:
        subs1["FREEFILE"] = rfree
    if not dry and phenix.eff_takes_substitution(pcfg["eff1"]):
        eff1 = phenix.render_eff(pcfg["eff1"], refine1 / "refine.eff", **subs1)
        pass1 = f'phenix.refine "{eff1}"'
    else:
        pass1 = (
            f'phenix.refine "{Path(pcfg["eff1"]).resolve()}" '
            f'refinement.input.xray_data.file_name="{merged.resolve()}" '
            f'refinement.input.pdb.file_name="{Path(pcfg["pdb"]).resolve()}" '
            + (f"{phenix.rfree_overrides(rfree)} " if rfree else "")
            + f"refinement.output.prefix={prefix} --overwrite"
        )
    run(in_env(pass1, pcfg["env"]), cwd=refine1, dry=dry)

    # pass 1's MTZ carries both the data and the R-free set it generated, so
    # pass 2 holds out the same reflections. phenix writes that `_data.mtz`
    # only to export flags it generated itself, so with a fixed set there is
    # none and pass 2 re-reads the merged data -- the same data and the same
    # test set in both passes, as the mono chain does.
    data = merged if rfree else refine1 / f"{prefix}_data.mtz"
    model = refine1 / f"{prefix}_1.pdb"
    subs2 = {"MTZFILE": data, "MODELFILE": model}
    if rfree:
        subs2["FREEFILE"] = rfree
    if not dry and phenix.eff_takes_substitution(pcfg["eff2"]):
        eff2 = phenix.render_eff(pcfg["eff2"], refine2 / "refine.eff", **subs2)
        pass2 = f'phenix.refine "{eff2}"'
    else:
        pass2 = (
            f'phenix.refine "{Path(pcfg["eff2"]).resolve()}" '
            f'refinement.input.xray_data.file_name="{data}" '
            f'refinement.input.pdb.file_name="{model}" '
            + (
                f"{phenix.rfree_overrides(rfree)} "
                if rfree
                else f'refinement.input.xray_data.r_free_flags.file_name="{data}" '
            )
            + f"refinement.output.prefix={prefix} --overwrite"
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
        # phenix 2.0 writes the anomalous phase as PANOM where 1.20 wrote
        # PHANOM, and naming the wrong one makes rs.find_peaks raise
        # `KeyError: None of [Index(['PHANOM'])]`. Read it off the MTZ.
        phi = pcfg.get("phi")
        if not phi:
            cols = phenix.anomalous_columns(mtz)
            phi = cols[1] if cols else phenix.PHASE_LABELS[0]
        cmd = (
            f"rs.find_peaks {mtz} {pdb} "
            f"-f {pcfg.get('f', 'ANOM')} -p {phi} "
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
    for key in ("method", "dataset"):
        if getattr(args, key) is not None:
            cfg.setdefault("keys", {})[key] = getattr(args, key)

    # A chain that scaled outside this pipeline has no run directory to
    # resolve, so `--out-base` names the careless output itself and only the
    # `stats` step runs. Everything upstream of the merge already happened.
    if args.out_base is not None:
        out_base = args.out_base.resolve()
        print(f"out base : {out_base}")
        step_stats(cfg, out_base, args.epoch, args.dry_run)
        print("\ndone")
        return 0

    if args.run_dir is None:
        raise SystemExit("--run-dir is required unless --out-base is given")
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

    if {"careless", "stats", "phenix", "peaks"} & set(steps):
        targets = downstream_mtzs(cfg, run_dir, args)
        if len(targets) > 1:
            print(f"scoring  : {len(targets)} epochs: "
                  f"{', '.join(str(epoch_of(m)) for m in targets)}")
        for mtz in targets:
            process_one(cfg, mtz, steps, args)

    print("\ndone")
    return 0


def downstream_mtzs(cfg, run_dir: Path, args) -> list[Path]:
    """The unmerged MTZs the chain runs on, in epoch order.

    This follows the same scope `predict` used rather than taking whatever
    MTZ sits highest on disk. Under `every:<n>` predict writes one MTZ per
    selected checkpoint, and scaling only the highest would quietly report a
    single epoch as though the trajectory had been scored.
    """
    pred_dir = predictions_dir(run_dir)
    if args.epoch is not None:
        return [find_mtz(pred_dir, args.epoch, allow_missing=args.dry_run)]
    scope = cfg.get("predict", {}).get("epoch", "last")
    try:
        ckpts = None if scope == "all" else checkpoints.select(run_dir, scope)
    except (OSError, KeyError, ValueError):
        ckpts = None
    if not ckpts:
        # no run yet, or "all": fall back to what predict has written
        return [find_mtz(pred_dir, None, allow_missing=args.dry_run)]
    out = []
    for c in ckpts:
        e = epoch_of(c)
        expected = pred_dir / f"epoch_{e:04d}" / f"preds_epoch_{e:04d}.mtz"
        out.append(expected if (expected.exists() or args.dry_run)
                   else find_mtz(pred_dir, e, allow_missing=False))
    return out


def process_one(cfg, mtz: Path, steps, args) -> None:
    """Scale, score, refine and peak-pick one epoch's unmerged MTZ."""
    print(f"\nunmerged : {mtz}")
    if args.dry_run and not mtz.exists():
        print("           (does not exist yet; predict writes it)")
    ccfg = cfg.get("careless", {})
    config = int(ccfg.get("config", 4))
    scaling_dir = mtz.parent / ccfg.get("out_dir", "scaling")
    out_base = scaling_dir / f"config{config}"
    merged = scaling_dir / f"config{config}_0.mtz"
    work_dir = scaling_dir / f"config{config}_refine"
    refine2 = None

    if "careless" in steps:
        merged = step_careless(cfg, mtz, args.dry_run)
    if "stats" in steps:
        step_stats(cfg, out_base, epoch_of(mtz), args.dry_run)
    if "phenix" in steps:
        refine2 = step_phenix(cfg, merged, work_dir, args.dry_run)
    if "peaks" in steps:
        step_peaks(cfg, refine2 or work_dir / "refine2", merged, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
