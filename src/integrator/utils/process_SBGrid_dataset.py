#!/usr/bin/env python3
"""Fetch an SBGrid dataset and run it through the laue-dials pipeline.

    python process_SBGrid_dataset.py -c ../../../configs/poly/laue_dials_processing_config.yaml

All input parameters are specified in the YAML config.  The `shoeboxes`,
`train`, `predict` and `evaluate` config sections are optional: each one, when
present, runs the corresponding integrator step after the laue-dials pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

IMAGE_EXTS = (".mccd", ".cbf", ".img", ".h5", ".nxs")
SBGRID_MODULE = "10.15785/SBGRID"
FRAME_RE = re.compile(r"\d+(?=\D*$)")
EPOCH_RE = re.compile(r"epoch[_=](\d+)")
CHUNK = 1 << 20

LAUNCH_CWD = Path.cwd()
# merge_eval (phenix + peaks) and plot_ckpt_eval (the plots) live here
SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
# careless learns the scale function from these; they are the columns
# laue.integrate writes and that write_mtz_from_preds mirrors exactly.  dHKL is
# one of careless' computed keys, as are Hobs,Kobs,Lobs if you want the original
# indices in the scale function too.
METADATA_KEYS = "dHKL,xcal,ycal,wavelength,BATCH"
# careless models structure factor amplitudes, so only the amplitude variant of
# merge_eval.VARIANTS applies (no merged intensities to French-Wilson).
VARIANT = "F_noFW"
# --force takes stage names; these stand in for a whole half of the run.
FORCE_GROUPS = {
    "integrator": {"integrator.make_shoeboxes", "integrator.train",
                   "integrator.predict"},
    "evaluate": {"flags", "careless", "refine"},
}


def resolve(p) -> Path:
    p = Path(p)
    return (p if p.is_absolute() else LAUNCH_CWD / p).resolve()


def need(mapping, key, ctx="config"):
    """Savely fetch a required config key."""
    if key not in mapping:
        sys.exit(f"{ctx}: missing required key {key!r}")
    return mapping[key]


def run(cmd, **kw) -> subprocess.CompletedProcess:
    """Run a command as an argv list, echoing first."""
    cmd = [str(c) for c in cmd]
    print("[run ]", " ".join(cmd))
    return subprocess.run(cmd, **kw)


# --------------------------------------------------------------------------- #
# Locating the images
# --------------------------------------------------------------------------- #

def rsync_list(base: str) -> list[tuple[str, int]]:
    """Catalogue a remote rsync tree as (path relative to `base`, size).
    To choose the sweep before downloading anything.
    """
    p = run(["rsync", "-r", "--list-only", f"{base}/"], capture_output=True, text=True)
    if p.returncode:
        sys.exit(f"rsync could not list {base}\n{p.stderr.strip()}")

    entries = []
    for line in p.stdout.splitlines():
        # "-rw-r--r--  29,495,296 2024/06/27 12:05:07 HEWL_NaI_3_2_0001.mccd"
        parts = line.split(None, 4)
        if len(parts) == 5 and parts[0].startswith("-"):   # regular files only
            entries.append((parts[4], int(parts[1].replace(",", ""))))
    return entries


def local_list(images: Path) -> list[tuple[str, int]]:
    """Same shape as rsync_list(), for images already on disk."""
    return [(str(p.relative_to(images)), p.stat().st_size)
            for p in sorted(images.rglob("*")) if p.is_file()]


def template_of(name: str) -> tuple[str, int] | None:
    """Turn 'dir/img_0042.cbf' into ('dir/img_####.cbf', 42), or None."""
    p = PurePosixPath(name)
    m = FRAME_RE.search(p.name)
    if not m:
        return None
    stem = p.name[:m.start()] + "#" * len(m.group()) + p.name[m.end():]
    parent = str(p.parent)
    return (stem if parent == "." else f"{parent}/{stem}"), int(m.group())


def pick_images(entries, exts, frames=None, template=None):
    """Decide which files make up the sweep to process.
    """
    images = [(n, s) for n, s in entries if PurePosixPath(n).suffix.lower() in exts]
    if not images:
        sys.exit(f"no image files with extensions {list(exts)} found")

    masters = [(n, s) for n, s in images
               if PurePosixPath(n).suffix.lower() in (".h5", ".nxs")
               and "master" in PurePosixPath(n).name.lower()]
    if masters and not template:
        best = max(masters, key=lambda e: e[1])[0]
        print(f"[auto] HDF5 master: {best}")
        return best, [best]

    # group the numbered frames by the template they belong to
    sweeps: dict[str, dict[int, str]] = {}
    for name, _ in images:
        parsed = template_of(name)
        if parsed:
            tmpl, index = parsed
            sweeps.setdefault(tmpl, {})[index] = name

    if template:
        if template not in sweeps:
            sys.exit(f"dataset.template {template!r} matches no files "
                     f"(available: {', '.join(sorted(sweeps)) or 'none'})")
        chosen = template
    else:
        if not sweeps:
            sys.exit("images found but none are sequentially numbered; "
                     "set dataset.template explicitly")
        chosen = max(sweeps, key=lambda t: len(sweeps[t]))
        if len(sweeps) > 1:
            others = ", ".join(sorted(set(sweeps) - {chosen}))
            print(f"[auto] {len(sweeps)} sweeps present, ignoring: {others}")

    numbered = sorted(sweeps[chosen].items())
    if frames:
        lo, hi = frames
        numbered = [(i, n) for i, n in numbered if lo <= i <= hi]
        if not numbered:
            sys.exit(f"no frames of {chosen} fall in the range {frames}")
    print(f"[auto] template: {chosen} "
          f"({len(numbered)} frames, {numbered[0][0]}..{numbered[-1][0]})")
    return chosen, [n for _, n in numbered]


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #

def fetch_files(base, files, images: Path, list_file: Path, dry_run: bool) -> bool:
    """Download exactly `files` from the rsync daemon.
    """
    images.mkdir(parents=True, exist_ok=True)
    list_file.write_text("\n".join(files) + "\n")
    cmd = ["rsync", "-a", "--partial", "--info=progress2",
           f"--files-from={list_file}", f"{base}/", f"{images}/"]
    if dry_run:
        cmd.insert(1, "-n")
    if run(cmd).returncode:
        sys.exit("rsync failed — check the SBGrid id, the host, and your network.")
    return not dry_run



def prepare_images(cfg, run_dir: Path, dry_run: bool) -> Path:
    """Make the images available locally and return the dials.import template.

    fetch: remote -> list the dataset on the server, choose the sweep, then
                     download only the frames that sweep needs
           none   -> use images already on disk under dataset.images_dir
    dataset.template pins the sweep when auto-detection picks the wrong one.
    """
    d = need(cfg, "dataset")
    exts = tuple(e.lower() for e in d.get("extensions", IMAGE_EXTS))
    frames = d.get("frames")
    template = d.get("template")

    if d.get("fetch", "remote") == "none":
        images = resolve(need(d, "images_dir", "dataset (fetch: none)"))
        if not images.is_dir():
            sys.exit(f"dataset.images_dir does not exist: {images}")
        chosen, _ = pick_images(local_list(images), exts, frames, template)
        return images / chosen

    host = d.get("host", "data.sbgrid.org")
    base = f"rsync://{host}/{SBGRID_MODULE}/{need(d, 'sbgrid_id', 'dataset')}"
    images = run_dir / "images"

    chosen, wanted = pick_images(rsync_list(base), exts, frames, template)
    downloaded = fetch_files(base, wanted + ["files.sha"], images,
                             run_dir / ".rsync-files", dry_run)
    return images / chosen


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #

@dataclass
class Step:
    name: str
    args: list[str]
    outputs: dict[str, str]

    @property
    def key(self) -> str:
        return self.name.split(".")[-1]

    @property
    def files(self) -> list[str]:
        return list(self.outputs.values())

    @property
    def cmd(self) -> list[str]:
        return [self.name, *self.args,
                *(f"{k}={v}" for k, v in self.outputs.items())]


def build_steps(cfg, template) -> list[Step]:
    g, p = need(cfg, "geometry"), need(cfg, "processing")
    n = need(p, "nproc", "processing")
    wav = [f"wavelengths.lam_min={p['wav_min']}",
           f"wavelengths.lam_max={p['wav_max']}",
           f"reciprocal_grid.d_min={p['d_min']}"]

    frames = need(cfg, "dataset").get("frames")
    image_range = ([f"geometry.scan.image_range={frames[0]},{frames[1]}"]
                   if frames else [])

    steps = [
        Step("dials.import", [
            f"geometry.scan.oscillation={g['oscillation'][0]},{g['oscillation'][1]}",
            f"geometry.goniometer.axes={','.join(map(str, g['goniometer_axes']))}",
            f"geometry.beam.wavelength={g['wavelength']}",
            f"geometry.detector.panel.pixel={g['pixel'][0]},{g['pixel'][1]}",
            f"input.template={template}",
            *image_range,
        ], {"output.experiments": "imported.expt"}),
        Step("laue.find_spots", [
            "imported.expt",
            f"spotfinder.mp.nproc={n}",
            f"spotfinder.threshold.dispersion.gain={p['gain']}",
            f"spotfinder.filter.max_separation={p['max_separation']}",
        ], {"output.reflections": "strong.refl"}),
        Step("laue.index", [
            "imported.expt", "strong.refl",
            f"indexer.indexing.nproc={n}",
            f"indexer.indexing.known_symmetry.space_group={p['space_group']}",
            "indexer.indexing.refinement_protocol.mode=refine_shells",
            "indexer.refinement.parameterisation.auto_reduction.action=fix",
            "laue_output.index_only=False",
        ], {"laue_output.indexed.experiments": "indexed.expt",
            "laue_output.indexed.reflections": "indexed.refl",
            "laue_output.final_output.experiments": "monochromatic.expt",
            "laue_output.final_output.reflections": "monochromatic.refl"}),
        Step("laue.sequence_to_stills",
             ["indexed.expt", "indexed.refl"],
             {"output.experiments": "stills.expt",
              "output.reflections": "stills.refl"}),
        Step("laue.assign",
             ["stills.refl", "stills.expt", *wav, f"nproc={n}"],
             {"output.experiments": "optimized.expt",
              "output.reflections": "optimized.refl"}),
        Step("laue.refine",
             ["optimized.expt", "optimized.refl", f"nproc={n}"],
             {"output.experiments": "poly_refined.expt",
              "output.reflections": "poly_refined.refl"}),
        Step("laue.predict",
             ["poly_refined.expt", "poly_refined.refl", *wav, f"nproc={n}"],
             {"output.reflections": "predicted.refl"}),
        Step("laue.integrate",
             ["poly_refined.expt", "predicted.refl", f"nproc={n}"],
             {"output.filename": "integrated.mtz"}),
    ]

    # per-step overrides from the config, keyed by the short step name
    extra = cfg.get("extra") or {}
    outputs = cfg.get("outputs") or {}
    known = {s.key for s in steps}
    for label, section in (("extra", extra), ("outputs", outputs)):
        for k in section:
            if k not in known:
                print(f"[warn] {label}: unknown step {k!r} — ignored "
                      f"(known steps: {', '.join(sorted(known))})")
    for s in steps:
        s.args += extra.get(s.key) or []
        s.outputs = {**s.outputs, **(outputs.get(s.key) or {})}
    return steps


def preflight(names) -> None:
    """Fail before the first long step if any tool is missing from PATH."""
    absent = sorted({n for n in names if shutil.which(n) is None})
    if absent:
        sys.exit(f"not on your PATH: {', '.join(absent)} — is the laue-dials "
                 f"environment activated?")


def toolchain() -> str:
    """Identify the installed DIALS/laue-dials so a version bump invalidates.

    Output filenames and algorithms both move between releases, so results
    produced by a different toolchain must not be silently reused.
    """
    p = subprocess.run(["laue.version"], capture_output=True, text=True)
    if p.returncode:
        print("[warn] could not run laue.version — results will not be "
              "invalidated by a toolchain change")
        return "unknown"
    return " ".join(p.stdout.split())


def fingerprint(*paths) -> str:
    """Identity of the files a stage reads, as their size and mtime.

    Hashing the contents would mean re-reading hundreds of megabytes of .refl
    on every run, and a step only ever changes these files by rewriting them,
    which moves both size and mtime_ns.  A file that is not there fingerprints
    as absent, so producing it later invalidates whatever ran without it.
    """
    out = []
    for p in sorted(str(x) for x in paths):
        try:
            s = Path(p).stat()
        except OSError:
            out.append(f"{p}:-")
        else:
            out.append(f"{p}:{s.st_size}:{s.st_mtime_ns}")
    return "|".join(out)


class Stamps:
    """The rolling stamp that decides which stages are already current.

    Every stage folds its own identity — its command line, and the size and
    mtime of the upstream files it reads — into one rolling hash, and records
    that hash once it succeeds.  A stage is skipped only when its output is on
    disk *and* the recorded hash still matches.

    Because the hash rolls forward, re-running any stage invalidates every
    stage after it.  Without that, a fresh laue.integrate left the previous
    run's shoeboxes, training and evaluation untouched and the summary table
    reported their numbers as if they described the new integration.
    """

    def __init__(self, root: Path, seed: str, force: list[str] | None):
        self.dir = root / ".stamps"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chain = seed
        self.force = force

    def advance(self, *parts) -> None:
        """Fold a stage in; call before asking whether that stage is current."""
        self.chain = hashlib.sha256(
            "\0".join([self.chain, *map(str, parts)]).encode()).hexdigest()

    def branch(self, *parts) -> "Stamps":
        """A stamp for one evaluation arm, hanging off the chain without
        advancing it: the arms are siblings of each other, not a sequence."""
        s = Stamps.__new__(Stamps)
        s.dir, s.force, s.chain = self.dir, self.force, self.chain
        s.advance(*parts)
        return s

    def forced(self, name: str) -> bool:
        """--force with no names forces everything; otherwise a stage matches
        by full name (careless:dials) or by group (careless, evaluate)."""
        if self.force is None:
            return False
        if not self.force:
            return True
        group = name.split(":")[0]
        return any(t in (name, group) or group in FORCE_GROUPS.get(t, ())
                   or name in FORCE_GROUPS.get(t, ()) for t in self.force)

    def stale(self, name: str, outputs=(), have=None) -> str | None:
        """Why `name` has to run, or None when its result is already current."""
        if self.forced(name):
            return "forced"
        if have is not None:
            if not have():
                return "no usable output on disk"
        else:
            absent = [o for o in outputs if not Path(o).exists()]
            if absent:
                return f"missing {Path(absent[0]).name}"
        stamp = self.dir / f"{name}.sha256"
        if not stamp.exists():
            return "no stamp on record"
        if stamp.read_text().strip() != self.chain:
            return "inputs or parameters changed"
        return None

    def begin(self, name: str, outputs=(), have=None) -> bool:
        """True when the stage has to run, printing the reason either way."""
        why = self.stale(name, outputs, have)
        print(f"[skip] {name}" if why is None else f"[run ] {name} ({why})")
        return why is not None

    def record(self, name: str) -> None:
        (self.dir / f"{name}.sha256").write_text(self.chain)


def run_pipeline(steps, proc: Path, force, dry_run: bool) -> Stamps:
    """Run the laue-dials steps in order, skipping those already current.

    The chain is seeded with the toolchain version and rolls over each step's
    command line, so a version bump or a changed parameter reruns that step and
    everything after it.  It is handed back so the integrator and evaluation
    stages can continue the same chain rather than start their own.
    """
    proc.mkdir(parents=True, exist_ok=True)
    chain = toolchain()
    print(f"toolchain: {chain}")
    st = Stamps(proc, chain, force)

    for s in steps:
        st.advance(*s.cmd)
        if not st.begin(s.name, [proc / o for o in s.files]):
            continue
        if dry_run:
            print("[plan]", " ".join(s.cmd))
            continue

        if subprocess.run(s.cmd, cwd=proc).returncode:
            sys.exit(f"{s.name} failed (see its DIALS log in {proc})")
        missing = {k: v for k, v in s.outputs.items()
                   if not (proc / v).exists()}
        if missing:
            sys.exit(f"{s.name} finished but did not write "
                     f"{sorted(missing.values())}. Its output PHIL parameters "
                     f"({', '.join(missing)}) may have been renamed in your "
                     f"laue-dials version — check `{s.name} -c -e3` and set "
                     f"outputs.{s.key} in the config accordingly.")
        st.record(s.name)
    return st


# --------------------------------------------------------------------------- #
# Downstream: shoeboxes -> training -> prediction
# --------------------------------------------------------------------------- #

def checkpoints_of(train_dir: Path) -> list[Path]:
    """Checkpoints integrator.predict would find for this run dir, if any.

    `run_paths.yaml` is written before training starts, so it says nothing about
    whether a checkpoint was ever reached; only the .ckpt files themselves do.
    """
    meta = train_dir / "run_paths.yaml"
    if not meta.exists():
        return []
    m = yaml.safe_load(meta.read_text()) or {}
    log_dir = m.get("log_dir") or (m.get("wandb") or {}).get("log_dir")
    if not log_dir:
        return []
    return sorted(Path(log_dir).glob("**/epoch*.ckpt"))


def run_stage(name, cmd, done, st: Stamps, dry_run: bool, inputs=()) -> None:
    """Run one integrator command unless it is already current.

    `done` is either a path that must exist or a predicate returning True when
    the stage's real output is already on disk.  `inputs` are the upstream
    files the command reads: their size and mtime go into the stamp, so
    re-running an earlier stage makes this one stale too, instead of the output
    of a superseded run being silently reused.
    """
    st.advance(*[str(c) for c in cmd], fingerprint(*inputs))
    have = done if callable(done) else (lambda: done.exists())
    if not st.begin(name, have=have):
        return
    if dry_run:
        print("[plan]", " ".join(str(c) for c in cmd))
        return
    if run(cmd).returncode:
        sys.exit(f"{name} failed")
    st.record(name)


def make_shoeboxes(cfg, proc: Path, run_dir: Path,
                   st: Stamps, dry_run: bool) -> Path | None:
    """Turn the integrated reflections into a PyTorch shoebox dataset."""
    s = cfg.get("shoeboxes")
    if s is None:
        return None

    frames = need(cfg, "dataset").get("frames")
    max_images = s.get("max_images") or (frames[1] + 1 if frames else None)
    if max_images is None:
        sys.exit("shoeboxes: set max_images (or dataset.frames)")

    out = run_dir / "shoebox_data"
    refl, expt = s.get("refl", "predicted.refl"), s.get("expt", "poly_refined.expt")
    cmd = ["integrator.make_shoeboxes", "--laue",
           "--data-dir", proc,
           "--refl", refl,
           "--expt", expt,
           "--out-dir", out,
           "--w", s.get("w", 21), "--h", s.get("h", 21), "--d", 1,
           "--max-images", max_images,
           "--test-fraction", s.get("test_fraction", 0.1)]
    run_stage("integrator.make_shoeboxes", cmd, out / "dataset.yaml",
              st, dry_run, inputs=[proc / refl, proc / expt])
    return out


def train_model(cfg, data_dir: Path | None, run_dir: Path,
                st: Stamps, dry_run: bool) -> Path | None:
    """Train the integrator on the shoebox dataset."""
    t = cfg.get("train")
    if t is None or data_dir is None:
        return None

    # the training config is read by name, so its *contents* have to enter the
    # stamp too; editing it in place must not leave the old checkpoints current
    conf = resolve(need(t, "config", "train"))
    out = run_dir / "run"
    cmd = ["integrator.train",
           "--config", conf,
           "--data-dir", data_dir,
           "--run-dir", out]
    run_stage("integrator.train", cmd, lambda: bool(checkpoints_of(out)),
              st, dry_run,
              inputs=[conf, data_dir / "dataset.yaml",
                      *sorted(data_dir.glob("*.npy"))])
    return out


def predict(cfg, train_dir: Path | None, st: Stamps, dry_run: bool) -> None:
    """Predict from every checkpoint written during training."""
    p = cfg.get("predict")
    if p is None or train_dir is None:
        return

    if not checkpoints_of(train_dir) and not dry_run:
        sys.exit(f"integrator.predict: no checkpoints under {train_dir} — "
                 f"training did not get far enough to write one (see its log). "
                 f"Rerun with --force, or lower checkpoint.every_n_epochs.")

    cmd = ["integrator.predict", "--run-dir", train_dir]
    if p.get("write_mtz", True):
        cmd.append("--write-mtz")
    run_stage("integrator.predict", cmd,
              train_dir / "predictions" / "test_preds_all.parquet",
              st, dry_run, inputs=checkpoints_of(train_dir))


# --------------------------------------------------------------------------- #
# Evaluation: careless -> phenix.refine + rs.find_peaks -> plots
# --------------------------------------------------------------------------- #
# The integrator predicts *unmerged* intensities, and neither R-values nor
# anomalous peak heights can be computed from those: R needs one F_obs per
# unique reflection to compare against F_calc, and a peak height needs an
# anomalous difference map.  Both wait on scaling and merging, which is careless'
# job here.  Each MTZ evaluated is an "arm": the DIALS reference and a selection
# of checkpoints take the identical path, so their numbers are comparable.

def eval_imports():
    """merge_eval + plot_ckpt_eval from scripts/, imported only when evaluating.

    They pull in torch and matplotlib, which the laue-dials half of this script
    has no use for.
    """
    for p in (SCRIPTS, SCRIPTS / "merging"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import merge_eval
    import plot_ckpt_eval
    return merge_eval, plot_ckpt_eval


def eval_root(train_dir: Path) -> Path:
    """Where the evaluation writes, beside the run's other heavy outputs."""
    meta = train_dir / "run_paths.yaml"
    m = yaml.safe_load(meta.read_text()) if meta.exists() else {}
    return Path((m or {}).get("output_root") or train_dir) / "ckpt_eval"


def select_checkpoints(dirs: list[Path], n: int) -> list[Path]:
    """`n` evenly spaced checkpoints ending on the last, never the first.

    The first checkpoint is barely-trained and costs a full careless run to
    learn nothing, so it is dropped before spacing.  n <= 0 keeps every
    checkpoint but that first one.
    """
    pool = dirs[1:] if len(dirs) > 1 else dirs
    if n <= 0 or n >= len(pool):
        return pool
    if n == 1:
        return [pool[-1]]
    step = (len(pool) - 1) / (n - 1)
    return [pool[round(i * step)] for i in range(n)]


def find_arms(pred_dir: Path, dials_mtz: Path | None, root: Path,
              n_ckpt: int, ckpts=()) -> list[tuple[str, Path, Path]]:
    """(label, unmerged MTZ, work dir) for the DIALS reference and checkpoints.

    A checkpoint contributes an arm only if integrator.predict wrote its MTZ;
    the epoch_NNNN work dir name is what the plotting step keys epochs on.

    integrator.predict writes into a directory it does not clear first, so a
    retrain that lands on different epoch numbers leaves the previous run's
    epoch dirs sitting beside the new ones.  Those are dropped here rather than
    entering the table as arms: an epoch_NNNN older than the checkpoint it
    claims to come from would put two different trainings in one comparison.
    """
    arms = []
    if dials_mtz:
        arms.append(("dials", dials_mtz, root / "dials"))

    epochs = sorted(d for d in pred_dir.glob("epoch_*") if d.is_dir())
    found = [(d, next(iter(sorted(d.glob("preds_epoch_*.mtz"))), None))
             for d in epochs]
    havemtz = [(d, m) for d, m in found if m]
    if len(havemtz) < len(found):
        print(f"[warn] {len(found) - len(havemtz)} epoch dir(s) have no "
              f"preds_epoch_*.mtz (was integrator.predict run with --write-mtz?)")

    if ckpts:
        by_epoch = {int(m.group(1)): c for c in ckpts
                    if (m := EPOCH_RE.search(c.name))}
        fresh, stale = [], []
        for d, mtz in havemtz:
            m = EPOCH_RE.search(d.name)
            ckpt = by_epoch.get(int(m.group(1))) if m else None
            if ckpt is None:
                stale.append(f"{d.name} (no checkpoint of that epoch remains)")
            elif mtz.stat().st_mtime_ns < ckpt.stat().st_mtime_ns:
                stale.append(f"{d.name} (predates {ckpt.name})")
            else:
                fresh.append((d, mtz))
        if stale:
            print(f"[warn] ignoring {len(stale)} epoch dir(s) left over from an "
                  f"earlier training: {', '.join(stale)}")
        havemtz = fresh

    by_dir = dict(havemtz)
    chosen = select_checkpoints([d for d, _ in havemtz], n_ckpt)
    return arms + [(d.name, by_dir[d], root / d.name) for d in chosen]


def normalize(src: Path, out: Path) -> None:
    """Rewrite careless output with the column names phenix.eff selects.

    careless writes its structure factor posterior as `F`/`SigF`, and
    `--anomalous` merges into the anomalous ASU rather than splitting columns,
    so the Friedel mates arrive as separate rows.  The eff template selects
    `F(+),SIGF(+),F(-),SIGF(-)`, so unstack them and uppercase SIG.
    """
    import reciprocalspaceship as rs

    ds = rs.read_mtz(str(src))
    if {"H", "K", "L"} <= set(ds.columns):
        ds = ds.set_index(["H", "K", "L"])
    ds = ds.rename(columns={c: c.replace("Sig", "SIG") for c in ds.columns})

    if "F(+)" not in ds.columns:
        # unstack_anomalous refuses unmerged data; careless just merged this,
        # whatever the MERGED record in the file it wrote says.
        ds.merged = True
        ds = ds.unstack_anomalous()
        ds = ds.rename(columns={c: c.replace("Sig", "SIG") for c in ds.columns})

    want = ["F(+)", "SIGF(+)", "F(-)", "SIGF(-)"]
    missing = [c for c in want if c not in ds.columns]
    if missing:
        sys.exit(f"{src}: careless output has no {missing}; its columns are "
                 f"{list(ds.columns)}. Adjust normalize() to match this "
                 f"careless version.")
    ds = ds[want + [c for c in ds.columns if c not in want]]
    ds.write_mtz(str(out), skip_problem_mtztypes=True)
    print(f"[ok  ] normalized -> {out} ({len(ds)} reflections)")


def careless_merge(label: str, mtz: Path, work: Path, e: dict,
                   st: Stamps) -> Path:
    """Scale and merge one unmerged MTZ with careless.

    The arm's stamp covers the unmerged MTZ it reads, so a re-integrated
    integrated.mtz or a re-predicted checkpoint remerges rather than leaving
    the previous run's numbers standing.
    """
    work.mkdir(parents=True, exist_ok=True)
    merged = work / "careless_merged.mtz"
    base = work / "careless" / "out"
    # careless pulls TensorFlow, which pins numpy<2.1 while integrator needs
    # >=2.3, so it usually lives in its own env; careless_bin points at it.
    cmd = [e.get("careless_bin", "careless"), "poly", "--anomalous",
           f"--iterations={e.get('iterations', 10000)}",
           # careless defaults this to 'Wavelength'; laue-dials writes lowercase
           f"--wavelength-key={e.get('wavelength_key', 'wavelength')}"]
    if e.get("dmin"):
        cmd.append(f"--dmin={e['dmin']}")
    cmd += [str(a) for a in (e.get("careless_args") or [])]
    cmd += [e.get("metadata_keys", METADATA_KEYS), mtz, base]

    arm = st.branch(*[str(c) for c in cmd], fingerprint(mtz))
    name = f"careless:{label}"
    if not arm.begin(name, [merged]):
        return merged

    base.parent.mkdir(parents=True, exist_ok=True)
    if run(cmd).returncode:
        sys.exit(f"careless failed on {label} (see the log above)")

    # careless writes <base>_<i>.mtz per input file, alongside _xval_<i>.mtz and
    # _predictions_<i>.mtz; the digit glob keeps those out of the way.
    written = sorted(base.parent.glob(f"{base.name}_[0-9]*.mtz"))
    if not written:
        sys.exit(f"careless wrote no {base.name}_<i>.mtz in {base.parent}")
    normalize(written[0], merged)
    arm.record(name)
    return merged


def make_flags(src: Path, out: Path, e: dict, st: Stamps) -> Path:
    """One R-free flag set, generated once and shared by every arm.

    Fresh random flags per arm would make R-free incomparable across them, which
    is the whole point of the comparison; the flags cover the full ASU to dmin,
    so the same set joins onto each arm's slightly different HKL list.

    They are regenerated when the arm they are drawn from changes: flags from a
    superseded HKL list would silently stop covering part of the new one.
    """
    cmd = ["rs.rfree", "-f", src, "-r", e.get("rfree_fraction", 0.05),
           "-s", e.get("rfree_seed", 0), "-o", out]
    arm = st.branch(*[str(c) for c in cmd], fingerprint(src))
    if not arm.begin("flags", [out]):
        return out
    if run(cmd).returncode:
        sys.exit("rs.rfree failed")
    arm.record("flags")
    return out


def attach_flags(src: Path, flags: Path, out: Path) -> Path:
    """Join the shared R-free flags onto an arm's merged MTZ.

    Skipped when the join is already newer than both its inputs: rewriting it
    every run would move its mtime and make refine look stale forever.
    """
    if out.exists() and out.stat().st_mtime_ns >= max(
            src.stat().st_mtime_ns, flags.stat().st_mtime_ns):
        return out
    import reciprocalspaceship as rs

    ds = rs.read_mtz(str(src))
    fl = rs.read_mtz(str(flags))
    ds["R-free-flags"] = fl[fl.columns[0]].reindex(ds.index).fillna(0)
    ds["R-free-flags"] = ds["R-free-flags"].astype("int32").astype(
        rs.MTZIntDtype())
    ds.write_mtz(str(out), skip_problem_mtztypes=True)
    return out


def peak_stats(peaks_csv: Path | None) -> dict:
    """Top peak height and count from an rs.find_peaks CSV."""
    if peaks_csv is None or not peaks_csv.exists():
        return {"top_anom_peak": None, "n_anom_peaks": 0}
    import pandas as pd

    df = pd.read_csv(peaks_csv)
    if df.empty:
        return {"top_anom_peak": None, "n_anom_peaks": 0}
    col = "peakz" if "peakz" in df.columns else df.columns[-1]
    return {"top_anom_peak": float(df[col].max()), "n_anom_peaks": len(df)}


def refine(label: str, work: Path, template: str, st: Stamps) -> dict:
    """phenix.refine + rs.find_peaks on this arm, into its result.json.

    Everything from the MTZ onward is merge_eval's, so an arm evaluated here and
    a checkpoint evaluated by scripts/merging/process_single_ckpt.py produce the
    same numbers in the same layout.
    """
    me, _ = eval_imports()
    result_path = work / "result.json"
    mtz = work / "merged.mtz"

    def done() -> bool:
        """A result on disk only counts if phenix converged on it."""
        if not result_path.exists():
            return False
        res = json.loads(result_path.read_text())
        return bool((res.get("variants") or {})
                    .get(VARIANT, {}).get("phenix_ok"))

    arm = st.branch(label, template, fingerprint(mtz))
    name = f"refine:{label}"
    if not arm.begin(name, have=done):
        return json.loads(result_path.read_text())

    _, labels, star, fw = next(v for v in me.VARIANTS if v[0] == VARIANT)
    out = work / VARIANT
    out.mkdir(exist_ok=True)
    eff = out / "phenix.eff"
    eff.write_text(me.render_eff(template, mtz, labels, star, fw))

    ok = me.run_phenix_refine(eff, out, mtz)
    r = me.parse_phenix_r_factors(out) if ok else {}
    peaks = me.run_find_peaks(out) if ok else None

    m = EPOCH_RE.search(label)
    res = {
        "epoch": int(m.group(1)) if m else -1,
        "label": label,
        "mtz": str(mtz),
        "variants": {VARIANT: {
            "phenix_ok": ok,
            "r_work": r.get("r_work_final"),
            "r_free": r.get("r_free_final"),
            "r_work_start": r.get("r_work_start"),
            **peak_stats(peaks),
        }},
    }
    result_path.write_text(json.dumps(res, indent=2))
    arm.record(name)
    return res


def eval_plots(root: Path, dials_dir: Path | None) -> None:
    """R-factors and per-residue anomalous peaks over the evaluated epochs.

    The DIALS arm lives outside the epoch_* dirs the plots iterate, so it enters
    as reference lines: its peak height per site, its R-factors.
    """
    _, pce = eval_imports()

    ref_peaks = ref_vals = None
    if dials_dir:
        peaks = dials_dir / VARIANT / "peaks.csv"
        ref_peaks = str(peaks) if peaks.exists() else None
        result = dials_dir / "result.json"
        if result.exists():
            v = json.loads(result.read_text())["variants"][VARIANT]
            ref_vals = {"r_work": v.get("r_work"), "r_free": v.get("r_free")}

    saved = pce.make_all_plots(root, ref_peaks=ref_peaks, ref_vals=ref_vals)
    print("\n".join(["plots:"] + [f"  {p}" for p in saved]))


def eval_summary(results: list[dict], out: Path) -> None:
    """One table over all arms: what the whole pipeline was run to produce."""
    head = f"{'arm':<12} {'Rwork':>7} {'Rfree':>7} {'top peak':>9} {'peaks':>6}"
    rows = [head, "-" * len(head)]
    for res in results:
        v = res["variants"][VARIANT]
        rows.append(f"{res['label']:<12} {num(v['r_work']):>7} "
                    f"{num(v['r_free']):>7} "
                    f"{num(v['top_anom_peak'], '{:.2f}'):>9} "
                    f"{v['n_anom_peaks']:>6}")
    text = "\n".join(rows)
    out.write_text(text + "\n")
    print("\n" + text)
    print(f"\nsummary : {out}")


def num(x, fmt="{:.4f}") -> str:
    """Format a result number; a failed step leaves None or NaN behind."""
    if not isinstance(x, (int, float)) or math.isnan(x):
        return "-"
    return fmt.format(x)


def evaluate(cfg, proc: Path, train_dir: Path | None,
             st: Stamps, dry_run: bool) -> None:
    """Merge every arm with careless, refine it, then plot the comparison."""
    e = cfg.get("evaluate")
    if e is None or train_dir is None:
        return
    template = resolve(need(e, "eff_template", "evaluate"))
    pdb = resolve(need(e, "pdb", "evaluate"))
    # a different model or refinement template gives different R-factors, so
    # both belong in every arm's stamp
    st.advance("evaluate", fingerprint(pdb, template))
    import os

    os.environ["PHENIX_ENV"] = need(e, "phenix_env", "evaluate")

    dials_mtz = proc / "integrated.mtz" if e.get("dials_reference", True) else None
    root = eval_root(train_dir)
    if dry_run:
        print(f"[plan] careless + phenix.refine + rs.find_peaks per arm "
              f"(DIALS reference{'' if dials_mtz else ' off'}, "
              f"{e.get('n_checkpoints', 4)} checkpoints) -> {root}")
        return

    root.mkdir(parents=True, exist_ok=True)
    meta = yaml.safe_load((train_dir / "run_paths.yaml").read_text())
    pred_dir = Path(meta.get("predictions_dir")
                    or Path(meta.get("output_root", train_dir)) / "predictions")
    arms = find_arms(pred_dir, dials_mtz, root, e.get("n_checkpoints", 4),
                     ckpts=checkpoints_of(train_dir))
    if not arms:
        sys.exit(f"evaluate: nothing to evaluate — no preds_epoch_*.mtz under "
                 f"{pred_dir} and no DIALS reference")
    print(f"arms    : {', '.join(label for label, _, _ in arms)}")
    print(f"model   : {pdb}")

    # 1. merge every arm first, so one flag set can be shared across them
    merged = [(label, careless_merge(label, mtz, work, e, st), work)
              for label, mtz, work in arms]

    flags = make_flags(merged[0][1], root / "rfree.mtz", e, st)

    # 2. refine each arm against the same flags and model
    results = []
    for label, src, work in merged:
        attach_flags(src, flags, work / "merged.mtz")
        results.append(refine(label, work, template.read_text(), st))

    # 3. plots + one summary table
    eval_plots(root, next((w for lbl, _, w in arms if lbl == "dials"), None))
    eval_summary(results, root / "eval_summary.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--name", help="override dataset name")
    ap.add_argument("--base-dir", help="override base directory")
    ap.add_argument("--force", nargs="*", metavar="STAGE",
                    help="rerun stages even when they look current: with no "
                         "names, everything; otherwise name a step "
                         "(laue.integrate, integrator.train), an evaluation "
                         "arm (careless:dials, refine:epoch_0019) or a group "
                         "(" + ", ".join(sorted(FORCE_GROUPS)) + ")")
    ap.add_argument("--dry-run", action="store_true", help="print the plan only")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    cfg["name"] = a.name or need(cfg, "name")
    cfg["base_dir"] = a.base_dir or need(cfg, "base_dir")
    # None = nothing forced, [] = force everything, [...] = force those stages
    force = a.force if a.force is not None else (
        [] if cfg.get("force", False) else None)

    run_dir = resolve(Path(cfg["base_dir"]) / cfg["name"])
    proc = run_dir / "proc"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir : {run_dir}")

    steps = build_steps(cfg, "")
    downstream = {"shoeboxes": "integrator.make_shoeboxes",
                  "train": "integrator.train",
                  "predict": "integrator.predict"}
    preflight([s.name for s in steps]
              + [t for k, t in downstream.items() if cfg.get(k) is not None]
              # careless is a separate install, usually its own env
              + ([(cfg["evaluate"] or {}).get("careless_bin", "careless"),
                  "rs.rfree", "rs.find_peaks"]
                 if cfg.get("evaluate") is not None else []))
    if cfg.get("evaluate") is not None:
        # The refinement inputs, checked here rather than after training
        for key in ("eff_template", "pdb", "phenix_env"):
            p = resolve(need(cfg["evaluate"], key, "evaluate"))
            if not p.exists():
                sys.exit(f"evaluate: {key} not found: {p}")

    template = prepare_images(cfg, run_dir, a.dry_run)
    print(f"template: {template}")
    # one rolling stamp threads all four halves: reprocessing invalidates
    # training, retraining invalidates prediction, and so on to the R-factors
    st = run_pipeline(build_steps(cfg, template), proc, force, a.dry_run)

    data_dir = make_shoeboxes(cfg, proc, run_dir, st, a.dry_run)
    train_dir = train_model(cfg, data_dir, run_dir, st, a.dry_run)
    predict(cfg, train_dir, st, a.dry_run)
    evaluate(cfg, proc, train_dir, st, a.dry_run)
    if not a.dry_run:
        print(f"done -> {run_dir}")


if __name__ == "__main__":
    main()
