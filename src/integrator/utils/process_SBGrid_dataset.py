#!/usr/bin/env python3
"""Fetch an SBGrid dataset and run it through the laue-dials pipeline.

    python process_SBGrid_dataset.py -c ../../../configs/poly/laue_dials_processing_config.yaml

All input parameters are specified in the YAML config.
"""

from __future__ import annotations

import argparse
import hashlib
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
CHUNK = 1 << 20

LAUNCH_CWD = Path.cwd()


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


def verify_checksums(images: Path, files) -> None:
    """Check the downloaded frames against the dataset's files.sha (SHA-1)."""
    manifest = images / "files.sha"
    if not manifest.exists():
        print("[warn] no files.sha in this dataset — skipping verification")
        return

    expected = {}
    for line in manifest.read_text().splitlines():
        digest, _, path = line.partition("  ")
        if digest and path:
            expected[path.removeprefix("./")] = digest

    bad, unlisted = [], 0
    for rel in files:
        want = expected.get(rel)
        if want is None:
            unlisted += 1
            continue
        h = hashlib.sha1()
        with open(images / rel, "rb") as fh:
            for chunk in iter(lambda: fh.read(CHUNK), b""):
                h.update(chunk)
        if h.hexdigest() != want:
            bad.append(rel)
    if bad:
        sys.exit(f"checksum mismatch for {len(bad)} file(s), e.g. {bad[:3]}; "
                 f"delete them and rerun")
    if unlisted:
        print(f"[warn] {unlisted} file(s) not listed in files.sha")
    print(f"[ok  ] {len(files) - unlisted} file(s) match files.sha")


def prepare_images(cfg, run_dir: Path, dry_run: bool, verify: bool) -> Path:
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
    if downloaded and verify:
        verify_checksums(images, wanted)
    return images / chosen


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #

@dataclass
class Step:
    name: str
    args: list[str]
    outputs: dict[str, str]      # PHIL parameter -> the filename we demand

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


def preflight(steps) -> None:
    """Fail before the first long step if any tool is missing from PATH."""
    absent = sorted({s.name for s in steps if shutil.which(s.name) is None})
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


def run_pipeline(steps, proc: Path, force: bool, dry_run: bool) -> None:
    """Run the steps in order, skipping those whose result is already current.

    A step is current only if its outputs exist *and* its stamp matches.  The
    stamp is a rolling hash over the toolchain version, this step's command
    line and every command line before it, so changing e.g. wav_min or
    upgrading laue-dials invalidates that step and all its descendants instead
    of being silently skipped.
    """
    proc.mkdir(parents=True, exist_ok=True)
    stamps = proc / ".stamps"
    stamps.mkdir(exist_ok=True)

    chain = toolchain()
    print(f"toolchain: {chain}")
    for s in steps:
        chain = hashlib.sha256("\0".join([chain, *s.cmd]).encode()).hexdigest()
        stamp = stamps / f"{s.name}.sha256"
        have_outputs = all((proc / o).exists() for o in s.files)
        current = stamp.exists() and stamp.read_text().strip() == chain

        if have_outputs and current and not force:
            print(f"[skip] {s.name}")
            continue
        if have_outputs and not force:
            reason = "parameters changed" if stamp.exists() else "no stamp on record"
            print(f"[stale] {s.name}: {reason}, rerunning")
        if dry_run:
            print("[plan]", " ".join(s.cmd))
            continue

        print(f"[run ] {s.name}")
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
        stamp.write_text(chain)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--name", help="override dataset name")
    ap.add_argument("--base-dir", help="override base directory")
    ap.add_argument("--force", action="store_true", help="rerun everything")
    ap.add_argument("--dry-run", action="store_true", help="print the plan only")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the files.sha checksum pass")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    cfg["name"] = a.name or need(cfg, "name")
    cfg["base_dir"] = a.base_dir or need(cfg, "base_dir")
    force = a.force or cfg.get("force", False)
    verify = not a.no_verify and cfg.get("verify", True)

    run_dir = resolve(Path(cfg["base_dir"]) / cfg["name"])
    proc = run_dir / "proc"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir : {run_dir}")

    steps = build_steps(cfg, "")          # cheap: validates the config first
    preflight(steps)

    template = prepare_images(cfg, run_dir, a.dry_run, verify)
    print(f"template: {template}")
    run_pipeline(build_steps(cfg, template), proc, force, a.dry_run)
    if not a.dry_run:
        print(f"done -> {proc}")


if __name__ == "__main__":
    main()
