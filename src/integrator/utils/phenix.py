"""Put phenix on PATH, and read the columns it writes.

Phenix 2.0 is the version this project refines with: it is what the June
merging runs used, and it is the only install that survived the storage
migration intact.  Its own `phenix_env.sh` cannot be used.  That script
exports the pre-migration prefix `/n/hekstra_lab/garden_2/phenix-2.0-5729`
and prepends a `phenix_bin/` whose entries are symlinks into that dead tree,
so sourcing it leaves `phenix.refine: command not found` while reporting
success.  The real wrappers are in `bin/`, so the prefix is set here instead
of being sourced.

Phenix 2.0 also renamed the anomalous-difference phase column: it writes
`PANOM` where 1.20 wrote `PHANOM`, and `rs.find_peaks -p PHANOM` on a 2.0 MTZ
raises `KeyError: None of [Index(['PHANOM'])]`.  Ask `anomalous_columns` for
the label rather than naming one.
"""

import os
from pathlib import Path

DEFAULT_PREFIX = "/n/lab_storage/hekstra_lab/garden_2/phenix-2.0-5729"
DEFAULT_VERSION = "2.0-5729"

# in the order phenix has used them; the first one present in an MTZ wins
PHASE_LABELS = ("PANOM", "PHANOM")


def prefix(explicit: str | os.PathLike | None = None) -> str:
    """The phenix install to use: explicit, then `$PHENIX_PREFIX`, then 2.0."""
    return str(explicit or os.environ.get("PHENIX_PREFIX") or DEFAULT_PREFIX)


def version(explicit: str | None = None) -> str:
    """The version string phenix tools expect in `$PHENIX_VERSION`."""
    return explicit or os.environ.get("PHENIX_VERSION") or DEFAULT_VERSION


def shell_prelude(
    env_script: str | os.PathLike | None = None,
    install: str | os.PathLike | None = None,
) -> str:
    """A shell fragment that puts phenix on PATH, to prepend with ` && `.

    Args:
        env_script: A `phenix_env.sh` to source instead.  Only for the 1.20
            install, whose env script still works; 2.0's does not.
        install: A phenix prefix directory.  Defaults to `prefix()`.

    Returns:
        A fragment such as `export PHENIX=... PATH=.../bin:$PATH`, or
        `source <env_script>` when one is given.
    """
    if env_script is not None and str(env_script).endswith(".sh"):
        return f'source "{env_script}" >/dev/null 2>&1'

    p = Path(prefix(install or env_script))
    return (
        f'export PHENIX="{p}" PHENIX_PREFIX="{p}" '
        f'PHENIX_VERSION="{version()}" PATH="{p}/bin:$PATH"'
    )


def anomalous_columns(mtz_path: str | os.PathLike) -> tuple[str, str] | None:
    """`(amplitude, phase)` labels of the anomalous-difference map in an MTZ.

    Returns None when the MTZ has no anomalous map, which is the common case
    for a refinement that was handed no anomalous data.
    """
    import gemmi

    try:
        labels = gemmi.read_mtz_file(str(mtz_path)).column_labels()
    except Exception:
        return None
    if "ANOM" not in labels:
        return None
    for ph in PHASE_LABELS:
        if ph in labels:
            return "ANOM", ph
    return None


def find_anomalous_map(refine_dir: str | os.PathLike) -> tuple[Path, str] | None:
    """The MTZ in `refine_dir` carrying an anomalous map, and its phase label.

    The serial number in a phenix output filename varies with the run, so this
    scans by column label instead of guessing the name.
    """
    for mtz in sorted(Path(refine_dir).glob("*.mtz")):
        cols = anomalous_columns(mtz)
        if cols is not None:
            return mtz, cols[1]
    return None


def render_eff(template: str | os.PathLike, out: str | os.PathLike, **subs) -> Path:
    """Write `template` to `out` with `$NAME` placeholders substituted.

    Phenix 2.0 takes its files through the `data_manager` scope, where each
    file sits in the same block as the labels that select its columns.  A
    `data_manager.miller_array.file=` on the command line opens a *new* block
    with no labels instead of filling that one in, so the file has to go into
    the eff itself.  Paths are absolutized, since phenix runs in the output
    directory.

    Args:
        out: Destination path, created with parents.
        **subs: `MTZFILE="..."` substitutes `$MTZFILE`.
    """
    text = Path(template).read_text()
    for key, value in subs.items():
        text = text.replace(f"${key}", str(Path(value).resolve()))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


def eff_takes_substitution(template: str | os.PathLike) -> bool:
    """Whether an eff is a 2.0-form template rather than a 1.20 eff.

    A 1.20 eff names no files -- the pipeline passed them as
    `refinement.input.*.file_name` overrides, which 2.0 still accepts for the
    model but which cannot carry column labels.
    """
    return "$MTZFILE" in Path(template).read_text()


def requested_phase_label(eff: str | os.PathLike) -> str:
    """The anomalous phase label an eff asks phenix to write.

    An eff that names `mtz_label_phases` for its `map_type = "anomalous"` map
    gets that label whatever the version; only an unnamed map follows the
    version default, which changed from `PHANOM` to `PANOM` in 2.0.  Use this
    when the peak search has to be composed before the MTZ exists.
    """
    text = Path(eff).read_text()
    blocks = text.split("map_coefficients")
    for block in blocks[1:]:
        if "anomalous" not in block.split("map_type")[1].split("\n")[0]:
            continue
        for line in block.splitlines():
            if "mtz_label_phases" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return PHASE_LABELS[0]
