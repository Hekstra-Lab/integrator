"""One HTML page showing every figure currently rendered, in spec order.

Generalized from `finished_figs_local/scripts/build_gallery.py`, which
hardcoded the five `d_<dataset>` directories of the finished-model run. This
one walks a figure directory and sorts what it finds into the sections of
the reported figure list in `docs/figures/README.md`, so a figure added
later appears without editing anything here -- and one that is still missing
shows up as a gap rather than silently not being in the page.

PNGs are inlined as data URIs, so the page is one self-contained file that
can be opened anywhere. It is deliberately NOT published: these are
unreleased measurements.

    uv run python scripts/figures/build_gallery.py
    uv run python scripts/figures/build_gallery.py docs/figures/m9 out.html
"""

from __future__ import annotations

import base64
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (section title, note, predicate on the file stem) in the order the figures
# are reported. The first rule that matches a file claims it.
RULES: list[tuple[str, str, Callable[[str], bool]]] = [
    (
        "Input shoeboxes against the predicted profiles",
        "One row per reflection across the DIALS summed-intensity quantiles: "
        "the observed counts, then each profile model's $q(p)$ mean on the "
        "same reflection. Under each panel, n_eff = 1/sum(p^2) and the mass "
        "outside the central 5x5.",
        lambda s: s.startswith("shoebox_profiles"),
    ),
    (
        "Training loss",
        "The ELBO and its parts -- Poisson NLL and the profile, intensity "
        "and background KL terms -- train against validation. No test split: "
        "loss_history.csv carries train and val only.",
        lambda s: s.startswith("loss_"),
    ),
    (
        "Model against the reference integrator, per reflection",
        "Intensity and background at the last checkpoint, and the same two "
        "correlations within resolution shells over training (one line per "
        "epoch on the ramp).",
        lambda s: s.startswith(("scatter_", "cc_resolution_")),
    ),
    (
        "Merging statistics by resolution shell",
        "CC-half, R-pim, CC-anom and I/sigma, one line per checkpoint on the "
        "sequential ramp, over the DIALS reference in a categorical colour "
        "so it cannot be read as a checkpoint.",
        lambda s: s.startswith("merging_"),
    ),
    (
        "Refinement over training",
        "R-work and R-free from phenix.refine on the integrator-merged data, "
        "against the DIALS reference line.",
        lambda s: s.startswith("refinement"),
    ),
    (
        "Anomalous peak heights over training",
        "Per-site peak height in sigma at every checkpoint, over the flat "
        "DIALS reference, plus the mean over the anomalous scatterers "
        "(IOD, CYS, MET, SER) -- waters sit at the noise floor and are "
        "excluded from that mean.",
        lambda s: s == "peakz_mean" or _is_site(s),
    ),
    (
        "Polychromatic: the learned spectrum",
        "G(lambda) and log G(lambda) over 21 checkpoints of hewl1118. "
        "Polychromatic figures are drawn at one checkpoint by default; the "
        "spectrum is free because it reads checkpoints, not reflections.",
        lambda s: s.startswith("spectrum"),
    ),
    (
        "Polychromatic: merging statistics from careless.stats",
        "CC-half, CC-anom, CC-pred, R-split, I/sigma and completeness by "
        "resolution shell. These come from careless rather than from "
        "dials.merge, because the polychromatic arm scales and merges with "
        "careless. Drawn at one checkpoint: one careless scale is tens of "
        "GPU-minutes. Cross-validation repeats and Train/Test are "
        "categorical, not a ramp -- a repeat is not an ordering.",
        lambda s: s.startswith("careless_"),
    ),
    (
        "The learned profile basis",
        "Every component of the profile decoder, depth summed, with the bias "
        "beside them. A column is not a profile: the profile is the simplex "
        "projection of Wh + b, so the components are drawn on a diverging "
        "scale about zero and share one symmetric limit, and only the "
        "combination is a distribution. The history panel is each "
        "component's norm and its cosine against its own final value, which "
        "is what catches one component collapsing while the rest look "
        "settled. Recovered from the checkpoints: ProfileBasisLogger is "
        "gated behind integrator.train --figures and no run passed it.",
        lambda s: s.startswith("basis_"),
    ),
    (
        "The Wilson prior over training",
        "The learned scale G and B-factor of the prior every intensity is "
        "shrunk toward. Two panels because square angstroms and counts do "
        "not share an axis; the band on G is the spread across scale "
        "groups. G and B collapse together when G is parameterized in "
        "linear space, so either still moving at the last epoch is a prior "
        "that never identified.",
        lambda s: s.startswith("wilson_"),
    ),
    (
        "PDZ2: CCsym by resolution shell",
        "The half-dataset correlation of F(h) - F(2-fold h), one panel per "
        "pump delay, Spearman as the deposited script computes it. The "
        "field-off panel is the control and is drawn rather than "
        "subtracted: laue.integrate reads 0.10-0.20 there where the true "
        "answer is exactly zero, so much of any raw value is reproducible "
        "non-field asymmetry. The band is the full spread over the three "
        "cross-validation repeats.",
        lambda s: s.startswith("ccsym_"),
    ),
    (
        "Profile shape against the spots beside it",
        "Kept from the geometry investigation: whether a profile model "
        "follows the local spot shape, and what that looks like on one patch "
        "of one image. Delta-theta is the axial orientation difference from "
        "the local bright mean, with 45 degrees as the null.",
        lambda s: s.startswith(("profile_shape", "profile_width",
                                "profile_anisotropy", "detector_patch")),
    ),
    (
        "Arm comparison at one checkpoint",
        "A diagnostic, not a reported figure: arms side by side at a single "
        "checkpoint.",
        lambda s: s.startswith("arm_metrics"),
    ),
]

# the figures named in docs/figures/README.md that nothing here matched
EXPECTED = {
    "shoebox_profiles": "input shoeboxes against the predicted profiles",
    "loss_": "training loss",
    "scatter_": "model against the reference integrator",
    "merging_": "merging statistics by shell",
    "refinement": "refinement R-factors",
    "peakz_mean": "mean anomalous peak height",
    "spectrum": "the learned spectrum",
    "careless_": "CC-half and I/sigma from careless.stats",
    "basis_components": "the learned profile basis",
    "wilson_params": "the Wilson prior over training",
    "ccsym_": "PDZ2 CCsym by resolution shell",
}

_RESIDUES = ("IOD", "CYS", "MET", "SER", "HOH", "ILE", "HIS", "ASP", "GLU")


def _is_site(stem: str) -> bool:
    """`IOD_204`, `CYS_6` -- a per-site anomalous peak panel."""
    head, _, tail = stem.partition("_")
    return head in _RESIDUES and tail.isdigit()


def _site_order(path: Path):
    """Iodides first, then the rest alphabetically, then by residue number."""
    head, _, tail = path.stem.partition("_")
    return (head != "IOD", head, int(tail) if tail.isdigit() else 0)


def uri(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def card(p: Path, base: Path) -> str:
    rel = p.relative_to(base)
    return (
        f'<figure><img src="{uri(p)}" alt="{p.stem}" loading="lazy">'
        f"<figcaption>{rel}</figcaption></figure>"
    )


def collect(base: Path) -> list[Path]:
    """Every PNG under `base`, with duplicate basenames resolved.

    The same figure can exist in two directories when a render was pointed
    at the wrong `--out-dir`; the newest wins, so the page never shows one
    figure twice.
    """
    best: dict[str, Path] = {}
    for p in sorted(base.rglob("*.png")):
        prev = best.get(p.name)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            best[p.name] = p
    return sorted(best.values())


def main() -> int:
    # resolved, because the page prints the base relative to the repository
    # root and a relative argument raises there rather than here
    # the whole figure tree, not just `m9`: the default was one dataset's
    # directory, so the PDZ2, 1118 and cross-dataset comparison figures were
    # all absent from the page that is supposed to show everything rendered,
    # and the PDZ2 CCsym figure was reported as "not yet rendered" while it
    # existed one directory over
    base = (Path(sys.argv[1]) if len(sys.argv) > 1
            else ROOT / "docs" / "figures").resolve()
    out = (Path(sys.argv[2]) if len(sys.argv) > 2
           else ROOT / "docs" / "figures" / "gallery.html").resolve()
    if not base.is_dir():
        raise SystemExit(f"no figure directory at {base}")

    files = collect(base)
    claimed: set[Path] = set()
    sections = []
    counts = []
    for title, note, matches in RULES:
        group = [p for p in files if p not in claimed and matches(p.stem)]
        if not group:
            continue
        claimed.update(group)
        if any(_is_site(p.stem) for p in group):
            group.sort(key=_site_order)
        counts.append((title, len(group)))
        body = "\n".join(card(p, base) for p in group)
        sections.append(
            f'<section><h2>{title}</h2><p class="note">{note}</p>'
            f'<div class="grid">{body}</div></section>'
        )

    rest = [p for p in files if p not in claimed]
    if rest:
        counts.append(("Unsorted", len(rest)))
        body = "\n".join(card(p, base) for p in rest)
        sections.append(
            '<section><h2>Unsorted</h2><p class="note">Rendered but not in '
            "the reported figure list; either a diagnostic or a rule missing "
            'from build_gallery.py.</p>'
            f'<div class="grid">{body}</div></section>'
        )

    stems = {p.stem for p in files}
    missing = [
        what for key, what in EXPECTED.items()
        if not any(s.startswith(key) or s == key.rstrip("_") for s in stems)
    ]
    gap = ""
    if missing:
        items = "".join(f"<li>{m}</li>" for m in missing)
        gap = (
            '<section><h2>Not yet rendered</h2><p class="note">Named in '
            "docs/figures/README.md, with nothing on disk here.</p>"
            f"<ul class='gap'>{items}</ul></section>"
        )

    tally = ", ".join(f"{n} {t.lower()}" for t, n in counts)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Integrator Figures</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#f5f6f8;--surface:#fff;--ink:#161a20;--ink2:#5a6472;--line:#e3e7ee;--accent:#725590}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#0e1116;--surface:#171c24;--ink:#e9edf3;--ink2:#9aa4b4;--line:#28303c;--accent:#b98fd8}}}}
:root[data-theme=dark]{{--bg:#0e1116;--surface:#171c24;--ink:#e9edf3;--ink2:#9aa4b4;--line:#28303c;--accent:#b98fd8}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0}}
.wrap{{max-width:1180px;margin:0 auto;padding:40px 20px 80px}}
header{{border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:28px}}
h1{{font-size:28px;font-weight:700;letter-spacing:-.02em;margin:0 0 8px}}
.lede{{color:var(--ink2);margin:0;max-width:78ch}}
section{{margin-top:34px}}
h2{{font-size:16.5px;font-weight:650;margin:0 0 4px;padding-bottom:8px;border-bottom:2px solid var(--accent);display:inline-block}}
.note{{color:var(--ink2);font-size:13.5px;margin:8px 0 15px;max-width:82ch}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}}
figure{{margin:0;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden}}
figure img{{display:block;width:100%;height:auto;background:#fff}}
figcaption{{font:12px/1.3 ui-monospace,Menlo,monospace;color:var(--ink2);padding:7px 9px;border-top:1px solid var(--line);word-break:break-word}}
ul.gap{{color:var(--ink2);font-size:13.5px}}
</style></head><body>
<div class="wrap">
<header><h1>Integrator figures</h1>
<p class="lede">Everything currently rendered under <code>{base.relative_to(ROOT)}</code>,
in the order of the reported figure list. {len(files)} figures: {tally}.
Drawn with figTeX and the witch scheme, typeset by LuaLaTeX in Minion&nbsp;Pro
with mtpro2 maths.</p></header>
{''.join(sections)}
{gap}
</div></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB, {len(files)} figures)")
    for title, n in counts:
        print(f"  {n:>3}  {title}")
    if missing:
        print("  not yet rendered: " + "; ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
