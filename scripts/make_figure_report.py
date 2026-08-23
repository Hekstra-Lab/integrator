"""Assemble a figures directory into one self-contained HTML page.

Images are inlined as data URIs, so the page can be opened locally, sent
to a collaborator, or published as an Artifact without carrying a folder
of assets alongside it.

Usage:
    uv run python scripts/make_figure_report.py --fig-dir <run>/figures
    uv run python scripts/make_figure_report.py --fig-dir <dir> \
        --title "HEWL hierarchical run" --out report.html
"""

from __future__ import annotations

import argparse
import base64
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MIME = {".png": "image/png", ".gif": "image/gif", ".svg": "image/svg+xml"}

SECTIONS: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    (
        "Predictions over training",
        "A fixed set of shoeboxes, chosen once to span the weak, medium, "
        "and strong I/σ regimes, pushed through every epoch.",
        (
            (
                "anim_tracked",
                "Observed counts, fitted rate, profile, and residual for "
                "each tracked shoebox, animated over epochs.",
            ),
            (
                "fig_tracked_filmstrip_rate",
                "Fitted rate I·p + B against the observed counts. The "
                "inset number is the posterior mean intensity.",
            ),
            (
                "fig_tracked_filmstrip_profile",
                "Posterior mean profile for the same shoeboxes.",
            ),
            (
                "fig_tracked_filmstrip_residual",
                "counts − rate. Structure that survives here is signal the "
                "model is not capturing.",
            ),
            (
                "fig_tracked_trajectories",
                "Posterior intensity ± 1 sd per epoch, against the DIALS "
                "value (dashed).",
            ),
            (
                "fig_tracked_uncertainty",
                "Relative posterior width and the rms Pearson residual, "
                "by regime.",
            ),
        ),
    ),
    (
        "Learned profile basis",
        "The profile surrogate decodes p = softmax(W h + b); W's columns "
        "are the profile modes.",
        (
            ("anim_basis", "The basis atlas over training."),
            (
                "fig_basis_atlas_weight",
                "Final basis: the mean profile softmax(b) and each mode as "
                "a signed image.",
            ),
            (
                "fig_basis_atlas_effect",
                "The same modes as their effect on a profile, "
                "softmax(b + 3·W_i) − softmax(b).",
            ),
            (
                "fig_basis_filmstrip",
                "Each mode across epochs, ordered and sign-aligned to the "
                "final basis.",
            ),
            (
                "fig_basis_convergence",
                "Decoder spectrum, effective rank, and the step size and "
                "principal angle to the final basis.",
            ),
        ),
    ),
    (
        "Latent space",
        "What the per-reflection profile latent encodes.",
        (
            (
                "fig_latent_pca",
                "PC1/PC2 of the latent, colored by physical covariates.",
            ),
            (
                "fig_latent_clusters",
                "k-means clusters with each centroid decoded back into a "
                "profile.",
            ),
            (
                "fig_latent_covariate_r2",
                "R² of each covariate against each PC and against the full "
                "latent.",
            ),
            (
                "fig_latent_detector_map",
                "The same reflections in detector coordinates.",
            ),
            (
                "fig_latent_usage",
                "Spread of each latent dimension against its posterior and "
                "prior scale; flat dimensions are unused capacity.",
            ),
        ),
    ),
    (
        "Model checks",
        "One full pass over the validation split with the final weights.",
        (
            (
                "fig_check_ppc",
                "Pixel-level Pearson residuals against N(0, 1), and their "
                "spread as a function of the fitted rate.",
            ),
            (
                "fig_check_model_vs_dials",
                "Model posterior intensity against DIALS, and the median "
                "ratio through the weak tail.",
            ),
        ),
    ),
    (
        "For a general audience",
        "The same results, told without the machinery.",
        (
            (
                "fig_explain_decomposition",
                "One spot as counts ≈ intensity × shape + background.",
            ),
            (
                "fig_explain_posterior_gallery",
                "Three spots with the posterior over their intensity drawn "
                "underneath.",
            ),
            (
                "fig_explain_shrinkage",
                "Model intensity against the conventional estimate, with "
                "error bars, by regime.",
            ),
        ),
    ),
)

STYLE = """
:root {
  --bg: #ffffff;
  --fg: #1a1a1a;
  --muted: #5c5c5c;
  --rule: #e2e2e2;
  --card: #fafafa;
  --accent: #0173B2;
}
:root:not([data-theme="light"]) {
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #14151a;
    --fg: #e8e8ea;
    --muted: #a0a3ab;
    --rule: #2b2d35;
    --card: #1c1e25;
    --accent: #6cb6e8;
  }
}
:root[data-theme="dark"] {
  --bg: #14151a;
  --fg: #e8e8ea;
  --muted: #a0a3ab;
  --rule: #2b2d35;
  --card: #1c1e25;
  --accent: #6cb6e8;
}
body {
  background: var(--bg);
  color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica,
    Arial, sans-serif;
  line-height: 1.55;
  margin: 0 auto;
  max-width: 60rem;
  padding: 2.5rem 1.25rem 5rem;
}
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
h2 {
  font-size: 1.15rem;
  margin: 2.5rem 0 .25rem;
  padding-top: 1.25rem;
  border-top: 1px solid var(--rule);
}
p.lede, p.section-note { color: var(--muted); margin: .25rem 0 1rem; }
figure {
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: .5rem;
  margin: 0 0 1.5rem;
  padding: .75rem;
}
figure img { display: block; width: 100%; height: auto; }
figcaption {
  color: var(--muted);
  font-size: .85rem;
  margin-top: .5rem;
}
figcaption b { color: var(--fg); font-weight: 600; }
code {
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: .25rem;
  font-size: .85em;
  padding: .1rem .3rem;
}
.missing { color: var(--muted); font-size: .85rem; }
"""


def parse_args():
    p = argparse.ArgumentParser(
        description="Bundle a figures directory into one HTML page"
    )
    p.add_argument("--fig-dir", type=str, required=True)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--title", type=str, default=None)
    p.add_argument(
        "--max-mb",
        type=float,
        default=6.0,
        help="Skip any single asset larger than this",
    )
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def data_uri(path: Path) -> str:
    """Inline a file as a base64 data URI."""
    mime = MIME.get(path.suffix, "application/octet-stream")
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def find_asset(fig_dir: Path, stem: str) -> Path | None:
    """Prefer the animation, then the raster, for a figure stem."""
    for ext in (".gif", ".png", ".svg"):
        candidate = fig_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def build_html(fig_dir: Path, title: str, max_mb: float) -> str:
    """Render the whole page as one HTML string."""
    parts = [
        f"<title>{title}</title>",
        f"<style>{STYLE}</style>",
        f"<h1>{title}</h1>",
        '<p class="lede">Generated by '
        "<code>scripts/make_figure_report.py</code> from "
        f"<code>{fig_dir}</code>.</p>",
    ]
    limit = max_mb * 1024 * 1024
    for heading, note, entries in SECTIONS:
        present = [
            (stem, caption, find_asset(fig_dir, stem))
            for stem, caption in entries
        ]
        present = [item for item in present if item[2] is not None]
        if not present:
            continue
        parts.append(f"<h2>{heading}</h2>")
        parts.append(f'<p class="section-note">{note}</p>')
        for stem, caption, path in present:
            if path.stat().st_size > limit:
                parts.append(
                    f'<p class="missing">{stem}: {path.name} is '
                    f"{path.stat().st_size / 1e6:.1f} MB; skipped.</p>"
                )
                continue
            parts.append(
                "<figure>"
                f'<img alt="{stem}" src="{data_uri(path)}">'
                f"<figcaption><b>{stem}</b> — {caption}</figcaption>"
                "</figure>"
            )
    return "\n".join(parts)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s | %(message)s",
    )
    fig_dir = Path(args.fig_dir)
    title = args.title or f"Training figures — {fig_dir.parent.name}"
    out = Path(args.out) if args.out else fig_dir / "report.html"
    out.write_text(build_html(fig_dir, title, args.max_mb), encoding="utf-8")
    size = out.stat().st_size / 1e6
    print(f"wrote {out} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
