"""Forward-evaluate the ported physical scaling model against DIALS.

Loads DIALS' own refined parameters from a scaled `.expt`, evaluates the
port on the same reflections, and compares with the
`inverse_scale_factor` column DIALS wrote. No refinement is involved, so
a match proves the model itself is right independently of any optimizer.

Run from the repository root with a DIALS-capable interpreter::

    python scripts/verify_dials_port_scale_model.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from integrator.dials_port.harmonics import (  # noqa: E402
    SQRT2_DIALS,
    harmonic_table,
)
from integrator.dials_port.io import (  # noqa: E402
    load_experiment,
    load_npz,
    load_reflections,
)
from integrator.dials_port.scale_model import (  # noqa: E402
    PhysicalScalingModel,
    build_features,
)

DEFAULT_EXPT = "/Users/luis/dials_out/816_sbgrid_HEWL/pass1/scaled.expt"
DEFAULT_CACHE = (
    "/private/tmp/claude-501/-Users-luis-integrator/"
    "a69c47c1-a7b1-409e-b127-f918bd14acbc/scratchpad/hewl_scaled.npz"
)


def _report(tag: str, model: torch.Tensor, ref: torch.Tensor) -> float:
    """Print and return the max relative difference of `model` vs `ref`."""
    rel = (torch.abs(model - ref) / torch.abs(ref)).detach().numpy()
    print(
        f"  {tag:<34s} max={rel.max():.3e}  median={np.median(rel):.3e}  "
        f"mean={rel.mean():.3e}"
    )
    return float(rel.max())


def main() -> int:
    """Run the comparison and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expt", default=DEFAULT_EXPT)
    parser.add_argument("--refl", default=None, help="a .refl, needs DIALS")
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="a .npz")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    expt = load_experiment(args.expt)
    refl = load_reflections(args.refl) if args.refl else load_npz(args.cache)
    config = expt.scaling_model.config
    lmax = int(config["lmax"])
    dials_scale = refl.scale

    print(f"{len(refl)} reflections, lmax={lmax}")
    print(
        "  s_norm_fac={s_norm_fac!r} d_norm_fac={d_norm_fac!r}".format(
            **config
        )
    )

    model = PhysicalScalingModel.from_dials(expt)

    def bisect(tag: str, feats) -> float:
        """Report each cumulative product against DIALS' column."""
        print(f"\n{tag}")
        scale = model.scale_component(feats)
        decay = model.decay_component(feats)
        absorption = model.absorption_component(feats)
        _report("scale", scale, dials_scale)
        _report("scale * decay", scale * decay, dials_scale)
        return _report(
            "scale * decay * absorption",
            scale * decay * absorption,
            dials_scale,
        )

    exact = build_features(refl, expt, lmax=lmax)
    bisect("exact spherical harmonic evaluation", exact)

    grid = build_features(refl, expt, lmax=lmax, points_per_degree=2)
    worst = bisect(
        "DIALS lookup-grid evaluation (points_per_degree=2, the path taken "
        "for >100k reflections)",
        grid,
    )

    truncated = harmonic_table(
        refl, expt, lmax=lmax, points_per_degree=2, sqrt2=SQRT2_DIALS
    )
    delta = torch.abs(truncated - grid.harmonics)
    denominator = torch.abs(grid.harmonics).clamp_min(1e-12)
    print(
        f"\nsqrt(2) truncation shifts the basis by at most "
        f"{float((delta / denominator).max()):.2e} relative"
    )
    with_truncation = build_features(
        refl, expt, lmax=lmax, points_per_degree=2, sqrt2=SQRT2_DIALS
    )
    _report("combined, truncated sqrt(2)", model(with_truncation), dials_scale)

    print(
        f"\nmax relative difference {worst:.3e} (tolerance {args.tolerance})"
    )
    if worst < args.tolerance:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
