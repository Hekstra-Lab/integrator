"""Compare `dials_port.stats` against cctbx and a `dials.scale` log.

Runs the torch merging statistics and
`iotbx.merging_statistics.dataset_statistics` over the same reflections,
in the same process, and prints the two side by side with the values
DIALS wrote to its log. Requires an environment carrying both cctbx and
torch (`integrator-dev`).

Usage:
    python scripts/verify_dials_port_stats.py \
        --npz /tmp/hewl_scaled.npz \
        --expt /path/to/scaled.expt \
        --log /path/to/dials.scale.log
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from integrator.dials_port import stats
from integrator.dials_port.io import Experiment, load_experiment, load_npz
from integrator.dials_port.symmetry import build_asu_map
from integrator.dials_port.types import Reflections

# Statistics cctbx computes without random numbers; these are expected to
# agree to round-off. The stochastic ones are handled separately.
_DETERMINISTIC = [
    ("n_obs", "n_obs"),
    ("n_uniq", "n_uniq"),
    ("multiplicity", "mean_redundancy"),
    ("completeness", "completeness"),
    ("i_mean", "i_mean"),
    ("sigi_mean", "sigi_mean"),
    ("i_over_sigma_mean", "i_over_sigma_mean"),
    ("i_mean_over_sigi_mean", "i_mean_over_sigi_mean"),
    ("unmerged_i_over_sigma_mean", "unmerged_i_over_sigma_mean"),
    ("r_merge", "r_merge"),
    ("r_meas", "r_meas"),
    ("r_pim", "r_pim"),
    ("cc_half_sigma_tau", "cc_one_half_sigma_tau"),
    ("cc_half_sigma_tau_n_refl", "cc_one_half_sigma_tau_n_refl"),
]

_STOCHASTIC = [
    ("cc_half", "cc_one_half"),
    ("cc_anom", "cc_anom"),
]

# Column order of the "Merging statistics by resolution bin" table.
_LOG_COLUMNS = [
    "d_max",
    "d_min",
    "n_obs",
    "n_uniq",
    "multiplicity",
    "completeness",
    "i_mean",
    "i_over_sigma_mean",
    "r_merge",
    "r_meas",
    "r_pim",
    "r_anom",
    "cc_half",
    "cc_anom",
]


@dataclass
class Inputs:
    """Everything the comparison needs."""

    refl: Reflections
    amap: object
    expt: Experiment
    n_bins: int


def load_inputs(npz: Path, expt_path: Path, n_bins: int) -> Inputs:
    """Load reflections and build the ASU map.

    Args:
        npz: cached `Reflections` from `io.dump_npz`.
        expt_path: matching `.expt`.
        n_bins: number of resolution shells.

    Returns:
        The loaded inputs.
    """
    refl = load_npz(npz)
    expt = load_experiment(expt_path)
    amap = build_asu_map(refl.hkl, expt.space_group_hall)
    return Inputs(refl=refl, amap=amap, expt=expt, n_bins=n_bins)


def cctbx_reference(inputs: Inputs) -> object:
    """Run `iotbx.merging_statistics` with the settings dials.scale uses.

    `merging_stats_from_scaled_array` passes `sigma_filtering=None`,
    `eliminate_sys_absent=False` and `use_internal_variance=False`, over
    an array of `intensity.scale.value / inverse_scale_factor`.

    Args:
        inputs: loaded reflections and experiment.

    Returns:
        The `dataset_statistics` instance.
    """
    import iotbx.merging_statistics
    from cctbx import crystal, miller, sgtbx
    from cctbx.array_family import flex

    refl, expt = inputs.refl, inputs.expt
    intensity = (refl.intensity / refl.scale).numpy()
    sigma = (refl.sigma / refl.scale).numpy()

    symmetry = crystal.symmetry(
        unit_cell=tuple(expt.cell),
        space_group=sgtbx.space_group(expt.space_group_hall),
        assert_is_compatible_unit_cell=False,
    )
    miller_set = miller.set(
        crystal_symmetry=symmetry,
        indices=flex.miller_index(refl.hkl.numpy().astype(np.int32)),
        anomalous_flag=False,
    )
    array = miller.array(
        miller_set, data=flex.double(intensity), sigmas=flex.double(sigma)
    )
    array.set_observation_type_xray_intensity()

    return iotbx.merging_statistics.dataset_statistics(
        i_obs=array,
        n_bins=inputs.n_bins,
        anomalous=False,
        sigma_filtering=None,
        eliminate_sys_absent=False,
        use_internal_variance=False,
        cc_one_half_significance_level=0.01,
    )


def parse_log(path: Path) -> tuple[list[dict], dict] | None:
    """Extract the per-shell table from a `dials.scale` log.

    Args:
        path: path to `dials.scale.log`.

    Returns:
        `(shells, overall)` as dictionaries keyed by statistic name, or
        `None` if the table is not present.
    """
    text = path.read_text()
    marker = "Merging statistics by resolution bin"
    if marker not in text:
        return None
    body = text.split(marker, 1)[1]
    rows: list[list[float]] = []
    for line in body.splitlines():
        fields = line.replace("*", "").split()
        if len(fields) != len(_LOG_COLUMNS):
            if rows:
                break
            continue
        try:
            rows.append([float(f) for f in fields])
        except ValueError:
            continue
    if not rows:
        return None
    parsed = [dict(zip(_LOG_COLUMNS, r, strict=True)) for r in rows]
    for row in parsed:
        row["completeness"] /= 100.0
    return parsed[:-1], parsed[-1]


def _rel(a: float, b: float) -> float:
    """Relative difference, falling back to absolute near zero."""
    if a is None or b is None:
        return float("nan")
    scale = max(abs(a), abs(b))
    if scale == 0:
        return 0.0
    if scale < 1e-12:
        return abs(a - b)
    return abs(a - b) / scale


def compare_overall(
    ours: stats.ShellStats, ref: object, log: dict | None
) -> None:
    """Print the overall comparison table."""
    print("\n=== OVERALL ===")
    print(
        f"{'statistic':30s} {'torch':>16s} {'cctbx':>16s} "
        f"{'rel.diff':>10s} {'dials.log':>10s}"
    )
    for mine, theirs in _DETERMINISTIC:
        a = getattr(ours, mine)
        b = getattr(ref, theirs)
        logged = log.get(mine) if log else None
        log_s = f"{logged:10.4f}" if logged is not None else " " * 10
        print(
            f"{mine:30s} {float(a):16.10g} {float(b):16.10g} "
            f"{_rel(float(a), float(b)):10.2e} {log_s}"
        )
    print("  -- stochastic (random half-dataset split) --")
    for mine, theirs in _STOCHASTIC:
        a = getattr(ours, mine)
        b = getattr(ref, theirs)
        spread = getattr(ours, f"{mine}_std")
        logged = log.get(mine) if log else None
        log_s = f"{logged:10.4f}" if logged is not None else " " * 10
        print(
            f"{mine:30s} {a:16.10g} {float(b):16.10g} "
            f"{_rel(a, float(b)):10.2e} {log_s}   split sd={spread:.2e}"
        )
    print("  -- no cctbx equivalent --")
    for name in ("cc_star", "r_split", "weighted_cc_half"):
        print(f"{name:30s} {getattr(ours, name):16.10g}")
    print(f"{'weighted_cc_half_neff':30s} {ours.weighted_cc_half_neff:16.10g}")


def compare_shells(
    ours: list[stats.ShellStats], ref: object, log: list[dict] | None
) -> None:
    """Print the per-shell comparison, one block per statistic."""
    print("\n=== PER SHELL ===")
    all_pairs = _DETERMINISTIC + _STOCHASTIC
    for mine, theirs in all_pairs:
        worst = 0.0
        lines = []
        for i, (a_shell, b_shell) in enumerate(zip(ours, ref, strict=True)):
            a = float(getattr(a_shell, mine))
            b = float(getattr(b_shell, theirs))
            logged = log[i].get(mine) if log else None
            rel = _rel(a, b)
            worst = max(worst, rel)
            log_s = f"{logged:9.4f}" if logged is not None else " " * 9
            lines.append(
                f"  {a_shell.d_max:7.3f} {a_shell.d_min:7.3f} "
                f"{a:15.8g} {b:15.8g} {rel:9.2e} {log_s}"
            )
        print(f"\n-- {mine} (worst rel.diff {worst:.2e}) --")
        print(
            f"  {'d_max':>7s} {'d_min':>7s} {'torch':>15s} "
            f"{'cctbx':>15s} {'rel':>9s} {'log':>9s}"
        )
        print("\n".join(lines))


def cctbx_split_noise(inputs: Inputs, n_seeds: int) -> dict[str, np.ndarray]:
    """Re-seed cctbx's own splitter to get its sampling distribution.

    `dataset_statistics` reports a single draw taken with the default
    Mersenne twister state, which says nothing about how much that draw
    moves. Driving `split_unmerged` directly with explicit seeds gives
    the distribution the torch implementation has to match.

    Args:
        inputs: loaded reflections and experiment.
        n_seeds: number of seeds to draw.

    Returns:
        Per-statistic arrays of `n_seeds` values.
    """
    import math

    from cctbx import crystal, miller, sgtbx
    from cctbx.array_family import flex
    from cctbx.miller import split_unmerged

    refl, expt = inputs.refl, inputs.expt
    symmetry = crystal.symmetry(
        unit_cell=tuple(expt.cell),
        space_group=sgtbx.space_group(expt.space_group_hall),
        assert_is_compatible_unit_cell=False,
    )
    array = miller.array(
        miller.set(
            symmetry,
            flex.miller_index(refl.hkl.numpy().astype(np.int32)),
            anomalous_flag=False,
        ),
        data=flex.double((refl.intensity / refl.scale).numpy()),
        sigmas=flex.double((refl.sigma / refl.scale).numpy()),
    )
    array.set_observation_type_xray_intensity()

    laue = array.map_to_asu().sort("packed_indices")
    laue = laue.select(laue.sigmas() > 0)
    anom = (
        array.customized_copy(anomalous_flag=True)
        .map_to_asu()
        .sort("packed_indices")
    )

    out: dict[str, list[float]] = {
        "cc_half": [],
        "cc_anom": [],
        "r_split": [],
    }
    # seed 0 leaves the generator in its default state, so start at 1.
    for seed in range(1, n_seeds + 1):
        split = split_unmerged(
            unmerged_indices=laue.indices(),
            unmerged_data=laue.data(),
            unmerged_sigmas=laue.sigmas(),
            seed=seed,
        )
        out["cc_half"].append(
            flex.linear_correlation(split.data_1, split.data_2).coefficient()
        )
        one = np.asarray(split.data_1)
        two = np.asarray(split.data_2)
        out["r_split"].append(
            np.abs(one - two).sum()
            / (math.sqrt(2.0) * 0.5 * (one + two).sum())
        )

        # CC(anom) splits the anomalous grouping with unweighted means.
        split_a = split_unmerged(
            unmerged_indices=anom.indices(),
            unmerged_data=anom.data(),
            unmerged_sigmas=anom.sigmas(),
            weighted=False,
            seed=seed,
        )
        base = miller.set(
            crystal_symmetry=array,
            indices=split_a.indices,
            anomalous_flag=True,
        )
        dano_1 = miller.array(
            base, data=split_a.data_1
        ).anomalous_differences()
        dano_2 = miller.array(
            base, data=split_a.data_2
        ).anomalous_differences()
        out["cc_anom"].append(
            dano_1.correlation(other=dano_2, use_binning=False).coefficient()
        )
    return {k: np.asarray(v) for k, v in out.items()}


def monte_carlo_noise(inputs: Inputs, n_seeds: int, n_repeats: int) -> None:
    """Compare the split-noise distributions of both implementations.

    The random-split statistics cannot agree draw for draw, because
    cctbx samples from a scitbx Mersenne twister. What they can agree on
    is the distribution, which is what this prints.

    Args:
        inputs: loaded reflections and experiment.
        n_seeds: number of independent seeds to draw.
        n_repeats: splits averaged within each seed.
    """
    print(
        f"\n=== MONTE-CARLO NOISE ({n_seeds} seeds x {n_repeats} repeats) ==="
    )
    refl, amap = inputs.refl, inputs.amap
    mine: dict[str, list[float]] = {
        "cc_half": [],
        "cc_anom": [],
        "r_split": [],
    }
    for seed in range(n_seeds):
        mine["cc_half"].append(
            stats.cc_half_random(
                refl, amap, seed=seed, n_repeats=n_repeats
            ).value
        )
        mine["cc_anom"].append(
            stats.cc_anom(refl, amap, seed=seed, n_repeats=n_repeats).value
        )
        mine["r_split"].append(
            stats.r_split(refl, amap, seed=seed, n_repeats=n_repeats).value
        )
    theirs = cctbx_split_noise(inputs, n_seeds)

    print(
        f"{'statistic':12s} {'source':>7s} {'mean':>11s} {'sd':>10s} "
        f"{'min':>11s} {'max':>11s}"
    )
    for name, values in mine.items():
        for source, v in (
            ("torch", np.asarray(values)),
            ("cctbx", theirs[name]),
        ):
            print(
                f"{name:12s} {source:>7s} {v.mean():11.6f} "
                f"{v.std(ddof=1):10.2e} {v.min():11.6f} {v.max():11.6f}"
            )


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: command-line arguments, defaulting to `sys.argv[1:]`.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--expt", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--n-seeds", type=int, default=20)
    args = parser.parse_args(argv)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    inputs = load_inputs(args.npz, args.expt, args.n_bins)
    print(f"{len(inputs.refl)} observations, cell {inputs.expt.cell}")

    ours = stats.merging_statistics(
        inputs.refl,
        inputs.amap,
        inputs.expt,
        n_bins=args.n_bins,
        seed=args.seed,
        n_repeats=args.n_repeats,
    )
    print("\n=== torch table ===")
    print(ours)

    ref = cctbx_reference(inputs)
    log = parse_log(args.log) if args.log else None
    log_shells, log_overall = log if log else (None, None)

    compare_overall(ours.overall, ref.overall, log_overall)
    compare_shells(ours.shells, ref.bins, log_shells)
    monte_carlo_noise(inputs, args.n_seeds, args.n_repeats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
