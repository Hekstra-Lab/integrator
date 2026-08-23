"""The careless recipes used for the Laue data, as flag lists.

These reproduce `hewl_1118/laue-dials-careless/config{1..6}.sh` and the
unified `refltorch/scripts/laue_output/careless_scale.sh`, so a run started
here is comparable with everything already scaled on the cluster.

Configs 2 and 3 split the data on Friedel mates before scaling and merge the
two outputs afterwards; that path is not implemented in `run_pipeline.py`
(use `careless_scale.sh` plus `submit_refinement.py` for those).
"""

from __future__ import annotations

METADATA_KEYS = "BATCH,xcal,ycal,dHKL,wavelength"

DESCRIPTIONS = {
    1: "anomalous, baseline",
    2: "Friedel split, separate files, double Wilson",
    3: "Friedel split, separate files, double Wilson, positional encoding",
    4: "anomalous, positional encoding",
    5: "anomalous, image-layers=2, mlp-width=24, 6k iterations",
    6: "anomalous, image-layers=2, mlp-width=24, positional encoding",
}

FRIEDEL_SPLIT_CONFIGS = frozenset({2, 3})

# shared by every config; dmin is filled in per run
COMMON = (
    "--merge-half-datasets",
    "--half-dataset-repeats=3",
    "--mc-samples=10",
    "--mlp-layers=10",
    "--studentt-likelihood-dof=64",
    '--wavelength-key=wavelength',
)

POSITIONAL_ENCODING = (
    "--positional-encoding-frequencies=4",
    '--positional-encoding-keys=xcal,ycal,BATCH',
)

_PER_CONFIG = {
    1: ("--anomalous", "--mlp-width=32", "--image-layers=0",
        "--iterations=30000"),
    4: ("--anomalous", "--mlp-width=32", "--image-layers=0",
        *POSITIONAL_ENCODING, "--test-fraction=0.1", "--iterations=30000"),
    5: ("--anomalous", "--mlp-width=24", "--image-layers=2",
        "--test-fraction=0.1", "--iterations=6000"),
    6: ("--anomalous", "--mlp-width=24", "--image-layers=2",
        *POSITIONAL_ENCODING, "--test-fraction=0.1", "--iterations=30000"),
}


def flags(config: int, dmin: float = 1.5, seed: int | None = None) -> list[str]:
    """Flag list for one careless config, excluding keys, input, and output."""
    if config in FRIEDEL_SPLIT_CONFIGS:
        raise NotImplementedError(
            f"config {config} splits on Friedel mates before scaling; run it "
            "with refltorch/scripts/laue_output/careless_scale.sh"
        )
    if config not in _PER_CONFIG:
        raise ValueError(f"unknown careless config {config!r}")
    out = [*COMMON, f"--dmin={dmin}", *_PER_CONFIG[config]]
    if seed is not None:
        out.append(f"--seed={seed}")
    return out
