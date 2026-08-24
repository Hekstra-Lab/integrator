"""Patch a base ablation config into one arm of the dispersion screen.

Reads a base YAML (e.g. `configs/ablation_likelihood/hierarchical_nbinom.yaml`),
overrides only the `loss.args` pixel-likelihood knobs, and writes the result. The
screen holds the dispersion `r` FIXED (`nb_learn_dispersion: false`) and varies
`nb_dispersion_init` across arms, so each run measures the ELBO at one point on the
Poisson (`r -> inf`) to heavily-overdispersed axis.

Usage:
    python make_screen_config.py --base BASE --out OUT --likelihood poisson
    python make_screen_config.py --base BASE --out OUT \
        --likelihood negative_binomial --dispersion 10 --scope global
    python make_screen_config.py --base BASE --out OUT \
        --likelihood negative_binomial --dispersion 10 --learn-dispersion
"""

import argparse

import yaml

_NB_KEYS = (
    "nb_dispersion_init",
    "nb_dispersion_scope",
    "nb_dispersion_floor",
    "nb_learn_dispersion",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base config YAML to patch.")
    parser.add_argument("--out", required=True, help="Where to write the patched config.")
    parser.add_argument(
        "--likelihood",
        required=True,
        choices=["poisson", "negative_binomial"],
    )
    parser.add_argument(
        "--dispersion",
        type=float,
        default=None,
        help="Fixed NB dispersion r (nb_dispersion_init). Required for NB.",
    )
    parser.add_argument(
        "--scope",
        default="global",
        choices=["global", "per_bin"],
        help="NB dispersion scope.",
    )
    parser.add_argument(
        "--learn-dispersion",
        action="store_true",
        help="Learn r instead of holding it fixed (reference arm only).",
    )
    args = parser.parse_args()

    with open(args.base) as f:
        cfg = yaml.safe_load(f)

    loss_args = cfg.setdefault("loss", {}).setdefault("args", {})
    loss_args["likelihood"] = args.likelihood

    if args.likelihood == "poisson":
        # Poisson takes no dispersion; drop the NB keys so the config reads clean.
        for k in _NB_KEYS:
            loss_args.pop(k, None)
    else:
        if args.dispersion is None:
            parser.error("--dispersion is required for negative_binomial.")
        loss_args["nb_dispersion_init"] = args.dispersion
        loss_args["nb_dispersion_scope"] = args.scope
        loss_args["nb_learn_dispersion"] = args.learn_dispersion

    with open(args.out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


if __name__ == "__main__":
    main()
