"""Generate a set of stop-criterion variants from a base YAML.

Takes the same base config and produces N delta variants differing only
in the `early_stop` and `checkpoint` sections. Run each variant, then
compare the final single checkpoint each one selects.

Usage:
    uv run python scripts/generate_stop_criteria_configs.py \\
        --base configs/wilson_comparison/hierC_learned_reg_warm_frozen_b.yaml \\
        --out-dir configs/stop_criteria/
"""

import argparse
import copy
from pathlib import Path

import yaml

# Hold patience + min_delta fixed across all variants so the comparison
# only varies the monitor (what you're optimizing for). Lightning's
# EarlyStopping fires on val_epoch_end by default for every metric, so
# patience is in val-check units — with check_val_every_n_epoch=5,
# patience=3 means "no improvement in 15 train epochs".
#
# min_delta=0.5 nats is large enough to ignore fractional-nat drift at
# convergence and small enough to catch real learning early.
PATIENCE = 3
MIN_DELTA = 0.5

# (name, monitor, mode)
VARIANTS = [
    ("val_nll", "val nll", "min"),
    ("train_nll", "train nll", "min"),
    ("train_elbo", "train elbo", "min"),
    ("val_elbo", "val elbo", "min"),
    ("gap_elbo", "gap elbo", "max"),  # gap is negative → max closes toward 0
    ("gap_nll", "gap nll", "max"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(args.base.read_text())

    for name, monitor, mode in VARIANTS:
        cfg = copy.deepcopy(base)
        cfg["early_stop"] = {
            "monitor": monitor,
            "mode": mode,
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
        }
        cfg["checkpoint"] = {"save_top_k": 1}
        # 80-epoch safety cap (the signal plateaus by ~40; anything
        # past 80 is just optimizer drift).
        cfg["trainer"]["max_epochs"] = 80

        out_path = args.out_dir / f"{args.base.stem}__{name}.yaml"
        out_path.write_text(
            yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
        )
        print(f"wrote {out_path}")

    # Baseline — no early stop, save every epoch as today
    baseline = copy.deepcopy(base)
    baseline["trainer"]["max_epochs"] = 100
    base_path = args.out_dir / f"{args.base.stem}__no_stop.yaml"
    base_path.write_text(
        yaml.safe_dump(baseline, sort_keys=False, default_flow_style=False)
    )
    print(f"wrote {base_path} (baseline, no early stop)")


if __name__ == "__main__":
    main()
