#!/bin/bash
# make_figures.sh --- run the config-driven multi-model comparison figures.
#
# One plot_cfg.yaml (a `runs:` map of {path, label} plus an optional
# `reference_data:` block; see scripts/plot_cfg.example.yaml) drives every
# multi-model plot script. This wires them into one command that writes the
# whole comparison (the `architecture_comparison/`-style figures) to --out-dir.
#
# Usage:
#   scripts/make_figures.sh --plot-cfg plot_cfg.yaml [--out-dir figures] \
#       [--epoch N] [--with-single]
#
#   --plot-cfg   FILE  model manifest (required); schema in plot_cfg.example.yaml
#   --out-dir    DIR   where figures go (default: <repo>/figures)
#   --epoch      N     compare/merging at one epoch (default: latest / all)
#   --with-single      also run the per-model plots (profiles, learned basis),
#                      one subdir per label under --out-dir
#
# Env:
#   RUNNER   how to launch python (default "uv run python"; on the cluster set
#            RUNNER=python inside the micromamba env)
#
# A missing optional input (no merged.html, no peaks.csv, ...) makes only that
# one step warn and is skipped; the rest still run.

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${INTEGRATOR_ROOT:-$(dirname "$SCRIPT_DIR")}"   # repo root holds the plot_*.py
RUNNER="${RUNNER:-uv run python}"

PLOT_CFG=""
OUT_DIR="$REPO/figures"
EPOCH=""
WITH_SINGLE=0

usage() { sed -n '2,20p' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--plot-cfg)  PLOT_CFG="$2"; shift 2;;
        -o|--out-dir)   OUT_DIR="$2";  shift 2;;
        -e|--epoch)     EPOCH="$2";    shift 2;;
        --with-single)  WITH_SINGLE=1; shift;;
        -h|--help)      usage; exit 0;;
        *) echo "ERROR: unknown argument '$1'" >&2; usage; exit 1;;
    esac
done

[[ -n "$PLOT_CFG" ]] || { echo "ERROR: --plot-cfg is required." >&2; usage; exit 1; }
[[ -f "$PLOT_CFG" ]] || { echo "ERROR: plot-cfg not found: $PLOT_CFG" >&2; exit 1; }
PLOT_CFG="$(cd "$(dirname "$PLOT_CFG")" && pwd)/$(basename "$PLOT_CFG")"   # absolutize
mkdir -p "$OUT_DIR"
cd "$REPO"   # so `python plot_compare.py` resolves and `uv run` finds the project

# epoch flag only applies to compare + merging
epoch_flag=""
[[ -n "$EPOCH" ]] && epoch_flag="--epoch $EPOCH"

echo "repo:     $REPO"
echo "plot-cfg: $PLOT_CFG"
echo "out-dir:  $OUT_DIR"
echo "runner:   $RUNNER"
echo

run_step() {
    local name="$1"; shift
    echo "── $name ──"
    if "$@"; then
        echo "   ok: $name"
    else
        echo "   WARN: $name failed (exit $?) — continuing" >&2
    fi
    echo
}

# ── multi-model, config-driven (reproduce the architecture_comparison figures) ─
run_step "loss"    $RUNNER plot_loss.py    --plot-cfg "$PLOT_CFG" --out-dir "$OUT_DIR"
run_step "compare" $RUNNER plot_compare.py --plot-cfg "$PLOT_CFG" --out-dir "$OUT_DIR" $epoch_flag
run_step "merging" $RUNNER plot_merging.py --plot-cfg "$PLOT_CFG" --out-dir "$OUT_DIR" $epoch_flag
run_step "peaks"   $RUNNER plot_peaks.py   --plot-cfg "$PLOT_CFG" --out-dir "$OUT_DIR"

# ── per-model plots (profiles, learned basis): one subdir per label ────────────
if [[ "$WITH_SINGLE" == "1" ]]; then
    while IFS=$'\t' read -r label run_dir; do
        [[ -n "$run_dir" ]] || continue
        sub="$OUT_DIR/${label// /_}"
        mkdir -p "$sub"
        run_step "profiles:$label" $RUNNER plot_profiles.py --run-dir "$run_dir" --out-dir "$sub"
        run_step "basis:$label"    $RUNNER plot_basis.py    --run-dir "$run_dir" --out "$sub/learned_basis.png"
    done < <($RUNNER - "$PLOT_CFG" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])) or {}
for name, v in (d.get("runs") or {}).items():
    print(f"{v.get('label', name)}\t{v['path']}")
PY
    )
fi

echo "figures written to $OUT_DIR"
