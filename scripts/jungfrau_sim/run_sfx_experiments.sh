#!/usr/bin/env bash
# Full SFX per-image-Wilson experiment matrix. Edit N_IMAGES / EPOCHS / DEVICE for the
# cluster, then run this from the repo root. Each arm writes to data/sfx_runs/<tag>/ and
# is independent, so you can also launch them as separate cluster jobs (one line each).
#
# Prereq: generate the dataset once (per-image G, global B, 100 refl/image):
#   uv run python scripts/jungfrau_sim/sfx_generate.py --n-images "$N_IMAGES" --n-refl 100 --out data/sfx_sim
#   # add --sigma-b 5 for the per-image-B dataset (then also pass --per-image-B below).
set -euo pipefail

N_IMAGES=${N_IMAGES:-2000}
EPOCHS=${EPOCHS:-400}
DEVICE=${DEVICE:-auto}
DATA=${DATA:-data/sfx_sim}
OUT=${OUT:-data/sfx_runs}
COMMON="--data $DATA --out $OUT --epochs $EPOCHS --device $DEVICE --eval-every 5 --resume"

run () { echo ">>> $*"; uv run python -u scripts/jungfrau_sim/sfx_experiment.py $COMMON "$@"; }

# 1. Headline likelihood comparison on the REAL model (learned-basis profile, per-image G).
run --likelihood poisson        --profile learned --scale per_image
run --likelihood normal_coupled --profile learned --scale per_image
run --likelihood normal_free    --profile learned --scale per_image

# 2. Oracle-profile controls: isolates scale/intensity recovery from profile learning.
run --likelihood poisson        --profile known --scale per_image
run --likelihood normal_coupled --profile known --scale per_image

# 3. Per-image G vs a single global G (the architecture question).
run --likelihood poisson        --profile learned --scale global

# 4. Per-image B (needs a dataset generated with --sigma-b > 0).
# run --likelihood poisson      --profile learned --scale per_image --per-image-B

echo ">>> analysis"
uv run python scripts/jungfrau_sim/sfx_analyze.py --runs "$OUT" --data "$DATA"
