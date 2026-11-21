#!/bin/bash
# Re-render every reported figure from docs/figures/data/, then rebuild the
# gallery. This is the command to run after a change to figTeX, a colour
# scheme or the mydesign LaTeX styles: it is the only way to see what such a
# change did to the whole set rather than to one figure.
#
#   bash scripts/figures/render_all.sh                 # all of it
#   bash scripts/figures/render_all.sh merging peaks    # two targets
#   OUT=/tmp/try bash scripts/figures/render_all.sh     # somewhere else
#   TEMPLATE=slides bash scripts/figures/render_all.sh  # the PDZ2 set, sized
#                                                        # for one template
#
# With TEMPLATE set, the default targets are the PDZ2 set (ccsym pdz2 refine
# basis basiscmp cost), every one is drawn with `--template "$TEMPLATE"` into
# OUT_DIR (default docs/figures/templates/$TEMPLATE), the basis figures are
# limited to the PDZ2 arms, and the gallery is not rebuilt: the template
# copies are the same figures at other sizes. Without TEMPLATE nothing here
# differs from before it existed.
#
# Rendering is local by necessity: the documents are set in Minion Pro with
# mtpro2 maths through LuaLaTeX and those fonts are on this laptop only. The
# tables these read are small and already here, so a full run costs minutes
# and no cluster time -- `pull_figdata.sh` is what goes to Cannon.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$HERE"
DATA=${DATA:-$HERE/docs/figures/data}
OUT=${OUT:-$HERE/docs/figures/m9}
CFG=${CFG:-$DATA/cfg_9b7c.yaml}
FIGTEX=${FIGTEX:-/Users/luis/master/software/figtex}
PY=(uv run --with-editable "$FIGTEX" python)

PDZ2_OUT=${PDZ2_OUT:-$HERE/docs/figures/pdz2}
TEMPLATE=${TEMPLATE:-}
# per-template arguments; empty, and expanded to nothing, without TEMPLATE.
# Written `${A[@]+"${A[@]}"}` below because bash 3.2, which is /bin/bash on
# this laptop, calls an empty array unbound under `set -u`
TPL=()      # --template, for every step
TOUT=()     # --out-dir, for the steps that otherwise choose their own
BASIS=()    # the PDZ2 arms only
SPLIT=()    # basis_convergence one row per quantity: the grid is too tall
            # for a slide frame
if [ -n "$TEMPLATE" ]; then
  OUT_DIR=${OUT_DIR:-$HERE/docs/figures/templates/$TEMPLATE}
  PDZ2_OUT=$OUT_DIR
  TPL=(--template "$TEMPLATE")
  TOUT=(--out-dir "$OUT_DIR")
  BASIS=(--dataset pdz2_1116)
  [ "$TEMPLATE" = slides ] && SPLIT=(--split)
  mkdir -p "$OUT_DIR"
else
  mkdir -p "$OUT" "$OUT/peaks" "$OUT/scatter" "$PDZ2_OUT"
fi

want=("$@")
if [ ${#want[@]} -eq 0 ]; then
  if [ -n "$TEMPLATE" ]; then
    want=(ccsym pdz2 refine basis basiscmp cost)
  else
    want=(shoebox loss wilson basis basiscmp cost scatter merging peaks spectrum careless ccsym pdz2 refine shape patch arms simsampling simimage)
  fi
fi
has() { printf '%s\n' "${want[@]}" | grep -qx "$1"; }

failed=()
step() {  # step <name> <command...>
  local name=$1; shift
  echo "===== $name ====="
  if ! "$@"; then
    echo "!!!!! $name failed" >&2
    failed+=("$name")
  fi
}

# input shoeboxes against the predicted profiles
has shoebox && step shoebox "${PY[@]}" plot_profile_grid.py \
  --data "$DATA/shoebox_profiles.npz" --out "$OUT/shoebox_profiles.png"

# the ELBO and its parts, train against validation
has loss && step loss "${PY[@]}" plot_loss.py \
  --plot-cfg "$CFG" --out-dir "$OUT"

# model against the reference integrator, per reflection
has scatter && step scatter "${PY[@]}" plot_compare.py \
  --data "$DATA" --out-dir "$OUT/scatter" --dataset hewl_9b7c

# merging statistics by resolution shell, one line per checkpoint
has merging && step merging "${PY[@]}" plot_merging.py \
  --plot-cfg "$CFG" --out-dir "$OUT"

# refinement R-factors and anomalous peak heights over training
has peaks && step peaks "${PY[@]}" plot_peaks.py \
  --plot-cfg "$CFG" --out-dir "$OUT/peaks"

# polychromatic: the learned spectrum
has spectrum && step spectrum "${PY[@]}" scripts/visualize_spectrum.py \
  --data "$DATA/spectrum.csv" --out "$OUT/spectrum.png"

# polychromatic: whatever careless.stats reported, by shell. The table is
# written by `scripts/poly/run_pipeline.py --steps stats` on the cluster and
# is absent until a polychromatic run has been scored, so this target is
# skipped rather than failed when it is not there.
if has careless; then
  if [ -s "$DATA/careless_stats.csv" ]; then
    step careless "${PY[@]}" scripts/figures/fig_careless_stats.py \
      --data "$DATA/careless_stats.csv" --out-dir "$OUT"
  else
    echo "===== careless ====="
    echo "careless: no $DATA/careless_stats.csv yet (poly stats step)"
  fi
fi

# the Wilson prior's learned scale and B-factor over training
if has wilson; then
  if [ -s "$DATA/wilson_params.csv" ]; then
    step wilson "${PY[@]}" scripts/figures/fig_wilson_params.py \
      --data "$DATA/wilson_params.csv" --out-dir "$OUT"
  else
    echo "===== wilson ====="
    echo "wilson: no $DATA/wilson_params.csv yet (pull_figdata.sh mono)"
  fi
fi

# every component of the learned profile basis, and how it settled. The
# figure routes each arm to its own dataset's directory, so no --out-dir
if has basis; then
  if [ -d "$DATA/basis" ]; then
    step basis "${PY[@]}" scripts/figures/fig_profile_basis.py \
      --data "$DATA/basis" ${TPL[@]+"${TPL[@]}"} ${TOUT[@]+"${TOUT[@]}"} \
      ${BASIS[@]+"${BASIS[@]}"}
  else
    echo "===== basis ====="
    echo "basis: no $DATA/basis yet (pull_figdata.sh basis)"
  fi
fi

# PDZ2: CCsym by shell, one panel per pump delay
if has ccsym; then
  if [ -s "$DATA/pdz2_ccsym.csv" ]; then
    step ccsym "${PY[@]}" scripts/figures/fig_ccsym.py \
      --data "$DATA/pdz2_ccsym.csv" --out-dir "$PDZ2_OUT" \
      ${TPL[@]+"${TPL[@]}"}
  else
    echo "===== ccsym ====="
    echo "ccsym: no $DATA/pdz2_ccsym.csv yet (pull_figdata.sh pdz2)"
  fi
fi

# PDZ2: what careless reports per shell, drawn by the same script as 1118
if has pdz2; then
  if [ -s "$DATA/careless_stats_pdz2.csv" ]; then
    step pdz2 "${PY[@]}" scripts/figures/fig_careless_stats.py \
      --data "$DATA/careless_stats_pdz2.csv" --out-dir "$PDZ2_OUT" \
      ${TPL[@]+"${TPL[@]}"}
  else
    echo "===== pdz2 ====="
    echo "pdz2: no $DATA/careless_stats_pdz2.csv yet (pull_figdata.sh pdz2)"
  fi
fi

# svae against hierarchical: the matched pairs, and what the families cost
if has basiscmp; then
  if [ -d "$DATA/basis_cmp" ]; then
    step basiscmp "${PY[@]}" scripts/figures/fig_basis_convergence.py \
      ${TPL[@]+"${TPL[@]}"} ${TOUT[@]+"${TOUT[@]}"} ${SPLIT[@]+"${SPLIT[@]}"}
  else
    echo "===== basiscmp ====="
    echo "basiscmp: no $DATA/basis_cmp yet (pull_figdata.sh basis)"
  fi
fi

if has cost; then
  if ls "$DATA"/cmp_*.csv >/dev/null 2>&1; then
    step cost "${PY[@]}" scripts/figures/fig_model_cost.py \
      ${TPL[@]+"${TPL[@]}"} ${TOUT[@]+"${TOUT[@]}"}
  else
    echo "===== cost ====="
    echo "cost: no $DATA/cmp_*.csv yet (pull_figdata.sh models)"
  fi
fi

# PDZ2 refinement against the deposited model, per arm
if has refine; then
  if [ -s "$DATA/pdz2_refinement.csv" ]; then
    step refine "${PY[@]}" scripts/figures/fig_pdz2_refinement.py \
      --out-dir "$PDZ2_OUT" ${TPL[@]+"${TPL[@]}"}
  else
    echo "===== refine ====="
    echo "refine: no $DATA/pdz2_refinement.csv yet (pull_figdata.sh pdz2)"
  fi
fi

# kept from the geometry investigation
has shape && step shape "${PY[@]}" scripts/figures/fig_profile_shape.py
has patch && step patch "${PY[@]}" scripts/figures/fig_detector_patch.py

# datsimx sampling check (docs/sim_truth.md)
has simsampling && [ -f "$DATA/sim_sampling_patches.npz" ] && step simsampling "${PY[@]}" scripts/figures/fig_sim_sampling.py
has simimage && [ -f "$DATA/sim_image_view.npz" ] && step simimage "${PY[@]}" scripts/figures/fig_sim_image.py

# a diagnostic, not a reported figure
has arms && step arms "${PY[@]}" scripts/figures/fig_arm_metrics.py

if [ -n "$TEMPLATE" ]; then
  echo "===== gallery: skipped for TEMPLATE=$TEMPLATE ====="
else
echo "===== gallery ====="
# over docs/figures rather than $OUT: the set now spans two dataset
# directories and a gallery of one of them is not the whole set
uv run python scripts/figures/build_gallery.py \
  "${GALLERY_BASE:-$HERE/docs/figures}" "$HERE/docs/figures/gallery.html"
fi

if [ ${#failed[@]} -gt 0 ]; then
  echo
  echo "failed: ${failed[*]}" >&2
  exit 1
fi
