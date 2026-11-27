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
    want=(shoebox loss wilson basis basiscmp cost scatter merging peaks spectrum careless ccsym pdz2 refine shape patch arms simsampling simimage simtruth simshoebox simtruthpc simprofile simcmp simsbc simprior simcrps simprofcmp hewl1118profiles simmosaic simtruth025 simtruth025b)
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
has simtruth && [ -f "$DATA/sim_truth_joined_mos010.csv" ] && step simtruth "${PY[@]}" scripts/figures/fig_sim_truth_scatter.py
has simshoebox && [ -f "$DATA/shoebox_examples.npz" ] && step simshoebox "${PY[@]}" scripts/figures/fig_sim_shoeboxes.py
# the same reflections under mosaic spread 0.10 and 0.25 (mos010, mos025)
has simmosaic && [ -f "$DATA/shoebox_examples_mosaic.npz" ] && step simmosaic "${PY[@]}" scripts/figures/fig_sim_shoeboxes.py \
  --src "$DATA/shoebox_examples_mosaic.npz" --out sim_shoeboxes_mosaic
# mos010 photon-counting: laue.integrate and the svae arms, both estimators, against truth
SIMPC="$DATA/sim_truth_joined_mos010_pc.csv.gz"
has simtruthpc && [ -f "$SIMPC" ] && step simtruthpc "${PY[@]}" scripts/figures/fig_sim_truth_scatter.py --src "$SIMPC" --psf-fraction 0
has simtruthpc && [ -f "$DATA/sim_truth_scores_mos010_pc.csv" ] && step simtruthpc "${PY[@]}" scripts/figures/fig_sim_truth_scores.py
has simtruthpc && [ -f "$DATA/sim_truth_scores_by_d_mos010_pc.csv" ] && step simtruthpc "${PY[@]}" scripts/figures/fig_sim_truth_scores.py --by-d "$DATA/sim_truth_scores_by_d_mos010_pc.csv"
# mos025 (mosaic spread 0.25, laue-dials n_macrocycles=0): the arms against truth, as simtruthpc / simcrps / simcmp do for mos010
S25="$DATA/sim_truth_joined_mos025_pc.csv.gz"
if has simtruth025 && [ -f "$S25" ]; then
  step simtruth025 "${PY[@]}" scripts/figures/fig_sim_truth_scatter.py --src "$S25" --psf-fraction 0 --tag mos025
  step simtruth025 "${PY[@]}" scripts/figures/fig_sim_truth_scores.py --src "$DATA/sim_truth_scores_mos025_pc.csv" --tag mos025
  step simtruth025 "${PY[@]}" scripts/figures/fig_sim_truth_scores.py --src "$DATA/sim_truth_scores_mos025_pc.csv" \
    --by-d "$DATA/sim_truth_scores_by_d_mos025_pc.csv" --tag mos025
  step simtruth025 "${PY[@]}" scripts/figures/fig_sim_crps.py --src "$DATA/sim_crps_mos025_pc.csv" --base b1_rb
  for est in rb posterior; do
    step simtruth025 "${PY[@]}" scripts/figures/fig_intensity_scatter.py --data "$S25" --id shot,h,k,l --value i_hat --in-domain \
      --dataset mos025_pc_m0_default --methods laue_integrate,svae_bg_$est,t0_$est,p2_$est,b1_$est \
      --out-dir docs/figures/sim/cmp --prefix mos025_pc_m0_default_$est
  done
fi
# the same on 41x41 shoeboxes (pc_m0_default_41), where the 25x25 window held 0.836 of the truth photons
S41="$DATA/sim_truth_joined_mos025_41_pc.csv.gz"
if has simtruth025b && [ -f "$S41" ]; then
  step simtruth025b "${PY[@]}" scripts/figures/fig_sim_truth_scatter.py --src "$S41" --psf-fraction 0 --tag mos025_41
  step simtruth025b "${PY[@]}" scripts/figures/fig_sim_truth_scores.py --src "$DATA/sim_truth_scores_mos025_41_pc.csv" --tag mos025_41
  step simtruth025b "${PY[@]}" scripts/figures/fig_sim_truth_scores.py --src "$DATA/sim_truth_scores_mos025_41_pc.csv" \
    --by-d "$DATA/sim_truth_scores_by_d_mos025_41_pc.csv" --tag mos025_41
  step simtruth025b "${PY[@]}" scripts/figures/fig_sim_crps.py --src "$DATA/sim_crps_mos025_41_pc.csv" --base b1_rb
  for est in rb posterior; do
    step simtruth025b "${PY[@]}" scripts/figures/fig_intensity_scatter.py --data "$S41" --id shot,h,k,l --value i_hat --in-domain \
      --dataset mos025_pc_m0_default41 --methods laue_integrate,svae_bg_$est,t0_$est,p2_$est,b1_$est \
      --out-dir docs/figures/sim/cmp --prefix mos025_pc_m0_default41_$est
  done
fi
has simprofile && ls "$DATA"/profile_truth_examples_*.npz >/dev/null 2>&1 && step simprofile "${PY[@]}" scripts/figures/fig_profile_truth_examples.py
# simulation-based calibration, one directory per scored run (scripts/sim_truth/sbc_ranks.py)
has simsbc && ls "$DATA"/sbc/*/sbc_ecdf.csv >/dev/null 2>&1 && step simsbc "${PY[0]}" "${PY[1]}" --with "arviz<1" "${PY[@]:2}" scripts/figures/fig_sim_sbc.py --src "$DATA"/sbc/*/
# prior checks of the Wilson intensity prior (scripts/sim_truth/prior_check.py)
has simprior && [ -f "$DATA/prior_check/prior_check_summary.csv" ] && step simprior "${PY[@]}" scripts/figures/fig_sim_prior_check.py --src "$DATA/prior_check"
# CRPS of every export against the truth (scripts/sim_truth/crps_against_truth.py)
has simcrps && [ -f "$DATA/sim_crps_mos010_pc.csv" ] && step simcrps "${PY[@]}" scripts/figures/fig_sim_crps.py --src "$DATA/sim_crps_mos010_pc.csv"
# hewl_1118: counts and each arm's profile for the named reflections, and the mean profile per
# detector region, with laue.integrate (upstream main) as a column when its reduced profiles are here
# (laue_profiles.py --reduce-from; pull_figdata.sh hewl1118_profiles)
H1118_LAUE=()
[ -f "$DATA/laue_profiles_hewl1118_main.npz" ] && H1118_LAUE=(--laue-profiles "$DATA/laue_profiles_hewl1118_main.npz" --laue-label "laue-dials")
has hewl1118profiles && [ -f "$DATA/profile_maps_hewl1118.npz" ] && step hewl1118profiles "${PY[@]}" scripts/figures/fig_profile_gate_examples.py \
  --data "$DATA/profile_maps_hewl1118.npz" --out-dir "$HERE/docs/figures/hewl1118" --prefix profiles_hewl1118 \
  --profiles-only --regions --labels hewl1118_svae="learned basis" hewl1118_kmix_base=T0 \
  hewl1118_kmix_h128_grid05=P2 hewl1118_kmixv2_post=B1 ${H1118_LAUE[@]+"${H1118_LAUE[@]}"}
# true profiles beside prior and Laplace-mode profiles (scripts/sim_truth/dump_profile_compare.py)
has simprofcmp && ls "$DATA"/profile_compare_*.npz >/dev/null 2>&1 && for f in "$DATA"/profile_compare_*.npz; do step simprofcmp "${PY[@]}" scripts/figures/fig_sim_profile_compare.py --src "$f"; done
if has simcmp && [ -f "$SIMPC" ]; then
  for ds in mos010_pc_default mos010_pc_permissive; do for est in rb posterior; do
    step simcmp "${PY[@]}" scripts/figures/fig_intensity_scatter.py --data "$SIMPC" --id shot,h,k,l --value i_hat \
      --in-domain --dataset $ds --methods laue_integrate,svae_$est,svae_bg_$est,svae_conjugate_$est \
      --out-dir docs/figures/sim/cmp --prefix ${ds}_$est
  done; done
fi

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
