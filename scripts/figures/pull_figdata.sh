#!/bin/bash
# Bring the plot-ready tables down from Cannon into docs/figures/data/.
#
# Figures are rendered here, not on the cluster: the documents are set in
# Minion Pro with mtpro2 maths through LuaLaTeX and those fonts are installed
# on this laptop only. So the cluster reduces each measurement to a
# long-format CSV of exactly the columns a figure draws -- tens of rows, not
# the 1.94 million reflections they were computed from -- and only those
# files cross the wire.
#
# Usage: bash scripts/figures/pull_figdata.sh [name ...]
#        with no argument, every table listed below.
set -u
S=/n/netscratch/hekstra_lab/Lab/laldama
J=$S/jobs/geomshape_1116_1730
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DEST=$HERE/docs/figures/data
P=/n/lab_storage/hekstra_lab/people/aldama/micromamba/envs/integrator-cuda-dev/bin/python
mkdir -p "$DEST"

# name                       how the cluster makes it
# profile_shape/_width       tidy_shape.py over the latest geom_profile_transfer
# arm_metrics                tidy_xtal.py over the DIALS logs
# dials_logs                 scale.log and merge.log per arm, read by
#                            scripts/figures/make_tables.py
# careless_stats             the poly pipeline's stats step
# pdz2                       tidy_pdz2.py over the PDZ2 arms: CCsym, the
#                            refinement, the training loss and the per-arm
#                            careless.stats tables
# poly                       tidy_poly.py over the 1118 ablation arms and the
#                            nanoBragg DHFR simulation arms
# hparams                    tidy_hparams.py over each run's config_log.yaml:
#                            what every model was configured with, with a
#                            `varies` flag across the arms compared
# basis                      dump_basis.py over the checkpoints: the learned
#                            profile basis per epoch, as npz plus a
#                            per-component table
# peak_models                the refined PDB beside each peaks.csv, which is
#                            what names the atom carrying each peak
want=("$@")
[ ${#want[@]} -eq 0 ] && want=(mono scatter profile_shape profile_anisotropy dials_logs arm_metrics)

has() { printf '%s\n' "${want[@]}" | grep -qx "$1"; }

if has profile_shape; then
  # the latest results directory wins: prof, then prof2 (reference profiles
  # re-centred before averaging), then prof3 (every arm). `ls -d a b c` sorts
  # its arguments, so the order has to be imposed one test at a time.
  SRC=$(ssh cannon "for d in $J/prof3 $J/prof2 $J/prof; do \
    [ -s \$d/geom_profile_transfer.json ] && { echo \$d; break; }; done")
  if [ -n "$SRC" ]; then
    echo "profile_shape <- $SRC"
    ssh cannon "$P $J/tidy_shape.py $SRC/geom_profile_transfer.json \
      $SRC/profile_shape.csv $SRC/profile_width.csv hewl_9b7c" || exit 1
    scp -q "cannon:$SRC/profile_shape.csv" "cannon:$SRC/profile_width.csv" "$DEST/"
  else
    echo "profile_shape: no results directory on the cluster yet"
  fi
fi

if has profile_anisotropy; then
  if ssh cannon "[ -s $J/aniso.json ]"; then
    echo "profile_anisotropy <- $J/aniso.json"
    ssh cannon "$P $J/tidy_shape.py --aniso $J/aniso.json \
      $J/profile_anisotropy.csv hewl_9b7c" || exit 1
    scp -q "cannon:$J/profile_anisotropy.csv" "$DEST/"
  else
    echo "profile_anisotropy: aniso.json not on the cluster"
  fi
fi

if has mono; then
  # the monochromatic set: training loss, merging statistics by shell,
  # refinement, anomalous peak heights
  echo "mono <- $J/mono"
  ssh cannon "cd $S && PYTHONPATH=$S/integrator_plumbing/src $P $J/tidy_mono.py $J/mono ${MONO_ARMS:-m9_gate_learned m9_geomshift m9_geom dials_ref}" || exit 1
  for f in training_loss wilson_params merging_shells refinement anomalous_peaks; do
    ssh cannon "[ -s $J/mono/$f.csv ]" && scp -q "cannon:$J/mono/$f.csv" "$DEST/"
  done
fi

if has scatter; then
  # model against the reference integrator, per reflection: the points at the
  # last epoch and the correlation within resolution bins at every epoch.
  # `plot_compare.py --dump-dir` does the reduction where the 1.9 million
  # reflections are; it draws nothing and needs no figTeX.
  echo "scatter <- $J/scatter"
  scp -q "$HERE/plot_compare.py" "cannon:$J/plot_compare.py" || exit 1
  ssh cannon "cd $S && PYTHONPATH=$S/integrator_plumbing/src $P \
    $J/plot_compare.py --plot-cfg $J/cfg_9b7c_scatter.yaml \
    --dump-dir $J/scatter --dataset hewl_9b7c --split test" || exit 1
  for f in scatter_points scatter_cc; do
    ssh cannon "[ -s $J/scatter/$f.csv ]" && scp -q "cannon:$J/scatter/$f.csv" "$DEST/"
  done
fi

if has patch; then
  # images, not a table: the detector patch and the profiles drawn in it,
  # dumped by `geom_position_figure.py` as it selects them
  if ssh cannon "[ -s $J/pos/patch_data.npz ]"; then
    echo "patch <- $J/pos/patch_data.npz"
    scp -q "cannon:$J/pos/patch_data.npz" "$DEST/"
  else
    echo "patch: patch_data.npz not on the cluster yet"
  fi
fi

if has dials_logs; then
  # `dials.scale` and `dials.merge` report a good deal that never reaches
  # merging_stats.csv -- the error model, every R-factor, the Wilson
  # analysis, the anomalous differences by shell. The logs are ~16 kB each,
  # so they come down whole and the tables are built here, which is what
  # lets a table definition change without another cluster trip.
  mkdir -p "$DEST/dials_logs"
  for n in ${DIALS_ARMS:-m9_gate_learned m9_geom m9_geomshift m9_geomresp m9_geomrespsteer}; do
    for e in rb model; do
      d=$S/dials9b7c/${n}_${e}
      if ssh cannon "[ -s $d/merge.log ]"; then
        echo "dials_logs <- ${n}_${e}"
        mkdir -p "$DEST/dials_logs/${n}_${e}"
        scp -q "cannon:$d/scale.log" "cannon:$d/merge.log" \
          "$DEST/dials_logs/${n}_${e}/" || true
      else
        echo "dials_logs: ${n}_${e} not on the cluster"
      fi
    done
  done
fi

if has peak_models; then
  # `rs.find_peaks` has a `name` column for the nearest atom and the fork in
  # use leaves it empty, so the atom carrying each peak is resolved against
  # the refined model the peak search itself ran on -- a sulfur peak is SD on
  # a methionine and SG on a cysteine, and the residue name does not say
  # which. Without this the per-site panels title themselves `CYS 30` rather
  # than `CYS 30 SG`; nothing else changes.
  #
  # PEAK_RUN is the run directory on the cluster whose per-epoch phenix
  # output the local mirror under docs/figures/data/runs was taken from. It
  # has no default: the mirror does not record where it came from.
  if [ -z "${PEAK_RUN:-}" ]; then
    echo "peak_models: set PEAK_RUN=<cluster run directory> (the one whose"
    echo "             predictions/epoch_*/dials/phenix_out the local mirror"
    echo "             under docs/figures/data/runs was copied from)"
  else
    n=0
    for local_csv in "$DEST"/runs/*/predictions/epoch_*/dials/phenix_out/peaks.csv; do
      [ -e "$local_csv" ] || continue
      ep=$(basename "$(dirname "$(dirname "$(dirname "$local_csv")")")")
      src=$PEAK_RUN/predictions/$ep/dials/phenix_out
      if ssh cannon "ls $src/*[0-9].pdb >/dev/null 2>&1"; then
        scp -q "cannon:$src/refine_001.pdb" "$(dirname "$local_csv")/" \
          && n=$((n + 1))
      fi
    done
    echo "peak_models: $n refined models"
  fi
fi

if has careless_stats; then
  # the polychromatic per-shell table, written by the pipeline's stats step
  if ssh cannon "[ -s $J/stats/careless_stats.csv ]"; then
    echo "careless_stats <- $J/stats"
    scp -q "cannon:$J/stats/careless_stats.csv" "$DEST/"
  else
    echo "careless_stats: not built yet (run_pipeline.py --steps stats)"
  fi
fi

if has arm_metrics; then
  if ssh cannon "[ -s $J/arm_metrics.csv ]"; then
    echo "arm_metrics"
    scp -q "cannon:$J/arm_metrics.csv" "$DEST/"
  else
    echo "arm_metrics: not built yet (run tidy_xtal.py on the cluster)"
  fi
fi

if has pdz2; then
  # PDZ2 is not in the run-directory layout: it scaled through the deposited
  # tutorial script, so the stats step is pointed at each arm's careless
  # output base and `tidy_pdz2.py` gathers the four tables.
  #
  # The careless.stats tables have to exist first. They are made by, per arm,
  #   $P $PJ/run_pipeline.py --cfg $PJ/pdz2_stats_cfg.yaml \
  #       --out-base <tree>/gain_0,3_from_stills/careless-cdef-0,3-ohp-mlpw/merged_e35 \
  #       --method <arm> --dataset pdz2_1116
  # which is CPU work and takes about two minutes an arm.
  PJ=${PDZ2_JOB:-$S/jobs/pdz2figs_1118}
  echo "pdz2 <- $PJ"
  # always, not only when missing: a staged reducer that is never refreshed
  # silently serves the version it was first copied with. `tidy_pdz2.py`
  # gained the svae arm here and the cluster kept running the copy without
  # it, so the pull came back without that arm and nothing said why.
  scp -q "$HERE/scripts/figures/tidy_pdz2.py" "cannon:$PJ/"
  ssh cannon "PYTHONPATH=$PJ $P $PJ/tidy_pdz2.py $PJ" || exit 1
  for f in pdz2_ccsym pdz2_refinement pdz2_training_loss careless_stats_pdz2; do
    ssh cannon "[ -s $PJ/$f.csv ]" && scp -q "cannon:$PJ/$f.csv" "$DEST/"
  done
fi

if has models; then
  # Parameter counts and training throughput per arm, from
  # `compare_models.py`: the svae-against-hierarchical family comparison.
  # `cmp_timing` is the matched-allocation measurement, kept separate from
  # the per-dataset tables because those were read off checkpoint
  # timestamps on whatever node the run happened to get, and only the
  # matched one supports a claim about speed.
  PJ=${PDZ2_JOB:-$S/jobs/pdz2figs_1118}
  echo "models <- $PJ"
  for f in cmp_mono cmp_1118 cmp_pdz2 cmp_timing; do
    if ssh cannon "[ -s $PJ/$f.csv ]"; then
      scp -q "cannon:$PJ/$f.csv" "$DEST/"
    else
      echo "  $f.csv: not built (run compare_models.py on the cluster)"
    fi
  done
fi

if has poly; then
  # the polychromatic arms, in the same columns the monochromatic tables
  # use, so one figure draws either. 1118 scales and refines; the nanoBragg
  # DHFR simulation arms are scored against truth instead, so they have
  # loss, Wilson and spectrum and nothing downstream -- that is a complete
  # arm, not a broken one, and the two are concatenated per table.
  PJ=${PDZ2_JOB:-$S/jobs/pdz2figs_1118}
  echo "poly <- $PJ"
  # always, not only when missing: a staged reducer that is never refreshed
  # silently serves the version it was first copied with. `tidy_pdz2.py`
  # gained the svae arm here and the cluster kept running the copy without
  # it, so the pull came back without that arm and nothing said why.
  scp -q "$HERE/scripts/figures/tidy_poly.py" "cannon:$PJ/"
  ssh cannon "export PYTHONPATH=$PJ
    ROOT=$S/ablate2 DATASET=hewl_1118 $P $PJ/tidy_poly.py $PJ/poly1118 \
      \$(ls -1 $S/ablate2 | grep -E '^q[0-9]' | tr '\n' ' ')
    ROOT=$S/sim_runs DATASET=dhfr_sim $P $PJ/tidy_poly.py $PJ/polysim \
      ${SIM_ARMS:-sim_qc sim_qc_resp sim_qc_sigma sim_qc_sigma_resp \
      sim_qc_amortized sim_qc_sigma_amortized sim_shapelet \
      sim_shapelet_learned sim_geom_full sim_geom_full_amortized}" || exit 1
  tmp=$(mktemp -d)
  for f in poly_training_loss poly_wilson_params poly_spectrum \
           poly_merging_shells poly_refinement poly_anomalous_peaks \
           poly_ccanom; do
    scp -q "cannon:$PJ/poly1118/$f.csv" "$DEST/$f.csv" 2>/dev/null
    if scp -q "cannon:$PJ/polysim/$f.csv" "$tmp/$f.csv" 2>/dev/null \
       && [ -s "$tmp/$f.csv" ]; then
      tail -n +2 "$tmp/$f.csv" >> "$DEST/$f.csv"
    fi
  done
fi

if has hparams; then
  # What each model was configured with, from each run's own config_log.yaml.
  # `varies` is a property of the SET, so every arm belonging in one table is
  # passed in one invocation and the four datasets are run separately.
  PJ=${PDZ2_JOB:-$S/jobs/pdz2figs_1118}
  echo "hparams <- $PJ"
  # always, not only when missing: a staged reducer that is never refreshed
  # silently serves the version it was first copied with. `tidy_pdz2.py`
  # gained the svae arm here and the cluster kept running the copy without
  # it, so the pull came back without that arm and nothing said why.
  scp -q "$HERE/scripts/figures/tidy_hparams.py" "cannon:$PJ/"
  ssh cannon "export PYTHONPATH=$PJ
    DATASET=hewl_9b7c $P $PJ/tidy_hparams.py $PJ/hp_mono \
      $S/mono9b7c/hewl9b7c_mono_expmean_lp $S/mono9b7c/hewl9b7c_svae \
      $S/sim_runs/m9_gate_learned $S/sim_runs/m9_geomshift $S/sim_runs/m9_geom \
      $S/sim_runs/m9_geomresp $S/sim_runs/m9_geomrespsteer
    DATASET=hewl_1118 $P $PJ/tidy_hparams.py $PJ/hp_1118 \
      $S/full1000/learned $S/hier1000/softmax_std \
      \$(ls -1d $S/ablate2/q* | tr '\n' ' ')
    DATASET=pdz2_1116 $P $PJ/tidy_hparams.py $PJ/hp_pdz2 \
      \$(ls -1d $S/laue_runs/pdz2_model/runs/pdz2_* | tr '\n' ' ')
    DATASET=dhfr_sim $P $PJ/tidy_hparams.py $PJ/hp_sim \
      ${SIM_HP_ARMS:-$S/sim_runs/sim_qc $S/sim_runs/sim_qc_resp \
      $S/sim_runs/sim_qc_sigma $S/sim_runs/sim_shapelet \
      $S/sim_runs/sim_geom_full}" || exit 1
  tmp=$(mktemp -d)
  for set in mono 1118 pdz2 sim; do
    scp -q "cannon:$PJ/hp_$set/hparams.csv" "$tmp/h_$set.csv" 2>/dev/null
    scp -q "cannon:$PJ/hp_$set/hparam_labels.csv" "$tmp/l_$set.csv" 2>/dev/null
  done
  if [ -s "$tmp/h_mono.csv" ]; then
    head -1 "$tmp/h_mono.csv" > "$DEST/hparams.csv"
    for set in mono 1118 pdz2 sim; do
      [ -s "$tmp/h_$set.csv" ] && tail -n +2 "$tmp/h_$set.csv" >> "$DEST/hparams.csv"
    done
    head -1 "$tmp/l_mono.csv" > "$DEST/hparam_labels.csv"
    cat "$tmp"/l_*.csv | grep -v "^parameter," | sort -u >> "$DEST/hparam_labels.csv"
  fi
fi

if has basis; then
  # The learned profile basis per epoch, recovered from the checkpoints:
  # `ProfileBasisLogger` would have written these during training but is gated
  # behind `integrator.train --figures`, which no production run passed.
  # Images stay in the npz because an image is not a table; the per-component
  # norms and cosines come down as one CSV.
  PJ=${PDZ2_JOB:-$S/jobs/pdz2figs_1118}
  # `basis_<set>` is one dataset's arms; `basis_cmp_<set>` is a matched
  # svae-against-hierarchical pair on one dataset, which is what the
  # family comparison draws, so the two land in separate directories and
  # `fig_profile_basis.py --data .../basis_cmp` selects between them.
  for set in m9 pdz2 cmp_mono cmp_1118 cmp_pdz2; do
    src=$PJ/basis_$set
    case "$set" in cmp_*) sub=basis_cmp; tag=${set#cmp_};; *) sub=basis; tag=$set;; esac
    if ssh cannon "[ -s $src/basis_components.csv ]"; then
      echo "basis ($set) <- $src"
      mkdir -p "$DEST/$sub"
      scp -qr "cannon:$src/"*/ "$DEST/$sub/" 2>/dev/null
      scp -q "cannon:$src/basis_components.csv" \
        "$DEST/$sub/basis_components_$tag.csv"
    else
      echo "basis ($set): not built yet (run dump_basis.py on the cluster)"
    fi
  done
  # one table per set, but the figure reads one file
  for sub in basis basis_cmp; do
    ls "$DEST"/$sub/basis_components_*.csv >/dev/null 2>&1 || continue
    head -1 "$(ls "$DEST"/$sub/basis_components_*.csv | head -1)" \
      > "$DEST/$sub/basis_components.csv"
    for f in "$DEST"/$sub/basis_components_*.csv; do
      case "$f" in *basis_components.csv) continue;; esac
      tail -n +2 "$f" >> "$DEST/$sub/basis_components.csv"
    done
  done
fi

ls -la "$DEST"
