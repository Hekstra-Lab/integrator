# Experiments, comparisons, and where the data is — 18 November 2026

Supersedes the 10 October list.
Read with `methods.md` (how each dataset was processed, and which script drives which tool), `scripts.md` (every script and what it does), `results_by_dataset.md` (what the numbers came out as) and `experiment_provenance.md` (which code produced them).

## Running now

| job | what | reads out as |
|---|---|---|
| **61851629** | RB export of `pdz2_geom_both`, third attempt | `i_rb`/`var_rb` in the prediction parquet |
| **61854673** | `tim_hier`, 12 epochs | training throughput, hierarchical, on a known allocation |
| **61854684** | `tim_svae`, 12 epochs, chained after 61854673 | the same for svae, on the same node by construction |

Finished since the last revision: **61766712**, PDZ2 svae training, 100 epochs,
val ELBO 1006.560 and still falling by about 0.005 per epoch at the end.

## The three comparisons

### 1. Learned basis, svae against hierarchical

The arms must differ by the integrator family **alone**, so the pairs are chosen on that and not on what happens to exist.

| dataset | svae | hierarchical | state |
|---|---|---|---|
| hewl_9b7c | `mono9b7c/hewl9b7c_svae` | `mono9b7c/hewl9b7c_mono_expmean_lp` | **done** — both bases dumped, 10 epochs each, basis (1323, 12) |
| hewl_1118 | `full1000/learned` | `hier1000/softmax_std` | **done** — 4 epochs each, basis (625, 12) |
| pdz2_1116 | `runs/pdz2_svae` | `runs/pdz2_w21_k7m` | **done** -- 50 epochs each, basis (441, 16) |

Data: `docs/figures/data/basis_cmp/`. Figures: `docs/figures/cmp/`.

The svae profile encoder is alive on PDZ2, which had to be checked: it died at
epoch 4 on 1118 and a dead encoder makes the comparison meaningless. Effective
rank of the 16-component basis falls 15.05 to 12.02 over training, the relative
step per checkpoint falls 0.56 to 0.001, and the principal angle to the final
basis falls from 88.7 degrees to 0. The hierarchical arm on the same data holds
slightly less: effective rank 12.84 to 11.24.

| | svae | hierarchical |
|---|---|---|
| effective rank, first checkpoint | 15.05 | 12.84 |
| effective rank, last | 12.02 | 11.24 |
| Frobenius norm, last | 17.30 | 19.16 |

So svae keeps about one more effective component than hierarchical with half
the parameters.

### 2. Parameter counts and training throughput

Parameter counts, read from the last checkpoint's state dict, so independent of
where the run happened:

| | hierarchical | svae | svae smaller by |
|---|---|---|---|
| hewl_9b7c | 403,086 | 230,841 | 43% |
| hewl_1118 | 664,928 | 393,809 | 41% |
| pdz2_1116 | 2,298,970 | 1,164,051 | 49% |

**The throughput half of this comparison does not hold yet, and the earlier
numbers for it should not be used.** Seconds per epoch was taken from
checkpoint timestamps with nothing recorded about the allocation, and the
allocations were not the same:

| arm | s/epoch | node | CPUs |
|---|---|---|---|
| pdz2_svae | 61.4 | holygpu8a29104 | 8 |
| pdz2_w21_k7m | 44.5 | holygpu8a22402 | 16 |
| hewl9b7c_svae | 65.2 | unrecorded | unrecorded |
| hewl9b7c_mono_expmean_lp | 66.0 | unrecorded | unrecorded |
| full1000/learned | 75.4 | unrecorded | unrecorded |
| hier1000/softmax_std | 100.1 | unrecorded | unrecorded |

Data loading scales with `-c`, so the PDZ2 gap is 8 CPUs against 16 two months
apart rather than a model difference, and it points the opposite way from the
1118 gap. The four 9b7c and 1118 runs have no recoverable log, so their
hardware is unknown rather than equal. `compare_models.py` now reports node and
CPU count beside the rate, or `hardware unrecorded`, so this cannot recur
silently. Jobs 61854673 and 61854684 measure the PDZ2 pair over 12 epochs each,
chained so they occupy one node in turn.

Data: `docs/figures/data/cmp_mono.csv`, `cmp_1118.csv`, `cmp_pdz2.csv`, built by
`scripts/figures/compare_models.py`.

**Inference time is still not measured.** It needs a predict pass with a clock
around it, on the allocation the training pair is being measured on.

### 3. PDZ2 against `laue.integrate`

| endpoint | arms | state |
|---|---|---|
| CCsym by delay | 6 including `laue` | **done**, `pdz2_ccsym.csv` |
| field-off R-work / R-free | 4 | **done**, `pdz2_refinement.csv` |
| `careless.stats` by shell | 6 | **done**, `careless_stats_pdz2.csv` |
| difference-map peaks, 20 sites | 19 trees have the map | **numbers exist only in job logs** — `wdm_peaks_compare.py` prints and writes no CSV |
| the RB estimator as an arm | `pdz2_w21_k7m` exported | **not scored** — needs `arm_pdz2.sbatch` with `ESTIMATOR=rb` |

Rank on R-free, not CCsym: `laue.integrate` borrows profiles from strong neighbours, which feeds both halves of a half-dataset correlation.

## Left to do

1. **Score the RB arm.** The export works for `pdz2_w21_k7m`; the chain has not been run on it. `split_pdz2.py --estimator rb` is in place, but `arm_pdz2.sbatch` re-predicts into the run directory while the RB parquet is in `rb_preds/`, so the two have to be pointed at each other first.
2. **Difference-map peaks into a table.** `wdm_peaks_compare.py` prints a fixed-width table; it needs a `--csv` so the 20 sites become a long-format file like every other endpoint.
3. **Wilson scale and B for PDZ2.** One reduction away; `plots/wilson_params.csv` is on disk for all 19 arms and nothing reads it.
4. **Inference timing**, for comparison 2. Training throughput is being
   re-measured on a matched allocation by 61854673 and 61854684; inference is
   a separate timed predict pass and is not queued.
5. **Score `pdz2_invN` and `pdz2_geom_shift`.** Both trained to completion and never put through the downstream. `invN` is the interesting one: `kl_per_multiplicity` applies the Wilson prior once per unique reflection instead of once per observation, which is the structural form of the fix the prior-free arm achieves after the fact.
6. **`combo` has no R-free.** The best CCsym arm has never seen the endpoint the dataset is judged on.

## Where the data is

All under `/n/netscratch/hekstra_lab/Lab/laldama`.

| what | path | size |
|---|---|---|
| **hewl_9b7c** training runs | `mono9b7c/` (4 arms), `sim_runs/m9_*` (8 arms) | 9.2G, part of 298G |
| 9b7c DIALS scaling and merging | `dials9b7c/` | 77G |
| 9b7c refinements | `refine9b7c/` | 1.9G |
| **hewl_1118** reference integration | `laue_runs/1118_hewl3/` | 23G |
| 1118 ablation, hierarchical | `ablate/`, `ablate2/` (20 arms) | 14G, 9.5G |
| 1118 1000-frame ablation, svae | `full1000/` (18) | 7.5G |
| 1118 1000-frame ablation, hierarchical | `hier1000/` (8) | 7.0G |
| **pdz2_1116** first processing | `laue_runs/pdz2/` | 1.5G |
| PDZ2 reference arm | `laue_runs/pdz2_ref/` | 42G |
| PDZ2 model arms, trees, refinements | `laue_runs/pdz2_model/` | 191G |
| **dhfr_sim** runs | `sim_runs/sim_*` (10 arms) | part of 298G |
| raw data | `/n/lab_storage/hekstra_lab/people/aldama/integrator_data/` | 3.5G |
| this session's job directory | `jobs/pdz2figs_1118/` | staged scripts and reduced tables |

Inside a PDZ2 arm: `runs/pdz2_<arm>/` is the training run, `tree_<arm>/` is its scored tree, and the careless output is at `tree_<arm>/gain_0,3_from_stills/careless-cdef-0,3-ohp-mlpw/`.

**netscratch is purge-eligible and has already removed this project's raw images once.** Anything that must survive belongs in the repository or in `lab_storage`.

## Standing caveats

- Training is unseeded: two trainings of one configuration differ by about 0.0050 in R-free and 0.86 $\sigma$ in mean peak height, so an $n=1$ gap below roughly 0.008 is noise.
- The careless merging seed moves CCsym by about 0.012 and R-free by about 0.0012, and dominates the phenix seed about ninefold on Laue data. Arms are averaged over merging seeds.
- Polychromatic scores the **last checkpoint only**; one careless scale is tens of GPU-minutes.
- A September checkpoint may no longer load: `geometric_basis_profile` took `smoothness_weight` then and does not now. `rb_predict_config.py` drops such an argument when its value was inert and refuses when it was not.
- **Four checkouts answer to `import integrator` on this cluster**, and which one ran is not implied by a config: holylabs `integrato_refac` (branch `scaling-refinement`, dirty) behind environment `integrator`, lab_storage `software/integrator` behind `integrator-cuda-dev`, `$S/integrator_cur`, and whatever a job directory has staged. `scripts/poly/train.slurm` now records the interpreter, the source directory and its commit in every log, and `SRC=` pins a tree.
- The console scripts in environment `integrator` are **dead**: their shebang is a pre-migration path and they fail with `bad interpreter`. Call `python -m integrator.cli.<tool>` through an environment's absolute interpreter instead. This is why `train.slurm` was unusable and the training it fronts was being launched from one-off sbatch files.
