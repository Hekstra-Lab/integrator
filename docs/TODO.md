# Experiments, comparisons, and where the data is — 18 November 2026

Supersedes the 10 October list.
Read with `methods.md` (how each dataset was processed, and which script drives which tool), `scripts.md` (every script and what it does), `results_by_dataset.md` (what the numbers came out as) and `experiment_provenance.md` (which code produced them).

## Running now

| job | what | reads out as |
|---|---|---|
| **61766712** | PDZ2 svae training, `configs/poly/pdz2_1116_svae.yaml` | the missing half of svae-against-hierarchical on PDZ2; val ELBO was still falling at 42 min |
| **61776542** | RB export of `pdz2_geom_both` | `i_rb`/`var_rb` in the prediction parquet |

## The three comparisons

### 1. Learned basis, svae against hierarchical

The arms must differ by the integrator family **alone**, so the pairs are chosen on that and not on what happens to exist.

| dataset | svae | hierarchical | state |
|---|---|---|---|
| hewl_9b7c | `mono9b7c/hewl9b7c_svae` | `mono9b7c/hewl9b7c_mono_expmean_lp` | **done** — both bases dumped, 10 epochs each, basis (1323, 12) |
| hewl_1118 | `full1000/learned` | `hier1000/softmax_std` | **done** — 4 epochs each, basis (625, 12) |
| pdz2_1116 | `runs/pdz2_svae` | `runs/pdz2_w21_k7m` | svae **training**; hierarchical done, basis (441, 16) |

Data: `docs/figures/data/basis_cmp/`. Figures: `docs/figures/cmp/`.

### 2. Parameter counts and training throughput

| | hierarchical | svae | svae smaller by |
|---|---|---|---|
| hewl_9b7c | 403,086 | 230,841 | 43% |
| hewl_1118 | 664,928 | 393,809 | 41% |
| pdz2_1116 | 2,296,407 | 1,164,048 | 49% |

Seconds per epoch, same hardware within each pair: 9b7c 66.0 against 65.2 (no difference), 1118 100.1 against 75.4 (svae 25% faster). PDZ2 counts are from `construct_integrator`, not from a trained checkpoint, until the svae run finishes.

Data: `docs/figures/data/cmp_mono.csv`, `cmp_1118.csv`, built by `scripts/figures/compare_models.py`.

**Inference time is not measured and is recorded nowhere.** It needs a predict pass with a clock around it on fixed hardware. That is the one piece of this comparison that cannot be recovered from what is on disk.

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
4. **Inference timing**, for comparison 2.
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
