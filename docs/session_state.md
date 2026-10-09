# Where things stand -- October 2026

A map of what is running, what was established, and what is left.
Read with `ablation_1118_360.md` (the ablation), `best_configs.md` (settings), `experiment_provenance.md` (which code produced which numbers) and `TODO.md`.

## Branch

`nanobragg-prior`, which descends from `profile-geometry-conditioning` and fully contains `svae-integrator`.
`add-mkrefl-cli-commands` (110 commits: the scaling/merging package) was merged in as `1f50717`.

**The merge broke twelve things, and every one was silent until something exercised that code path.**
Four were in files that CONFLICTED and were resolved by hand; the other eight were in files that merged CLEANLY and therefore got no attention -- `prepare_priors.py` lost helpers living in non-conflicted regions, and `profile_surrogates.py` never conflicted at all so it silently kept a version without the warm-start capability that is fix #1 for the poly collapse.
Lesson: after a merge like this, run an AST pass for undefined references over every touched file rather than trusting the conflict list.

Fixed so far: `skip_warmstart`, `data_dir`, `n_bins`, `init_stats`, the `optimizer` kwarg, the `pi_weight`/`pbg_weight`/`pprf_weight` aliases, the profile warm-start itself, `_hermite_polynomial`, `_build_hermite_basis_2d/3d`, `_bg_subtract_signal`, `_load_raw_data`, `_try_load_counts_masks`.

Tests: **428 passing**, 3 skipped. The 10 failures and 8 errors are all in `test_scaling`, `test_conjugate_integrator` and `test_refinement`, which arrive with the merge, plus the pre-existing `intensity_kl_bins` assertion. No test that passed before the merge fails after it.

## Established this session

**The sigma model is the defect, measured three independent ways.**
Against `laue.integrate` on the SAME 1,283,813 observations from the same 360 images:

| | ours (kl 0.1) | laue.integrate |
|---|---|---|
| within-HKL CV (real precision) | 0.913 | **0.809** |
| scatter / claimed sigma (1.0 = honest) | **9.62** | **2.47** |
| merged I/sigma achievable | 3.26 | **3.59** |
| merged I/sigma claimed | **27.15** | 8.61 |

Our intensities are only ~11% noisier. The claimed errors are ~4x further from the truth.
careless merges by 1/sigma^2 and phenix weights by 1/sigma^2, so that is where the peak height goes.

By resolution our calibration runs 1.42 (2.0-1.7 A) to 23.5 (13-2.75 A) -- nearly honest where the data is weak, wildly over-confident where it is strong, which is where the anomalous signal lives.
At the highest resolution our intensities are actually BETTER than laue's (CV 0.293 vs 0.403).

**Part of that gap is a type error, not a calibration error.**
A posterior sd `sd(I | counts)` is a different object from a measurement error `sd(I_hat | I_true)`: it is narrower because the prior adds information and because it conditions on the profile and background being known rather than inferred.
careless and phenix read SIGI as a sampling error.
Measured corrections: removing our Wilson prior (the exact Gamma cavity) widens sigma by x1.126 overall, x1.387 at the weak end; a Rao-Blackwellized Fisher sigma is x2.079 the posterior sd.
So roughly half the discrepancy is definitional and about a factor of 2 is real.

**The double-counted prior is NOT the main problem.** Exporting the cavity or the Fisher sigma did not improve peaks or R-free; the measured prior contribution is x1.13 against a x4-5 distortion elsewhere.

**Rubin's rules do not apply** to passing posterior draws downstream: they propagate the imputer's uncertainty without calibrating it, so a variance too small by 3-15x stays too small by the same factor. careless also has no residual-variance parameter, so the "complete-data analysis" the rules need does not exist here.

**The PyTorch careless is rejected**: 4h42m against 2h13m (2.1x SLOWER), F agreement pearson 0.915 with 23% median relative difference, and SigF **uncorrelated** (pearson -0.09) on identical input, flags and seed.

## In flight

| job | what |
|---|---|
| 51628571 | laue-dials baseline through the identical chain -- gives the R-free the documented 360-frame row lacks |
| 51631405 | 7 combination arms: 3x all-winners, 2 more kl_0.1 replicates, kl_0.05, leave-one-out without coord channels |
| 51636164 | Rao-Blackwellized sigma export for kl_0.1, to test the "wrong object" diagnosis |
| 51639211 | profile basis + background prior for the poly SVAE |

The RB export carries a falsifiable prediction: the weak shell should land near 1.0 and the correction should be NON-UNIFORM across shells.
A uniform rescaling would mean the sigma is not the problem and no export fixes it.

## Left to do

1. **Launch the poly SVAE.** `configs/poly/hewl1118_svae_merge.yaml` builds (HierarchicalSVAEIntegrator, 2.638M params). Waiting only on the two prior files. Check before training that the basis is genuinely peaked and the background prior is near Laue's ~12 counts/pixel, not the 1.0 default -- those two numbers are what the documented collapse turns on.
2. **Report error-model (a, b)** from `sigma'^2 = a^2(sigma^2 + (bI)^2)` instead of scatter/sigma. `a` is sigma-miscalibration, `b` the physical floor; scatter/sigma conflates them. The `dials_port` package already has the refinement.
3. **The Friedel work, untested.** Tying `log(scale)` across mates restores Bijvoet attenuation from 0.54 to 1.06 in simulation; a difference-as-latent (I0, delta) estimator is KL-invariant where the amortized one collapses 1.24 -> 0.32. `centric_anchor_weight` is dead code -- never wired into `__init__`/`_step` -- and centrics have I(+) == I(-) by symmetry, so it is a free brake on scale error. The `kl_0.1` ablation result is this mechanism showing up on real data.
4. **Integration-quality metrics with no ground truth** -- see TODO.md. Held-out pixel predictive likelihood and Pearson residual structure; neither needs a merge.
5. **The simulated data.** The nanoBragg tree was purged; `laue_dials_simdhfr_data_deposition.tar.gz` (163 GB) is extracting to `/n/holylabs/.../data/simdhfr`. Ground truth would let every metric above be validated rather than argued.
6. **Three config bugs** found by the factory scan: 8 configs use the removed `pi_weight`/`pbg_weight`/`pprf_weight` loss args (including `poly_3enc`/`poly_5enc`, which the notes call the working poly model), and `hierarchical_auto.yaml` puts `lr` in the wrong section.

## Things that are settled and should not be retried

- The Gamma likelihood in careless. The sampling distribution of a background-subtracted intensity is a scaled Skellam -- roughly Normal, support on all of R. A Gamma has ~50x too much skewness at I/sigma ~ 1.
- `beta = 0`: variance collapse; `cc_anom` 0.949 is an artifact of DIALS rejecting 26% of observations.
- A learned MLP scale for Laue: cannot recover G(lambda) against the merge gauge, stays near-constant.
- CC-half, cc_anom and the gate-conditioned score as arm-selection metrics: all reweighting-gameable, and the gate is anti-predictive.
