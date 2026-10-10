# TODO and in-flight — 10 October 2026

Supersedes the earlier list.
Read with `session_state.md` (what is established), `ablation_1118_360.md` (the 20 arms + the measured laue-dials baseline), `best_configs.md`, `experiment_provenance.md` (dataset boundaries).

## Running now

| job | what | reads out as |
|---|---|---|
| **51876783** (array 0-19) | all 20 Kabsch exports through the IDENTICAL chain as the posterior exports -- same careless config 4, same seed, same two-pass phenix, same peak script | R-free + peaks per arm. Below the noise floors (0.0070 / 0.31 sigma) means the export was never the limitation |
| **51876367** | poly SVAE, 50 epochs, first run since the checkpointing fix | whether it finishes, and whether `checkpoints/` is non-empty -- the fix's real test |

## Blocked / needs relaunch

1. **Combination arms.** 51726915: 7 of 21 tasks died on the softplus underflow that was fixed afterwards (`ffbbbd8`). Relaunch `combo.sbatch`; it is the only test of whether the ablation winners compose.
2. **SVAE NaN root cause.** Two late failures, pushed from epoch 37 to 47 by the scale floor, now NaN in the background Gamma from `training_step`. `k_min: 0.01` is set so `k = mu*r + k_min` cannot be the origin; the NaN arrives from upstream in the encoder. Chasing each surfacing site is the merge-casualty pattern again -- use `detect_anomaly` on a short run to catch the FIRST NaN.

## Open questions, in rough priority

3. ~~**The `n_eff` result challenges the unified theory**~~ -- **RESOLVED 10 Oct, and it CONFIRMS the theory.** The claim is per-observation, so median `n_eff` across arms was the wrong statistic: it mixes architectures whose baseline breadth differs by 28x (21 to 588 px) and averages away a within-arm effect. Tested properly, regressing each observation's `n_eff` on `intensity.sum.value` -- the DIALS summation intensity in the metadata, computed with no reference to our model, so not the selection trap that killed the 0.081 slope:
   - rho median **-0.354** (range -0.548 to +0.115); **19 of 20 arms** give broader profiles to weaker spots.
   - Q4/Q1 breadth ratio median **0.79**: strong spots get 21% narrower profiles. Largest swing `nakagami_softplus` at 0.58.
   - **Unplanned internal control:** the one arm without the effect is `q2_profile_fixed` (rho +0.115, ratio 1.00), whose fixed basis is architecturally incapable of adapting to SNR. `q2_profile_physical` is similarly constrained (ratio 0.99). The effect appears wherever the profile can adapt and vanishes where it cannot.
   - Remaining question is the CONSEQUENCE, not the existence: breadth varies ~21% weak-to-strong within an arm, and the Schur correction is `1/(1-rho^2)` with `rho^2 = n_eff/P`. At P = 625 that is a 3-28% variance effect across the measured `n_eff` range -- so quantify what the breadth costs rather than re-confirming it.
4. **Ground-truth shoeboxes from the nanoBragg sim.** `hkl_data/` has 180 frames with per-pixel true h,k,l and wavelength, so the exact pixel set of every reflection is known and an unbiased ORACLE intensity is a background-subtracted sum over it. The laue-dials route failed (CBFs carry no scan; they import as stills), and the better path is to skip indexing and cut shoeboxes from the ground truth directly -- the true geometry is in `Umat`/`Bmat`/`delta_phi`. This is the ONLY way to measure weak-end bias, which is not estimable from real data (see the retired 0.081 slope).
   - `SimulatedShoeboxLoader` and its `SIMULATED_COLS` (with `intensity`/`background` ground truth) were restored in `ff12667`, and `cli/sim_analyze.py` now imports; both were dead before.
   - True cell 33.157 43.968 95.547 90 90 90, 1.0 deg/frame, lambda 1.02-1.18 A (ratio 1.157, so NO harmonics).
   - Structure factors are NOT deposited. Not a limitation for bias (an unbiased reference suffices), but a slope needs the leave-one-out oracle mean as regressor or it is attenuated.
5. **The Friedel work, still untested.** Tying `log(scale)` across mates restores Bijvoet attenuation 0.54 -> 1.06 in simulation; a difference-as-latent (I0, delta) estimator is KL-invariant where the amortized one collapses 1.24 -> 0.32. `centric_anchor_weight` is **dead code** -- never wired into `__init__`/`_step` -- and centrics have I(+) == I(-) by symmetry, so it is a free brake on scale error. The poly SVAE's 4-layer MLP scale over Friedel-discriminating inputs is exactly the configuration the audit named Leak 1.
6. **Report error-model `(a, b)`** from `sigma'^2 = a^2(sigma^2 + (bI)^2)` instead of scatter/sigma. `a` is sigma-miscalibration, `b` the physical floor; scatter/sigma conflates them, and `(a, b)` is what a crystallographic reviewer expects. The `dials_port` package already has the refinement.
7. **Q5 and Q6 of the six-question plan, never run**: a profile encoder shared across datasets, and the export method as a factor (partly answered by 51876783).
8. **Integration quality without ground truth**: held-out pixel predictive likelihood, Pearson residual structure, Wilson-plot linearity, background-vs-border. None needs a merge.

## Settled — do not retry

- **The Gamma likelihood in careless.** The sampling distribution of a background-subtracted intensity is a scaled Skellam: roughly Normal, support on all of R. A Gamma has ~50x too much skewness at I/sigma ~ 1. And more generally: changing careless's likelihood can fix a wrong variance or a wrong shape, never a wrong mean -- every family in `mono.py` is location-scale about `I_obs`.
- **Rubin's rules for passing posterior draws.** They propagate the imputer's uncertainty without calibrating it, so a variance too small by 3-15x stays too small. careless also has no residual-variance parameter, so the "complete-data analysis" the rules need does not exist here.
- **The PyTorch careless fork.** 4h42m vs 2h13m (2.1x SLOWER), F agreement pearson 0.915 with 23% median relative difference, and SigF **uncorrelated** (pearson -0.09) on identical input, flags and seed.
- **`beta = 0`** (variance collapse; `cc_anom` 0.949 is an artifact of DIALS rejecting 26% of observations) and **a learned MLP scale for Laue G(lambda)** (cannot be recovered against the merge gauge).
- **CC-half, cc_anom and the gate-conditioned score as arm-selection metrics.** All reweighting-gameable, and the gate is anti-predictive: the two best-calibrated arms are the two total failures.
- **The 0.081 "shrinkage slope" and the negatives asymmetry.** Both retired 9 Oct. Slope on a neutral leave-one-out reference is 1.149 for us and 0.808 for laue, whose ratio equals the median-intensity ratio to three digits -- the whole gap is the known full-box-total scale difference. `laue.integrate` also emits 0.00% negatives on this data.
- **The cheap Kabsch diagnostics as a quality ranking.** `n_eff`, negative fraction, median I/sigma and `var_rb_cond` all have |rho| < 0.25 against R-free over 20 arms. The profile-variance share reaches rho = -0.60 but does not survive a Bonferroni correction for the 10 correlations tested, and weakens to -0.46 when the two degenerate arms are dropped. They ARE a reliable degeneracy detector (all five flag both failures unmistakably) -- that is a different and weaker claim.
