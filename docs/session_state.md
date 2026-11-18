# Where things stand -- October 2026

**For what is running, which comparisons are done and where the data
lives on the cluster, read `TODO.md`, which is current.** This file is
the October narrative: the merge, what it broke, and what was
established then.

A map of what is running, what was established, and what is left.
Read with `ablation_1118_1000.md` (the ablation), `exporting.md` (how intensities leave the model, and the defects found there),
`merging_svae.md` (the mono and polychromatic merging model), `pitfalls.md` (failures that produced plausible wrong numbers),
`best_configs.md` (settings), `model_selection.md` (best config per beam type, the dataset inventory, and what to run on new data),
`joint_model_closed_forms.md` (untested theory: which parts of the pixel-to-structure-factor model are conjugate, and the collapsed scale objective),
`experiment_provenance.md` (which code produced which numbers) and `TODO.md`.

## Branch

`nanobragg-prior`, which descends from `profile-geometry-conditioning` and fully contains `svae-integrator`.
`add-mkrefl-cli-commands` (110 commits: the scaling/merging package) was merged in as `1f50717`.

**The merge broke twelve things, and every one was silent until something exercised that code path.**
Four were in files that CONFLICTED and were resolved by hand; the other eight were in files that merged CLEANLY and therefore got no attention -- `prepare_priors.py` lost helpers living in non-conflicted regions, and `profile_surrogates.py` never conflicted at all so it silently kept a version without the warm-start capability that is fix #1 for the poly collapse.
Lesson: after a merge like this, run an AST pass for undefined references over every touched file rather than trusting the conflict list.

Fixed so far: `skip_warmstart`, `data_dir`, `n_bins`, `init_stats`, the `optimizer` kwarg, the `pi_weight`/`pbg_weight`/`pprf_weight` aliases, the profile warm-start itself, `_hermite_polynomial`, `_build_hermite_basis_2d/3d`, `_bg_subtract_signal`, `_load_raw_data`, `_try_load_counts_masks`.

Tests: **428 passing**, 3 skipped. The 10 failures and 8 errors are all in `test_scaling`, `test_conjugate_integrator` and `test_refinement`, which arrive with the merge, plus the pre-existing `intensity_kl_bins` assertion. No test that passed before the merge fails after it.

## Established earlier (the sigma-model phase)

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

## Established 15 October

**The merging deficit was an export defect.**
`SigF` was computed by a float32 subtraction that cancels twice over, which inflated it tenfold on the brightest reflections and deleted 4,500 rows outright.
With it fixed the merging port matches the June reference at every epoch.
Full account in [`exporting.md`](exporting.md); the four fixes are commits `c7a9469`, `0e8b3d7`, `4c7c878` and `ffff903`.

**Refinement is on phenix 2.0**, through `reference_data/laue_refine_pass{1,2}.eff` and `integrator.utils.phenix`.
The version itself is worth 0.0005 in $R_\mathrm{free}$ on a fixed chain, so the 1.20 tables remain comparable.

**Large Wilson $G$/$B$ gives the best $R_\mathrm{free}$ and the worst peak height**, 0.1615 against the SVAE's 0.1718 at two thirds of the peak mean.
See [`merging_svae.md`](merging_svae.md).

**The hierarchical integrator trails the SVAE on $R_\mathrm{free}$ in seven of eight arms**, and `fixed` ties `laue.integrate` on peaks in both families.
See [`ablation_1118_1000.md`](ablation_1118_1000.md), which also now carries the profile statistics of the same checkpoints.

**Polychromatic merging is still poor**, peak mean about 3 against careless's 8.3, and F agrees with I, so it is the merge and not the export.

**The spectrum was modelled twice**, once in the prior and once in the scale, and the two fought: the merging model's prior $G(\lambda)$ correlates $-0.716$ with the observed beam while its scale correlates $+0.78$.
Five arms tested it.
Removing $\lambda$ from the scale is catastrophic ($R_\mathrm{free}$ 0.4008), taking it out of the prior is worth about 0.009, and evaluating the prior per observation is equivalent to removing it -- but a peak mean of 3.3 against careless's 8.3 means the remaining gap is upstream of the merge.
See [`merging_svae.md`](merging_svae.md).

**The Laue chain generated its own R-free set per MTZ**, and `refinement.main.random_seed` moved that draw, so a seed spread measured there was partly the test set moving.
The chain now reads the 9b7c chain's `rfree.mtz` through `refine_fix.sh` / `refine_fix20.sh` and `reference_data/laue_refine_pass{1,2}_fixedfree.eff`, which puts every number in the project on one test set.
See [`exporting.md`](exporting.md).

## In flight

**Nothing is queued or running as of 5 November 2026.**
The last batch was the structured-versus-amortized replicate sweep: six
training draws plus five rescores, all COMPLETED, reported in
[`ablation_1118_1000.md`](ablation_1118_1000.md).
Also complete: the 180-shot nanoBragg oracle
(`$S/oracle/oracle_hkl_data_180.npz`, 869,147 observations, 191,235 unique
`hkl`) and the controlled family timing run.
### The oracle $\sigma_e$ comparison is BUILT, and it answers the project's main question

`$S/sim_sigma_e.py`, run on 153,280 reflections that all three arms share, inside a
wavelength and ring-count domain fixed by `spectrum_cut.py` from the oracle alone.
Each arm gets one global multiplicative scale by flux conservation and nothing else.
Error against the exact footprint sum, in counts, as $\sqrt{\mathrm{rms}^2 - \sigma_\mathrm{ref}^2}$:

| arm | rms, $r<0.05$ | own error, $r<0.05$ | own error, $r<0.004$ | slope | pedestal | own/claimed |
|---|---|---|---|---|---|---|
| `laue.integrate` | 212.6 | 178.5 | 222.7 | 0.880 | $+45.1$ | 2.38 |
| structured (`svae`) | **151.7** | **98.3** | 254.9 | 1.083 | $-31.3$ | 4.99 |
| amortized (`hierarchical`) | 576.7 | 565.0 | 342.5 | 0.654 | $+129.9$ | 10.00 |

Those are against the POOLED background (`--bg pool`, oracle `oracle_pool_180.npz`);
the ring-background run of the same file is the control and reproduces the earlier
numbers byte for byte.

**The amortized arm loses to both, at either rung, by a factor of 3 to 6.**
Its defect is prior shrinkage, not footprint: slope $0.65$ and pedestal $+131$ counts
barely move between rungs, and its error GROWS as the target widens.
This is the first measurement that separates the two families, and it separates them
decisively where $R_\mathrm{free}$ tied at $0.93\sigma$.

**Structured versus `laue.integrate` depends on which footprint is the target, and the
full footprint is the right one.**
On $r<0.05$, the unbiased footprint sum, structured wins: own error 94.8 against 170.7,
rms 183.7 against 232.1, CC 0.9936 against 0.9888.
On $r<0.004$ it loses, 256.3 against 221.9.
The $r<0.004$ rung was chosen to make the ORACLE precise (S/N 163 against 83), which is a
different goal from being a neutral target: it discards 29.5% of the signal, so an
integrator that captures the broad tail is scored as over-estimating.
That is the quantity under test, so CLAUDE.md's own rule disqualifies it as the ranking
domain.
The mechanism checks out -- structured's slope moves from 1.129 to 1.077 as the target
widens toward the full footprint, while `laue.integrate`'s moves the other way,
0.915 to 0.874, which is what an integrator that captures the core and misses the tail
must do.

**Every arm badly understates its error, and ours are worse than `laue.integrate`'s.**
own/claimed is 2.27 for `laue.integrate`, 4.81 for structured, 9.90 for amortized.
The intensities are good and the error bars are not, which is
[`posterior_exported_as_likelihood`](pitfalls.md) measured against truth for the first time.

**The pooled background sharpened the instrument and, unexpectedly, validated it.**
Replacing the oracle's per-reflection ring with the dataset-wide pooled model
$B(\rho, \phi) = R(\rho)\,A(\rho, \phi)$ cuts the reference's own sd from 157.4 to 115.5
counts at $r<0.05$ and widens the rms gap between structured and `laue.integrate` from
$1.26\times$ to $1.40\times$, which is what it was built for.
The validation is the `own error` column: it moved by only 4% (170.7 to 178.5, 94.8 to
98.3, 560.8 to 565.0) while the reference noise fell by $1.36\times$.
Those are two independent routes to the same quantity -- a large rms minus a large known
reference variance, against a smaller rms minus a smaller one -- so their agreement means
the shared-photon covariance I had flagged as making $\sqrt{\mathrm{rms}^2 -
\sigma_\mathrm{ref}^2}$ only a LOWER bound is small, and the bound is nearly tight.
Slopes and pedestals move under 1%, confirming they were already the trustworthy entries.
At $r<0.004$ nothing changes at all (every arm's rms within 1%), which is the right null:
the median core footprint is 79 px, so background error is not what limits that rung.
The one systematic is a common $-2$ count shift in every pedestal, because the pooled
model sits about 0.01 counts/px below the ring under a footprint; it is far below the
pedestals themselves and cancels between arms.

Built: `$S/bg_accumulate.py` + `bg_array.sbatch` (per-shot accumulation on a 72-sector,
1-px-radial grid), `bg_build.py` (aggregation, smoothing width chosen on held-out shots),
`bg_explore.py`, `bg_variance.py`, `bg_footprint_check.py`, `bg_validate.py`.
Validated four ways: shot-independence ($z$ sd 0.78), no Bragg leakage ($+0.005$% at
fixed $(\rho,\phi)$), unbiasedness UNDER a footprint at $0.04 < r < 0.05$ where the
rocking curve has died ($+0.0074 \pm 0.0068$ counts/px pooled against $+0.0036 \pm
0.0068$ ring, both zero, which rules out a local thermal-diffuse component the ring would
capture and pooling would miss), and end to end on 6,009 null patches -- real footprints
rotated about the beam centre to an azimuth where every pixel is unassigned, so true
$I = 0$ -- where the ring makes 125.7 counts rms against the pooled model's 85.8.
A radial-only background would have been $10.8\times$ worse than the full model and so
worse than the ring, which is what the standing warning against radial binning recorded.

**The counts are strongly sub-Poisson, and the oracle's $\sigma$ convention is wrong.**
`bg_variance.py` measures a per-pixel variance of about 10.3 ADU$^2$ while the bin mean
runs 11.7--31.2 (free fit $\mathrm{var} = 0.144\,\mathrm{mean} + 7.8$), so every `ref sd`
in the tables above is roughly $1.28\times$ too large.
Intensities are untouched, and because `own error` proved insensitive to the reference
variance the ranking survives: correcting it moves the structured-over-`laue.integrate`
own-error ratio from $1.82\times$ to about $1.58\times$, and the measured rms ratio
$1.40\times$ needs no variance model at all.
What it does raise is a model question, untested: our pixel likelihood is Poisson, so on
this data it is mis-specified by about $1.7\times$ in variance, an over-dispersed
likelihood is less informative, and the prior therefore dominates more -- which is
exactly the shrinkage both arms show.

Caveats that belong with the numbers: one training draw per arm, and training is
unseeded, so the structured-versus-`laue.integrate` gap needs replicating before it is
a family result (`laue.integrate` is deterministic, so only our side varies);
the weak end is still bad for everyone, with structured running $-15$% to $-41$% across
the three middle brightness bins and `laue.integrate` $+8$% to $+133$%;
and both arms were handed the simulation's exact geometry, which real data does not
provide.

The table below is the older merging work and is kept for its predictions.

| job | what |
|---|---|
| 53588133 | `polymerge_gateem2`: `pmmono` retrained with `gate_em_iters: 20`, i.e. with the split solved rather than amortized, scoring epochs 9/24/49 on `rfree.mtz` with F and I both. Then refit the physical scale from its dump (`merge_terms_gateem2.pt`), which is the first dump whose counts the gate has not distorted |

Three predictions ride on it: the refitted spectrum should fall from
72$\times$ toward 18.3$\times$; the counts-vs-integrator ratio should flatten
from 0.45--1.49 toward 1; and $R_\mathrm{free}$ should improve while peaks
probably do not, since gate error is common-mode across Friedel mates.

A fourth, measured while it trained, cuts against the first: **the free
per-observation background is a larger sink for scale error than the gate
ever was.** With a known per-observation tilt of 0.5 in the truth and a flat
model scale, a conditional fit recovers 0.011 of it from the signal counts as
they were dumped, 0.162 once the background pedestal is removed, and **0.544
with the background held at its true value** -- and that does not improve with
bigger shoeboxes (0.31 at 100 pixels, 0.32 at 1323). careless never faces it
because DIALS pins the background from the shoebox's background region before
any scaling. So the solve should fix the merged intensities without recovering
the scale, and if the spectrum stays inflated the next change is to the
background, not the scale. See [`merging_svae.md`](merging_svae.md).

The 72-refinement rescoring, the background comparison and the pinned-scale
test are complete; see [`ablation_1118_1000.md`](ablation_1118_1000.md) and
[`merging_svae.md`](merging_svae.md).

Landed:

- **Fixing the R-free set collapses the `phenix`-seed spread from sd 0.0094 to 0.0001**, and a freshly drawn set is about 0.021 optimistic because `shaken_9b7c.pdb` was refined with `rfree.mtz` held out -- its flags agree with `refine_001.mtz` on all 50,106 reflections. At fixed flags phenix 2.0 still differs from 1.20 by $-0.0057$ on weak data, so the version is not free after all.
- **careless carries the spectrum in its per-observation scale and nowhere else.** Joined at 99.6% of 2,664,858 rows, wavelength explains **50.7%** of the variance of its $\log$ `Scale` against $d$ 7.7%, $x$ 7.6%, $y$ 2.6%, image 1.6%, and the curve correlates +0.945 with the observed one.
- **The scale, not the spectrum's shape, is what fails.** Our per-observation scale disagrees with careless's by 89.5% rms, 39.8% after removing a smooth trend in $d$ (a gauge), where a 2--4% Bijvoet signal on stills needs $\lesssim 4\%$. Removing a smooth trend in $\lambda$ changes almost nothing, so it is not a wrong spectral curve.
- **The exported $\sigma$ is over-confident by a dispersion term**, 10$\times$ overall and 33$\times$ in the brightest decile at a fitted slope of 1.11 against $I/\sigma$. Built: `merge_error_model`.
- **Tying Friedel mates' scales would be wrong on stills**; conditioning on wavelength strata is the correct replacement, and 95.6% of strata hold both mates. Built: `scripts/stratified_bijvoet.py`.
- **The scale is the defect, and the experiment is decisive.** Swapping *only* the per-observation scale on the `pmmono` epoch-24 checkpoint: ours (learned, ~6k params) 0.2017 / 3.35; careless's smoothed (23p) 0.1872 / 2.98; **careless's full (~10k) 0.1786 / 6.17**; careless end to end 0.1780 / 8.64. That closes 98% of the $R_\mathrm{free}$ gap and 53% of the peak gap without retraining anything else. Built: `scale_from_metadata`.
- **Ten physics parameters match a thousand generic ones.** Fitting careless's `Scale` as a regression target, held out by reflection: constant 168.5%, $G(\lambda)_6$ 103.8%, **$G(\lambda)_6$ + polarization 58.5% (10 params)**, + per-image 54.0% (1015p), everything at degree 12 45.5% (1033p).
- **Five things are wrong with our scale**, in order: the spectrum is unidentifiable because on a still $\lambda = 2d\sin\theta(r)$, so a free function of $(r, d)$ absorbs 73% of a realistic $G(\lambda)$; polarization is absent from every arm (2.14$\times$ azimuthal swing at 1.5 Å); absorption is absent *and* unrepresentable, since no arm carries a crystal-frame coordinate; the scale is deterministic where careless's is sampled; and capacity sits where it does harm ($r$, $d$) and not where it is needed. Built: `scripts/fit_physical_scale.py`.
- **The responsibility gate is the exact E-step only by coincidence**, and `gate_em_iters` now solves the fixed point instead. The first attempt collapsed an arm to 0.3926 through three defects in nine lines -- the background's units, an iteration ~1.5% per step, and an acceleration taken in the wrong space -- plus freezing $b_i$ while solving for $I_h$. All four are in `merging_svae.md`; the lesson is in `pitfalls.md`.
- **A large Wilson $G$/$B$ does nothing for `svae_merging`.** Learnable and pinned both land on the unmodified SVAE (0.1717/14.36 and 0.1718/14.38 at epoch 44 against 0.1718/15.08), and the learnable $G$ moved 0.2% in 37 epochs: $G$/$B$ is a gauge direction, so the conjugate runs' better $R_\mathrm{free}$ belongs to their architecture, not to their initialisation.

## Development on simulated data

`scripts/simulate_laue_merging.py` writes a small Laue stills dataset in the
real loader's format with the truth kept alongside, and
`scripts/score_simulated_merge.py` scores an arm against it.
50k observations on 250 stills, generated in 1.8 s, 6 epochs in under three
minutes on a laptop.

The motivation is today: three GPU runs of 65--70 minutes to establish that
solving the responsibility split makes the model worse, with the verdict read
through careless, phenix, an R-free flag set and a merging-seed average -- four
stages that each move the number by as much as the effect under test, and none
of which can say *which* part of the model was wrong.
The scorer reports instead the scale error after its gauge, the recovery of
each physical term, the merged intensities' correlation and slope, and the
Bijvoet attenuation.

Calibrated against hewl_1118 where the facts are known: 96.6% background
share, a 16.4$\times$ spectrum, a $(d, r)$ surface absorbing 62.6% of it
against the measured 73%, median 24 observations per Friedel-separate
reflection, and a Friedel pair essentially never excited at the same
wavelength.
Two mismatches with our own model are deliberate, because a simulator drawing
from the model's assumptions flatters it: absorption is a crystal-frame
spherical-harmonic surface, which a lab-frame $(d, r)$ surface absorbs 0.2% of,
and the background is a tilted plane where the merge infers a constant.

It is **easier than reality** on everything not in that list -- a fixed
Gaussian profile, no mosaicity, divergence or partiality, exact centroids,
random rather than structural anomalous differences -- so it is for *localising*
a defect, not for predicting $R_\mathrm{free}$.
Four epochs already give $\epsilon = 11.7\%$ against the $\lesssim 4\%$ a
Bijvoet signal needs, and localise it by term: the spectrum comes back at
CC $+0.983$ and slope 1.06, polarisation at CC $-0.130$, absorption at
$+0.157$, and decay and the per-image gray at essentially zero.

Note what the polarisation row is and is not.  This arm is `LaueMLPScale`,
which has **no polarisation term at all**, so $-0.130$ is not the inverted sign
at `chebyshev_scale.py:762` -- that code is in `LaueSpectralScale` and is never
reached here.  It is the MLP's generic capacity picking up something that runs
the wrong way, which is a worse situation than the term simply being absent.

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
