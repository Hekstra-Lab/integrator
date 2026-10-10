# Ablation: HEWL 1118, 360 frames, October 2026

Twenty arms, one knob each, on the 360-frame set (1,283,813 observations, 13,829 unique ASU reflections, 25x25 2D shoeboxes).
Twenty epochs each -- against 40-50 for production -- so absolute values sit below the documented references and only the differences between arms mean anything.

Every arm went through the identical chain: predict -> careless config 4 (dmin 1.5, seed 2024) -> merging stats -> two-pass phenix -> `peak_heights.csv`.

## The baseline, and the noise floor it accidentally gave us

The common baseline is `configs/poly/hewl1118_poly_ref.yaml`: `transform: anscombe`, `reparameterization: shape_rate`, `positive_constraint: exp`, `intensity_kl_weight: 1.0`, `learned_basis_profile`.

**Four arms varied a knob to the value the baseline already had**, so they are four independent runs of one configuration.
There is no seeding in the training CLI, so they are genuine replicates:

| arm | R-work | R-free | IOD 204 | IOD 206 |
|---|---|---|---|---|
| q1_transform_anscombe | 0.1708 | 0.2196 | 12.46 | 10.36 |
| q2_profile_learned | 0.1789 | 0.2312 | 13.03 | 10.64 |
| q3_gamma_shape_rate_exp | 0.1811 | 0.2310 | 12.76 | 11.12 |
| q4_kl_1p0 | 0.1832 | 0.2362 | 13.18 | 11.66 |

**Noise floor:** R-work sd 0.0054 · R-free sd **0.0070** · IOD 204 sd **0.31** · IOD 206 sd 0.57 · IOD 205 sd 0.27 · MET 105 sd 0.20 · CYS 64 sd 0.26 · CYS 30 sd 0.28.

Nothing below 2 sigma of those should be read as an effect.
This was not planned; it is the single most useful number the ablation produced, because it retires several comparisons made earlier in the project.

## Results

Sorted by R-free. Full per-site values are in the session transcript; the headline sites are here.

| arm | R-work | R-free | IOD 204 | IOD 206 | IOD 205 |
|---|---|---|---|---|---|
| **q4_kl_0p1** | **0.1578** | **0.2024** | **15.84** | **14.89** | **8.36** |
| q3_gamma_nakagami_exp | 0.1685 | 0.2154 | 13.85 | 11.28 | 7.00 |
| q4_kl_0p5 | 0.1655 | 0.2163 | 13.89 | 12.59 | 7.18 |
| q3_gamma_mean_fano_exp | 0.1721 | 0.2183 | 13.01 | 11.80 | 6.04 |
| q2_profile_shapelet | 0.1724 | 0.2188 | 12.75 | 10.22 | 6.28 |
| q1_transform_anscombe | 0.1708 | 0.2196 | 12.46 | 10.36 | 6.00 |
| q1_transform_standardization | 0.1743 | 0.2214 | 12.41 | 10.82 | 6.22 |
| q2_profile_geomboth | 0.1701 | 0.2221 | 13.32 | 12.39 | 6.32 |
| q3_gamma_nakagami_softplus | 0.1797 | 0.2261 | 12.37 | 11.19 | 5.46 |
| q2b_coordchan | 0.1793 | 0.2280 | 13.92 | 11.68 | 7.64 |
| q3_gamma_mean_fano_softplus | 0.1813 | 0.2286 | 13.04 | 11.33 | 6.12 |
| q2b_coordchan_geomboth | 0.1763 | 0.2293 | 11.78 | 10.63 | 5.63 |
| q2_profile_physical | 0.1786 | 0.2296 | 12.46 | 10.57 | 6.07 |
| q3_gamma_shape_rate_exp | 0.1811 | 0.2310 | 12.76 | 11.12 | 6.23 |
| q2_profile_learned | 0.1789 | 0.2312 | 13.03 | 10.64 | 5.59 |
| q2_profile_fixed | 0.1830 | 0.2348 | 12.48 | 11.15 | 6.16 |
| q4_kl_1p0 | 0.1832 | 0.2362 | 13.18 | 11.66 | 6.02 |
| q4_kl_0p25 | 0.1855 | 0.2388 | 7.62 | 9.12 | 4.73 |
| q3_gamma_shape_rate_softplus | 0.3048 | 0.3997 | 0.82 | -0.38 | 1.96 |
| q1_transform_log1p | 0.2973 | 0.4016 | 1.06 | -0.08 | 2.09 |

## What actually cleared the noise floor

| effect | R-free | IOD 204 | |
|---|---|---|---|
| `intensity_kl_weight: 0.1` | +3.9 sigma | **+9.4 sigma** | the one large effect |
| `intensity_kl_weight: 0.5` | +1.9 | +3.3 | same knob |
| `nakagami` + `exp` | +2.0 | +3.1 | real but modest |
| `profile_coord_channels` | +0.2 | +3.4 | peaks only |

**Everything else is inside run-to-run variation**, including the entire profile-family comparison and anscombe vs standardization.
An earlier reading of this table called geom_both a clear winner; at 1.1 and 1.5 sigma it is not distinguishable.

`kl_0.1` lifts **every site**, not just the iodides -- IOD 205 +8.9 sigma, MET 105 +6.5 sigma, CYS 30 +4.5 sigma.
A uniform lift across strong and weak scatterers is what signal recovery looks like; a reweighting artifact would favour some sites.

## Two catastrophic settings

`transform: log1p` and `shape_rate` + `softplus` both give R-free ~0.40 and peaks at or below 1 sigma, with NEGATIVE peaks at most sites.
Both had the BEST sigma calibration in the whole ablation (Var(z) 28.1 and 15.7 against a 1.0 ideal).
A model can be perfectly honest about its uncertainty and measure nothing.

`q4_kl_0p25` is out of line with its neighbours (IOD 204 7.62 between 15.84 at kl 0.1 and 13.89 at kl 0.5), depressed roughly uniformly across sites while R-work is only slightly worse.
Treat it as a suspected bad run until a replicate says otherwise.

## Against the classical pipeline

`laue.integrate` on the same 360 frames (documented): IOD 204 **15.7**, IOD 206 **15.7**.
Our best arm: 15.84 / 14.89 -- a tie on one iodide and about -1.4 sigma on the other.

That matches the documented pattern: the model's edge is a high-multiplicity phenomenon, and at 360 frames the margin narrows rather than widens (at 1440 frames we tie, 21.1 vs 20.5).
R-free for the laue arm was never measured at 360 frames; that run is in flight.

## Metrics that did NOT work

A gate-conditioned score was built to triage arms cheaply: CC-half between two half-merges, read only where `Var(z)` is near 1.
**It is anti-predictive.** The two best-calibrated arms are the two total failures, and the worst-calibrated arm is the winner.
CC-half from careless behaves the same way -- the winner has among the lowest.
Of everything computed, only R-free and peak height tracked each other.

## Configs

Generated by `.scratch_ablate/gen_ablation.py` into `configs/ablate/` (committed), and the follow-on combination arms by `.scratch_ablate/gen_combo.py` into `configs/combo/`.
Both filter surrogate args against each class's signature, because the baseline's `latent_dim` / `prior_scale` / `smoothness_weight` are written for the learned basis and the fixed and shapelet surrogates reject them.

## Head-to-head against laue-dials -- the baseline, measured (October 2026)

The 360-frame laue-dials row was previously taken from documented iodide peaks (15.7 / 15.7) with no R-factors.
Job 51628571 put `laue.integrate` through the IDENTICAL chain as every arm: careless config 4 (dmin 1.5, seed 2024), two-pass phenix, the same peak script, same reference model.

| | laue-dials | ours (`kl_0.1`) | delta |
|---|---|---|---|
| **R-free** | 0.2061 | **0.2024** | **-0.0037** (0.5 sigma of the 0.0070 floor -- ours ahead) |
| R-work | 0.1570 | 0.1578 | +0.0008 (0.1 sigma -- even) |
| mean peak over 13 sites | 7.327 | **7.748** | **+0.421**, paired t = 1.66 on 12 df, p ~ 0.12 |

Per site (sigma is the replicate floor, 0.26 where not separately measured):

| site | laue | ours | delta | sigma |
|---|---|---|---|---|
| IOD 205 | 6.325 | 8.360 | +2.035 | **+7.5** |
| CYS 64 | 5.367 | 6.800 | +1.433 | **+5.5** |
| CYS 94 | 6.226 | 7.483 | +1.257 | **+4.8** |
| CYS 30 | 6.325 | 7.428 | +1.103 | **+3.9** |
| MET 12 | 4.454 | 5.411 | +0.957 | **+3.7** |
| CYS 80 | 6.141 | 6.761 | +0.620 | +2.4 |
| CYS 6 | 4.969 | 5.491 | +0.522 | +2.0 |
| IOD 206 | 15.135 | 14.889 | -0.246 | -0.4 |
| IOD 204 | 15.979 | 15.838 | -0.141 | -0.5 |
| CYS 115 | 5.712 | 5.479 | -0.233 | -0.9 |
| CYS 127 | 4.757 | 4.520 | -0.237 | -0.9 |
| CYS 76 | 5.498 | 5.073 | -0.425 | -1.6 |
| MET 105 | 8.364 | 7.196 | -1.168 | **-5.8** |

**We do not trail laue-dials at 360 frames. It is a tie, nominally ours on both primary metrics**, and our arms ran 20 epochs against 40-50 for production.
Seven wins, six losses; the wins are larger, the strong iodides are even, and MET 105 is the one real loss.

This retires the premise behind a long line of reasoning in this project.
Every earlier comparison showing laue ahead used its documented two-iodide numbers against one of our arms; this is the first matched measurement of all 13 sites plus R-factors.

**What still holds.** The paired per-observation comparison (same 1,283,813 observations) gives within-HKL CV 0.913 for us against 0.809 for laue, and scatter/sigma **9.62** against **2.47**.
Our sigma IS badly miscalibrated.
But a uniform miscalibration is free in the merge: with weights `w_i = 1/sigma_i^2` and true variances `v_i`, efficiency is `(sum k/v)^2 / [sum(1/v) * sum(k^2/v)]` for `k_i = v_i/sigma_i^2`, which by Cauchy-Schwarz equals 1 **iff `k` is constant**, whatever its value.
Our spread of `k` across shells (1.42-23.5) is comparable to laue's (0.66-13.0).
So the two results are consistent rather than contradictory: the sigma is wrong and it costs little here.
Report it as a calibration defect, not a precision defect.

The one mechanism that does NOT average away with multiplicity is the positivity bias -- 0% negative intensities against laue's real fraction, and the slope 0.081 in the weak stratum.
A bias survives averaging where a variance error does not.
That remains unmeasured against ground truth and is what the simulated data is for.

## Retired: the 0.081 "shrinkage slope" and the negatives asymmetry

Both were load-bearing in a long line of reasoning about an export bias.
Neither survives a check with a neutral stratifier, and the original script that produced 0.081 is in neither the repo nor the cluster, so it cannot be reconstructed.

**The recomputation.** The two MTZs correspond row-for-row (verified 100.00% on hkl + xcal + ycal over all 1,283,813 observations), so the pairing is exact.
Regressor: the leave-one-out mean of LAUE's intensities over the other observations of the same HKL -- independent of each observation's own noise, and identical for both arms so any attenuation applies equally.
Control: the same regression on laue itself, which is a classical unbiased profile fit and therefore sets the attenuation floor.

| | ours | laue |
|---|---|---|
| slope on the LOO reference, all data | **1.149** | 0.808 |
| median I | 30.66 | 21.49 |

Slope ratio 1.422, median-intensity ratio 1.427 -- agreeing to three digits.
**The whole slope gap is the known overall scale difference** (our I is the full-box total, documented at 1.3-1.5x laue's; see [[hewl1118_weak_spot_profile_plateau]]).
Scaled, both arms respond to signal with slope about 1.
There is no 12x shrinkage.

**Why the original number was not estimable.** The errors-in-variables correction `Cov(I,ref) / (Var(ref) - sigma_ref^2)` diverges to ~1e12 in the two weakest strata, because inside a narrow weak stratum there is no real signal variance in the regressor.
The raw weakest-eighth slope is **-1.872**, a negative number that is noise.
Any slope computed on a stratum selected by the reference's own noisy values is attenuated even for a perfectly unbiased estimator (Fuller 1987), and that attenuation is ALSO flat in multiplicity -- which was the second piece of evidence being read as proof of bias.

**The negatives asymmetry is false on this data.** `laue.integrate` exports **0.00%** negative intensities here, the same as ours.
The negative fraction cannot be used as a bias signature against this comparator.

**What the data does show, with the opposite sign.** The direct slope ours-on-laue is 1.330 against a median per-observation ratio of 1.680.
Slope below median ratio means weak reflections carry a larger ours/laue ratio than strong ones, so our weak intensities are relatively **inflated, not shrunk** -- the direction a floor at zero predicts, and the opposite of the retired claim.
The magnitude is not measurable from this data.

**Consequence.** The weak-end bias is not estimable without ground truth.
The nanoBragg simulation is not a nice-to-have for this question; it is the only way to answer it.

## The export matters, and it reshuffles the ranking (job 51876783, 10 Oct)

Every arm above was scored on its POSTERIOR export (`E_q[I]`, `sd_q[I]`).
This runs the MC Kabsch export -- unbiased profile fit, negatives allowed, Schur-complement Fisher sigma -- from the SAME checkpoints through a byte-identical chain: same careless config 4, same seed 2024, same two-pass phenix, same peak script.
Only the two MTZ columns differ.

### R-free improves substantially

| | |
|---|---|
| arms improving | **16 / 20** (13 by more than 1 sigma) |
| mean delta R-free, excluding the 2 mostly-noise arms | **-0.0185**, paired t = **-3.89** |
| best Kabsch R-free | **0.1799** (`mean_fano_softplus`) |
| vs the measured laue-dials baseline 0.2061 | **+3.7 sigma better than the classical pipeline** |
| arms beating laue-dials on R-free | **7/20 Kabsch, 1/20 posterior** |

### Peaks are mixed, and the best arm changes

Mean IOD 204 delta is -0.78 (-2.5 sigma) over all 20, but that is dominated by the two degenerate arms.
The best arm by mean peak height over 13 sites:

| | R-free | mean-13 peaks |
|---|---|---|
| `shapelet` + Kabsch | 0.2027 | **7.80** |
| `kl_0.1` + posterior (the old best) | 0.2024 | 7.748 |
| laue-dials baseline | 0.2061 | 7.327 |
| `mean_fano_softplus` + Kabsch | **0.1799** | 6.27 |

`shapelet` + Kabsch beats laue-dials on **both** metrics at once (both gaps under 2 sigma, so a tie nominally in our favour on both simultaneously -- stronger than the single-metric tie of 9 Oct).

### Three things this changes

1. **The export choice reshuffles the ranking.** `mean_fano_softplus` goes 11th -> 1st on R-free; `kl_0.1` goes 1st -> 11th. **The profile- and gamma-family conclusions in the table above were drawn under one export and do not transfer.** The "only the KL weight is a large effect" finding is specifically a posterior-export finding.
2. **R-free and peaks now pick different arms.** No arm is best at both, so "best config" has to name the metric. Physically coherent: R-free rewards unbiased intensities, while the anomalous difference is small and benefits from the posterior's shrinkage damping noise in `delta F`.
3. **The Kabsch export is only as good as the model's background.** The two catastrophic cases are exactly the two arms with >50% negative intensities -- `anscombe` (+22 sigma worse R-free) and `coordchan` (+13.3 sigma) -- where `c - bg` is mostly negative because the background absorbed the signal. A checkable precondition: measure the negative fraction before trusting the export.

### Independent support from the error model

Fitting `sigma'^2 = a^2(sigma^2 + (bI)^2)` per arm, the first-cycle `a` -- the binned multiplicative correction making normalised deviations have unit variance -- comes out **5.72 for the posterior export, 2.16 for the Kabsch export, 2.28 for `laue.integrate`**.
So the Kabsch sigma calibrates like the classical pipeline (5% apart) while the posterior sigma is 2.5x worse, which is what the downstream R-free gain then confirms.
Caveat: the full two-parameter model does NOT converge on this data -- all three runs drive `a` below the `min_a = 0.4` guard and take the fallback, with `b` landing at 0.92-2.29 against a typical 0.02-0.05 -- so only the one-step `a` is reportable, not converged `(a, b)`.
These shoeboxes are 96.6% background, so variance is ~20x intensity and the `(bI)^2` term has no purchase; the same regime mismatch makes the DIALS default filter keep 0 of 1.28M reflections.

## Posterior vs Kabsch export, every arm, every metric (10 Oct 2026)

Both columns from byte-identical chains off the SAME checkpoint -- same careless config 4 and seed 2024, same two-pass phenix, same peak script.
Only the two MTZ columns differ, so each row is a paired comparison.
`mean13` is the mean anomalous peak height over all 13 sites; `IOD204` is the strongest iodide.

| arm | R-work post | R-work **kab** | R-free post | R-free **kab** | IOD204 post | IOD204 **kab** | mean13 post | mean13 **kab** |
|---|---|---|---|---|---|---|---|---|
| anscombe | 0.1708 | 0.3147 | 0.2196 | 0.3738 | 12.46 | 5.58 | 5.93 | 2.74 |
| log1p | 0.2973 | 0.3029 | 0.4016 | 0.3922 | 1.06 | 2.30 | 0.30 | 0.80 |
| standardization | 0.1743 | 0.1678 | 0.2214 | 0.2181 | 12.41 | 13.09 | 6.08 | 6.57 |
| profile fixed | 0.1830 | 0.1679 | 0.2348 | 0.2082 | 12.48 | 10.71 | 6.29 | 5.02 |
| profile geomboth | 0.1701 | **0.1420** | 0.2221 | **0.1859** | 13.32 | 14.38 | 6.69 | 7.04 |
| profile learned | 0.1789 | 0.1612 | 0.2312 | 0.2102 | 13.03 | 12.18 | 6.26 | 5.67 |
| profile physical | 0.1786 | **0.1421** | 0.2296 | **0.1828** | 12.46 | 12.63 | 6.04 | 5.71 |
| profile shapelet | 0.1724 | 0.1539 | 0.2188 | 0.2027 | 12.75 | **15.49** | 6.05 | **7.80** |
| coordchan | 0.1793 | 0.2675 | 0.2280 | 0.3212 | 13.92 | 5.55 | 6.73 | 2.97 |
| coordchan_geomboth | 0.1763 | 0.1700 | 0.2293 | 0.2148 | 11.78 | 11.65 | 5.89 | 5.63 |
| mean_fano_exp | 0.1721 | 0.1730 | 0.2183 | 0.2163 | 13.01 | 11.59 | 6.51 | 5.48 |
| mean_fano_softplus | 0.1813 | **0.1408** | 0.2286 | **0.1799** | 13.04 | 12.73 | 6.64 | 6.27 |
| nakagami_exp | 0.1685 | 0.1602 | 0.2154 | 0.2001 | 13.85 | 11.88 | 6.67 | 5.68 |
| nakagami_softplus | 0.1797 | 0.1649 | 0.2261 | 0.2069 | 12.37 | 12.18 | 6.12 | 5.74 |
| shape_rate_exp | 0.1811 | 0.1583 | 0.2310 | 0.2005 | 12.76 | 11.78 | 6.16 | 5.26 |
| shape_rate_softplus | 0.3048 | 0.3016 | 0.3997 | 0.3938 | 0.82 | 2.27 | 0.30 | 0.87 |
| kl 0.1 | 0.1578 | 0.1628 | **0.2024** | 0.2121 | **15.84** | 13.63 | **7.75** | 6.14 |
| kl 0.25 | 0.1855 | 0.1762 | 0.2388 | 0.2173 | 7.62 | 10.53 | 4.46 | 4.76 |
| kl 0.5 | 0.1655 | 0.1817 | 0.2163 | 0.2422 | 13.89 | 13.97 | 6.82 | 6.87 |
| kl 1.0 | 0.1832 | **0.1443** | 0.2362 | **0.1841** | 13.18 | 12.35 | 6.45 | 5.85 |
| **laue-dials** | **0.1570** | -- | **0.2061** | -- | **15.98** | -- | **7.33** | -- |

### Winners, and there is no single one

| metric | best arm | value | vs laue-dials |
|---|---|---|---|
| R-work | `mean_fano_softplus` + **kab** | **0.1408** | beats 0.1570 |
| R-free | `mean_fano_softplus` + **kab** | **0.1799** | beats 0.2061 by 3.7 sigma |
| mean13 peaks | `shapelet` + **kab** | **7.80** | beats 7.33 |
| IOD 204 | `kl_0.1` + **post** | **15.84** | just under 15.98 |

### How many arms beat the classical pipeline

| metric | posterior | **Kabsch** |
|---|---|---|
| R-work | 0/20 | **5/20** |
| R-free | 1/20 | **7/20** |
| IOD 204 | 0/20 | 0/20 |
| mean13 peaks | 1/20 | 1/20 |

**The Kabsch export is where the R-factor wins live** -- 0 to 5 arms on R-work, 1 to 7 on R-free.
It buys nothing on peaks: no arm under either export beats laue's IOD 204 of 15.98, and exactly one arm beats its mean13 under each export.

### The trade-off is real and across arms

The R-factor champion (`mean_fano_softplus` + kab: R-free 0.1799) has mediocre peaks (mean13 6.27).
The peak champion (`shapelet` + kab: mean13 7.80) has mid-table R-free (0.2027).
No arm is best at both, so **"best config" must name a metric**.

### This invalidates the design of the combination arms

The combination arms now running were built from the POSTERIOR-export ranking: `kl_0.1`, `nakagami_exp`, `coordchan`.
Under the Kabsch export the winners are different -- `mean_fano_softplus`, `shapelet`, `physical`/`geomboth` -- and `coordchan` is the single worst arm (0.3212, 52% negatives).
So the running combination is a combination of the wrong factors for the export that wins.
A Kabsch-targeted combination would pair the `shapelet` or `geomboth` profile with `mean_fano` + `softplus`.
