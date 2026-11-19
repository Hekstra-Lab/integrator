# Results by dataset

Best integrator configuration per dataset, with the refinement, anomalous and merging numbers, and the reference each should be read against.
Everything here is collected from runs already done.
Companion to `reference_status.md`, which tracks whether each *reference* arm replicates its deposited values; this file tracks what the *model* achieved against those references.

## Three rules that decide whether a comparison is fair

Each of these was learned by getting it wrong.

**Match the frame count.**
The 1118 Laue arms exist at 360 and 1440 frames and the iodine peaks differ by 5-6 sigma between them.
Pairing a 360-frame reference with a 1440-frame model, or the reverse, invents a result.

**Share the free-R set.**
`dials.merge` writes its own `FreeR_flag` drawn independently per run, so two arms merged separately have incomparable R-free.
Use `refine.py --rfree-from <donor mtz>`, which copies the donor's flags and overrides any already present.

**Use the right peak FILE, not just the right mode.**
Both scripts run on the Laue arm (`peaks: method: both`) and they measure different things.
`rs.find_peaks` reports peaks at model atoms carrying density and writes `peaks.csv`; on 1118 that returns only cysteines and methionines.
`anomalous_peak_heights.py` samples the ANOM map at named sites with `elements: "[S,I]"` and writes `peak_heights.csv`; the IOD 204/205/206 values quoted here come from that file.
Monochromatic arms use `rs.find_peaks` on the ANOM/PANOM map (note `PANOM` for mono, `PHANOM` for poly -- passing the wrong one raises `KeyError`).
Reading `peaks.csv` and comparing it to a documented iodide number compares two different measurements.

**Read phenix pass 2, not pass 1.**
The Laue chain refines twice: pass 1 is a warm-up against the shaken starting model and lands near R-free 0.34, pass 2 picks up its output and reaches the real value.
`refined_2.log` is the file to read; `refined_1.log` reports the warm-up.

**On "best".**
Where a sweep gave a Pareto front rather than a winner (845 is the clear case) the front is listed rather than a single pick.
Peak height is scale-dependent and `cc_anom` is scale-invariant, so they fail differently: always read both, since a method can raise `cc_anom` by shrinking while peak magnitude falls.

## Summary

R values are work/free; peaks are the top anomalous site in sigma.
"Reference" is the matched arm through the same downstream pipeline, not the deposited structure.

| dataset | mode | best config | R-work/R-free | top peak | ref R-free | ref peak |
|---|---|---|---|---|---|---|
| HEWL 9b7c | mono | `hewl9b7c_mono_expmean_lp` (105gh86k ep49) | 0.161 / 0.180 | **IOD 204 36.5** | 0.174 | 32.7 |
| SBGrid 821 | mono | 4,260 | `config_full` (lp + exp + no basis decay) | 0.144 / **0.1626** | MN 203 42.9 | 0.164 | 45.5 |
| SBGrid 845 | mono | 3,373 | Pareto front over beta | 0.156 / **0.1835** (b=0.5) | 18.0 (b=0.25) | 0.1884 | 18.8 |
| 6VYN | mono | `ragged_conv/config_6vyn.yaml` | 0.239 / 0.265 | 2 peaks only | 0.263 | CYS40 12.9, n=24 |
| SBGrid 834 | mono | 146 | reference arm only | 0.201 / 0.216 | 6.4 (n=1) | dep 0.139 | weak control |
| SBGrid 140 | mono | 452 | **broken** | 0.584 / 0.606 | none | dep 0.221 | single-sweep bug |
| HEWL 1118 (1440 fr) | Laue | `screen_l8_k7_exp`, careless c4 | 0.217 | **IOD 204 21.1** | - | laue.integrate 20.5 |
| HEWL 1118 (360 fr) | Laue | same family | - | IOD 204 15.2 | - | laue.integrate 15.7 |
| HEWL 1118 (360 fr) | Laue | fixed-basis profile, Oct 2026 | 0.1678 / 0.2166 | IOD 204 11.8 | - | laue.integrate 15.7 |
| SBGrid 1117 (DHFR) | Laue | `configs/poly/dhfr_1117.yaml` | no deposited model | native S, weak | - | CC1/2 only |
| PDZ2 / 1116 | Laue | 18 arms, all `hierarchical` | 0.1355 / **0.1690** (`k7m_mlclamp`, off state) | n/a | - | laue.integrate 0.1746; CCsym and difference-map peaks below |

Three headline results: 9b7c is the clean win (36.5 vs 32.7 at the strongest iodine, R-free 0.180 vs 0.174, i.e. a little R traded for anomalous signal); 845 at beta 0.5 beats the DIALS reference on R-free *and* `cc_anom` simultaneously; 1118 at 1440 frames ties `laue.integrate` on iodine.

Two honest losses: 6VYN collapses 24 peaks to 2 while refining to the same R, and 845 loses its weakest site.
Both are weak native-sulfur cases, which is the documented limit -- the model's anomalous strength is strong scatterers, and it should not be pitched as a universal anomalous win.

## Monochromatic

### HEWL 9b7c -- iodide SAD, the strongest result

`configs/mono/hewl9b7c_mono_expmean_lp.yaml`, run `105gh86k` epoch 49, 50 epochs, 0.40M parameters.

| | model | DIALS ref | deposited |
|---|---|---|---|
| R-work / R-free | 0.161 / 0.180 | 0.157 / 0.174 | 0.111 / 0.123 |
| IOD 204 | **36.5** | 32.7 | - |
| IOD 206 | **33.5** | 29.7 | - |
| IOD 205 | 15.3 | 14.7 | - |
| mean over 13 shared sites | **17.07** | 16.15 | - |

Wins 10 of 13 sites, largest gains on the highest-f'' atoms -- the signature of real signal rather than sigma inflation, since peak z is map-rms normalised and a uniform sigma rescale cancels.
Knobs that matter: `mean_fano` + `positive_constraint: exp`, `lp_correction: true`, `decoder_weight_decay: 0.0`.
Four things changed at once, so the win is not attributed to any single one.

### SBGrid 821 -- 7LVC, Mn/S-SAD

Four arms, 40 epochs, identical but for the named knobs.

| arm | R-free | peaks >5s | top MN | outer CC-half |
|---|---|---|---|---|
| baseline | 0.1650 | 17 | 43.9 | 0.853 |
| +lp | 0.1642 | **19** | **44.0** | 0.852 |
| +lp +exp +no decay (`config_full`) | **0.1626** | 18 | 42.9 | 0.854 |
| DIALS reference | 0.164 | 20 | 45.5 | 0.923 |

I/sigma higher than DIALS in all 20 shells, R-work a dead heat (0.1444 vs 0.1446), consistent loss in outer-shell CC-half.
Best R-free and best anomalous are different arms.

### SBGrid 845 -- 7RIN, native S-SAD. A Pareto front

KL-weight sweep on the ragged window arm, identical pipeline, R-free on a shared held-out set.

| beta | sites | top s | sum peaks | cc_anom | R-work | R-free |
|---|---|---|---|---|---|---|
| 1.0 | 13 | 15.9 | 145 | 0.262 | 0.1625 | 0.1906 |
| **0.5** | 12 | 16.1 | 140 | 0.357 | 0.1560 | **0.1835** |
| **0.25** | 12 | **18.0** | 146 | 0.310 | 0.1590 | 0.1852 |
| 0.1 | 12 | 17.5 | 146 | **0.516** | 0.1637 | 0.1936 |
| 0.0 | 8 | 10.9 | 67 | 0.949 | 0.1767 | 0.2058 |
| DIALS | 13 | 18.8 | 161 | 0.194 | 0.1612 | 0.1884 |

beta 0.25 and 0.5 both Pareto-dominate DIALS.
Use 0.25 as the S-SAD default, 0.1 to push anomalous consistency hardest.
**beta = 0 is an artifact**: `cc_anom` 0.949 comes from collapsed variances causing DIALS to reject 26% of observations as outliers, leaving survivors that are trivially self-consistent while true magnitude craters.
That case is the standing argument for reading peak height alongside `cc_anom`.

### 6VYN -- external native S-SAD, the clearest failure

Reference: R 0.226/0.263, I/sigma 10.2, 97% complete, **24 sulfur peaks >=5 sigma** (top CYS40 12.9).
Model arm: 0.239/0.265 -- integration is fine -- but **only 2 peaks**, both near-noise at sites with no anomalous scatterer.

Neighbour contamination was ruled out by building explicit overlap masks and retraining: overlap is 0.3% on the clean sweeps and the result was unchanged.
The diagnosis is the documented weak-native-sulfur limit, the same phenomenon as 845.

6VYN is also the genuine depth-ragged case: bbox depth p50 4, p99 18, max 47, where a fixed window would need 209 GB.
It validates the ragged branch on external data.

### SBGrid 834 and 140

834 is a weak control: deposited 0.129/0.139, reference 0.201/0.216, one peak at 6.4 sigma, essentially no anomalous signal.
140 is **broken**: 0.584/0.606 against a deposited 0.166/0.221, from a single-sweep bug.
4U4H's deposited structure factors carry no I(+)/I(-), so its anomalous reference must come from reprocessing.

## Polychromatic (Laue)

### HEWL 1118 -- the paper's experimental benchmark

Reference files, all verified present on 2026-10-08 and all under `/n/holylabs/hekstra_lab/Users/laldama` unless noted.
Refinement starts from `reference_data/shaken_9b7c.pdb` -- the **monochromatic 9b7c model**, which is correct for this dataset -- with `laue_analysis/custom_refinement_param_{1,2}_.eff` for the two passes.
Peak heights come from `anomalous_peak_heights.py` (in `/n/lab_storage/.../refltorch/scripts/laue_output/`) against `reference_data/9b7c.pdb` with `elements: "[S,I]"`.

A current run of that chain reproduces the numbers below: IOD 204 **20.77** against 21.1, IOD 206 **19.87** against 19.7, R-free **0.1992** against 0.217.
That makes it a validated baseline to read other arms against.

NaI-soaked HEWL, BioCARS, P4(3)2(1)2, cell 79.4 79.4 37.8, 3049 stills.
The published Laue CC1/2 of **0.976 at 1.7 A** is reproduced by our laue-dials + careless reference arm -- the one clean published-number replication we have.

Best config family `screen_l8_k7_exp`: latent 8, kernel 7, `mean_fano` + exp mean, `stabilize_scale`, `init_G 5000`.
Kernel size plateaus at 7 (iodine mean sigma 8.16 / 9.01 / **9.12** / 8.70 / 8.92 / 8.74 for k = 3/5/7/9/11/13): it tracks the 5-10 px spot, not the 25 px shoebox.

Anomalous peaks, all arms through careless config 4 and two-pass phenix, **frame-matched**:

| arm | frames | IOD 204 | IOD 206 | IOD 205 |
|---|---|---|---|---|
| ours, `confirm_exp` ep39 | 1440 | **21.1** | 19.7 | 8.4 |
| `laue.integrate` c4 | 1440 | 20.5 | 19.4 | 8.6 |
| ours, older `hewl1440_cmp` | 1440 | 18.0 | 17.4 | 8.4 |
| ours, `w360_ours` | 360 | 15.2 | 13.6 | - |
| `laue.integrate` | 360 | 15.7 | 15.7 | - |
| ours, fixed-basis profile (Oct 2026) | 360 | 11.8 | 12.4 | 5.8 |

At 1440 frames this is a **tie** (+0.6 / +0.3 / -0.2).
`laue.integrate` is not naive summation -- it is profile fitting with KNN profile borrowing and Poisson optimal weighting, the same class of method as ours, so it is near-optimal on exactly the strong reflections that carry the iodide signal.
Our edge is on the weaker sulfurs, and it is a **high-multiplicity phenomenon**: the margin narrows at 360 frames rather than widening.

The October fixed-basis arm sits 3-4 sigma **below** both 360-frame references.
That reproduces a documented trade: a post-hoc profile swap (borrowing a binned strong profile for weak reflections) gave sulfur 6.11 -> 4.19 and iodine 12.19 -> 9.09 while R-free improved 0.217 -> 0.188.
A frozen global basis is the same operation.
Standing conclusion: **profile borrowing is an R-free lever, not an anomalous one**; the anomalous lever is the sigma model.

Fixed-basis merging statistics, careless config 4, for reference: CC-half 0.926 at low resolution rising to 0.999, then 0.947 at 1.83-1.76 A and 0.911 at 1.76-1.59 A; I/sigma 26.1 falling to 1.25; R-pim 0.125 to 0.049.
`cc_anom` is 0.15-0.26 through the mid shells and unusable at the lowest-resolution bin, where three repeats of the same calculation gave -0.675, +0.387 and -0.724 on 1512 unique reflections.

#### Gate architecture, and why the better gate lost downstream

Three architectures trained on hewl_1118, scored through careless config 4 and two-pass phenix on the **pinned** R-free set (`reference_data/hewl_1118_rfree.mtz`), epoch 19 in every arm.
`svae` computes the responsibility gate from the fitted intensity and background; `svae_bg` and `svae_conjugate` predict it.

Each arm was exported twice from the SAME checkpoint: the posterior mean and variance (`qi_mean`, `qi_var`), and the Rao-Blackwellized profile fit (`i_rb`, `var_rb`).
The careless merging seed is its fixed default 1234 and the flag set is the same file, so the two exports of one arm differ only in the estimator.

| arm | R-free posterior | R-free RB | Rsplit posterior | Rsplit RB | CC-anom posterior | CC-anom RB |
|---|---|---|---|---|---|---|
| `svae` | 0.1768 | 0.1712 | 0.0177 | 0.0211 | 0.350 | 0.366 |
| `svae_bg` | 0.1824 | 0.1710 | 0.0245 | 0.0207 | 0.341 | 0.381 |
| `svae_conjugate` | 0.1854 | 0.1720 | 0.0296 | 0.0208 | 0.323 | 0.388 |

CC-anom is the Pearson form, meaned over its shells; the Spearman form is 0.320 / 0.340 / 0.325 posterior and 0.351 / 0.355 / 0.368 RB.
Anomalous peaks, mean over the ten sulfur sites, go 8.37 / 5.92 / 5.78 posterior to 6.96 / **7.85** / 7.26 RB, and at IOD 204 20.91 / 13.82 / 14.97 to 18.84 / **19.95** / 19.11.

**The spread across architectures is 0.0086 in R-free under the posterior export and 0.0010 under the RB export.**
0.0010 is below the 0.005 unseeded-training floor, so under the RB export the three are tied; under the posterior export they are not, and the computed gate appears worst.

The effect is entirely in the weak data.
Rsplit by shell, mean over the three cross-validation repeats:

| shell (A) | `svae` post | `svae` RB | `bg` post | `bg` RB | `conj` post | `conj` RB |
|---|---|---|---|---|---|---|
| 39.76 - 3.91 | 0.0094 | 0.0094 | 0.0098 | 0.0096 | 0.0095 | 0.0096 |
| 2.46 - 2.28 | 0.0127 | 0.0136 | 0.0141 | 0.0137 | 0.0170 | 0.0133 |
| 2.04 - 1.95 | 0.0232 | 0.0270 | 0.0303 | 0.0271 | 0.0392 | 0.0274 |
| 1.95 - 1.88 | 0.0279 | 0.0363 | 0.0429 | 0.0357 | 0.0546 | 0.0372 |
| 1.88 - 1.70 | 0.0455 | 0.0611 | 0.0817 | 0.0572 | 0.0968 | 0.0576 |

In the first shell every arm and both exports agree to 4%.
In the last shell the predicted gates are 80% and 113% worse than the baseline under the posterior export, and 6% BETTER than it under the RB export.

The cause is the exported sigma, not the gate, and the bound on it is one-sided.
`_get_qi` sets $\beta' = \beta_I + \sum_i p_i m_i$, the profile is normalised so that sum is exactly 1, and $\beta_I$ is a Gamma rate and cannot be negative.
So $\beta' = 1 + \tau \ge 1$ and

$$\mathrm{Fano} = \frac{\mathrm{var}}{\mathrm{mean}} = \frac{1}{\beta'} \le 1$$

for every reflection: the posterior can be Poisson-like or slightly under-dispersed and can **never** be over-dispersed.
Measured at epoch 19 on 2,664,858 reflections per arm, the posterior Fano factor runs 0.873 to 0.99995 with **zero** reflections above 1 in any of the three arms, against a Rao-Blackwellized Fano of median 11.2 to 18.8 and p99 587 to 1395, every one of them above 1.
The data wants an order of magnitude more dispersion than the family can reach, in the direction it is closed.
$\mathrm{corr}(\log I, \log \mathrm{var})$ is 0.99993 to 0.99995 as a consequence.
careless config 4 selects `StudentTLikelihood`, whose `dist()` takes `get_uncertainties(inputs)` as the scale and does **not** refit an error model -- only the `Ev11` likelihoods carry `Sdfac`, and config 4 selects neither.
So the reported sigma reaches the likelihood intact and carries no per-reflection information: the merging weight of a reflection is fixed by the intensity it reported, and a measurement summed over 4 clean pixels and one summed over 200 pixels of background are weighted alike.
The RB estimator breaks the family -- $\mathrm{corr}(\log i_{rb}, \log \mathrm{var}_{rb})$ is 0.656 (`bg`) and 0.805 (`conj`) -- and the architecture difference disappears with it.

`intensity_variance: fisher`, which computes the same Schur-complement variance inside the model, was tried in the week of 2026-11-10 and does not work: it enters the ELBO and changes the training dynamics.
The post-hoc export does not, which is why it is the one that works.
The inference variance and the reported variance are different objects, and the family that makes training stable is exactly the one that cannot express the dispersion the data has.
**Score the RB export; treat the posterior sigma as diagnostic only.**

Second cause, specific to the computed gate: it is a plug-in estimator built from the same counts it then weights, so $E[\sum_i r_i(c) c_i] \neq I$ and the reuse adds both bias and variance.
It also sums $n_{\mathrm{eff}}$ of 23 to 28 pixels against the baseline's 4, in shoeboxes that are 96.6% background.

### SBGrid 1117 -- DHFR, CC1/2 comparison only

No deposited PDB, so refinement is not a valid reference.
Comparison is CC1/2 through identical careless (config 1, dmin 1.4, 30k iterations).

| | ours | `laue.integrate` |
|---|---|---|
| weighted (ML) CC-half, overall | **0.577** | 0.363 |
| Pearson CC-half | **0.392** | 0.240 |
| I/sigma overall | 5.71 | 5.69 |

Margin largest in the high-resolution shells (1.68-1.51 A, roughly 0.68-0.76 against 0.13-0.30) with ~2000 reflections per bin in both arms, so not a count artifact.
Caveat: careless's variational-merge CC-half is confound-prone -- its Pearson profile rises with resolution, which is backwards -- so treat it as a same-pipeline relative discriminator.

### PDZ2 / 1116 -- electric-field TR-Laue, where CCsym lives

LNX2 PDZ2, the Nature 2016 EF-X study.
The only dataset where **CCsym** is the endpoint: the half-dataset correlation of F(h) - F(2-fold h), which should be zero without a field.

Two things make the raw statistic unusable alone.
`laue.integrate` shows **CCsym 0.10-0.20 in the field-OFF state** where the true signal is exactly zero, so much of any raw CCsym is reproducible non-field asymmetry.
And on the field-response endpoint our apparent deficit inverts: raw CCsym put the prior-free ML arm behind nearly everywhere (-0.012 / -0.031 / -0.033 / +0.040 by quartile) while the response ties at Q2 and Q3 and leads by **+0.241** at Q4.

Three corrections to how those were reported.
The odds transform r = CC/(1-CC) is correct and Fisher's z is not, but r is strained as CC approaches 1, which is **Q4** where the headline sits, not Q1 as previously stated.
Normal-theory standard errors understate by ~1.8x from heavy tails, so "+19 sigma" should be restated nearer +10 sigma using a paired reflection bootstrap.
The recommended estimator is **CC_Delta**: correlate the per-reflection field response formed inside each half and report 2CC/(1-CC) = S/N; it is split-rule invariant and cancels bias common to on and off.

State on 2026-11-18: the reference pipeline is done, the integrator arm is done, and the model inventory is six weeks behind the current one.
Nothing has been trained or scored on this dataset since 2026-10-04.

The reference pipeline under `laue_runs/pdz2_ref/` reproduced the Laue-DIALS paper's EF-X result in September.
CCsym in the first bin came out 0.190 / 0.357 / 0.371 / 0.415 for off / 50 / 100 / 200 ns against the published 0.19 / 0.33 / 0.38 / 0.40, and the weighted 200 ns difference map reproduces all ten published peaks in sign within 0.5-1 $\sigma$.
The earlier claim that our reprocessing could not reach the ~1% field signal was a processing error on our side, not a limit of the data, and is withdrawn.

#### The arms that were trained

Eighteen under `laue_runs/pdz2_model/runs/`, every one `integrator: hierarchical` with `loss: polychromatic_wilson` on the 993,143-reflection pool.
CCsym is the mean over bins and repeats at 200 ns; the off column is the control, where the true value is zero.

| arm | `qp` | what differs | CCsym 200 ns | off | CCpred bin 0 |
|---|---|---|---|---|---|
| `kl1` | learned basis | baseline, `intensity_kl_weight` 1.0 | 0.211 | 0.037 | 0.845 |
| `kl0p1` | learned basis | `intensity_kl_weight` 0.1 | 0.223 | 0.028 | 0.865 |
| `w21_k3m` | learned basis | 21 px, kernel 3, neighbour-masked | 0.203 | 0.042 | 0.833 |
| `w21_k7m` | learned basis | kernel 7 | 0.216 | 0.042 | 0.847 |
| `w35_k3m` | learned basis | 35 px, kernel 3 | 0.176 | 0.020 | 0.790 |
| `w35_k7m` | learned basis | 35 px, kernel 7 | 0.176 | 0.021 | 0.790 |
| `geom_shift` | geometric | `local_shift` | **not scored** | -- | -- |
| `geom_steer` | geometric | `steer` | 0.218 | 0.025 | 0.838 |
| `geom_both` | geometric | both | 0.234 | 0.042 | 0.863 |
| `tau0p1` | geometric | `qp.prior_scale` 0.1 | 0.228 | 0.049 | 0.865 |
| `tau0p3` | geometric | 0.3 | 0.223 | 0.033 | 0.861 |
| `tau1p0` | geometric | 1.0 | 0.235 | 0.044 | 0.865 |
| `tau3p0` | geometric | 3.0, the `geom_both` default, re-run as the sweep control | 0.240 | 0.043 | 0.862 |
| `bgplane` | geometric | `qbg_plane: background_plane`, prior scale 0.1 | 0.229 | 0.037 | 0.856 |
| `bgplane03` | geometric | prior scale 0.3 | 0.235 | 0.036 | 0.862 |
| `invN` | learned basis | `kl_per_multiplicity: true` | **not scored** | -- | -- |
| `w35_k3`, `w35_k7` | learned basis | unmasked; stopped at 3 checkpoints, replaced by the masked pair | -- | -- | -- |

Every checkpoint of every arm is on disk, 14 to 51 per run, so any of these can be re-exported without retraining.

#### The arms that were built from a trained arm's predictions

No retraining; these change the export or the merge.

| arm | from | what it changes | CCsym 200 ns | off | CCpred bin 0 |
|---|---|---|---|---|---|
| `w21_k7m_bfix` | `w21_k7m` | BATCH convention repaired | 0.224 | 0.035 | 0.844 |
| `k7m_ml` | `w21_k7m` | prior-free profile fit, negatives kept | 0.239 | 0.043 | 0.857 |
| `k7m_mlclamp` | `w21_k7m` | the same, clamped at zero | 0.242 | 0.048 | 0.857 |
| `k7m_ev11` | `w21_k7m` | careless `--refine-uncertainties` | 0.206 | 0.057 | 0.848 |
| `combo` | `geom_both` | geometry-steered profiles **and** the prior-free fit | **0.267** | 0.049 | **0.884** |
| `swap_LL` / `LM` / `ML` / `MM` | both arms | $I$ and $\sigma$ crossed between arms | 0.241 / 0.244 / 0.221 / 0.215 | -- | -- |
| `laue.integrate` | -- | the reference | 0.253 | 0.076 | 0.869 |

`combo` is the best CCsym and the best CCpred of anything run here, reference included, and it was never written down.
Three merging seeds give 0.2667 / 0.2690 / 0.2498 against the reference's 0.2530 / 0.252 / 0.257, so +0.008 against a seed noise floor of 0.012 -- a tie, not a lead.
Its difference map is worse: mean $|{\rm peak}|$ 2.92 against 3.14, with 8 sites above 3 $\sigma$ against 10.
So the trade the prior-free export already showed reappears in `combo` and is larger: the two interventions add on CCsym, and they add on the map in the wrong direction.

#### Where each endpoint stands

| endpoint | arms scored | best | reference |
|---|---|---|---|
| CCsym, by delay | 36 trees, all of them | `combo` 0.262 $\pm$ 0.011 (n=3) | 0.254 |
| weighted difference map, 20 published sites | 19 trees | `geom_both` 3.12 | **3.14** |
| $R_\mathrm{free}$ on the off state | **4 arms** | `k7m_mlclamp` **0.1690 $\pm$ 0.0013** | 0.1746 $\pm$ 0.0007 |

No arm leads on all three, and the one that leads on $R_\mathrm{free}$ by 3.8 $\sigma$ is not the one that leads on CCsym.
$R_\mathrm{free}$ is the endpoint to weigh: it is the only one of the three that half-dataset profile borrowing cannot inflate, and the deposited pipeline computes none, so the refinement chain is ours (`pdz2_model/pdz2_refine_prep.py`, one R-free set of 404 reflections shared across every arm).

#### What has not been run

Three gaps in what exists, then the model generation.

- **`combo` has no $R_\mathrm{free}$.** The best CCsym arm has never been through the endpoint the dataset is judged on. The refinement chain is built and takes one merged MTZ, so this is a short CPU job.
- **`invN` was trained and never scored.** `kl_per_multiplicity: true` applies the Wilson prior once per unique reflection instead of once per observation at multiplicity ~25, which is the structural form of the fix that `k7m_ml` achieves by removing the prior after the fact. 42 checkpoints are on disk and no tree was ever built.
- **`geom_shift` was trained and never scored**, so `local_shift` alone is untested here while `steer` alone and both together are not.

Nothing from the current model generation has touched this dataset.

| what | state on PDZ2 | where it was established |
|---|---|---|
| `integrator: hierarchical_3enc` | never run; all 18 arms are `hierarchical` | every `ablate2` arm on 1118 |
| `integrator: svae` | never run | `configs/poly/ablation_1000/`, 13 poly arms |
| the `ablation_1000` profile set (`physical`, `shapelet`, `fixed`, `learned`, `geom_both` crossed with both integrators) | never run | `ablation_1118_1000.md`; `svae physical` best $R_\mathrm{free}$, `svae learned` best peaks |
| `intensity_variance: fisher` | never run | the $q(I)$ one-parameter family fix |
| `gate: responsibility` | never run | 1118 poly, CC-against-$F_\mathrm{calc}$ 0.8881 to 0.9218 |
| `predict --rb-export` (`i_rb`, `var_rb`) | never run | `scripts/figures/tidy_xtal.sh`, both mono arms |

The `--rb-export` line is the one to run first, and it needs no retraining.
`rao_blackwellized_intensity` is the same estimator class as `pdz2_model/ml_from_preds.py`, the script behind the winning arm -- the variance-weighted profile fit over the model's own profile and background, with negatives surviving because only the rate inside the weights is floored.
It differs in the two places the hand-written version is weakest: it fits once per profile sample and adds the spread of the fits to $\sigma$ by the law of total variance, where `ml_from_preds.py` conditions on one profile, and its conditional term is the Schur complement of the $(I,b)$ information rather than $1/F_{II}$, so it charges for the background being estimated.
Running it on the `w21_k7m` and `geom_both` checkpoints replaces a one-off script with maintained code and tests whether the better $\sigma$ moves the endpoint that the $\sigma$-swap trees said $\sigma$ does not own.

Two things that would otherwise have to be checked are already settled for this dataset.
The Anscombe manifest defect does **not** apply: `fix_manifest_stats.py` found 1118 and 1117 wrong by 1.85x and 1.47x in sd and PDZ2 in agreement with its own counts, and `pooled/dataset.yaml` has never been rewritten, so a predict pass on a September checkpoint is safe.
And profiles can be exported here: `pdz2_profile_predict.yaml` carries `additional_predict_keys: [profile]` and produced `prof_preds_geom_both`, so the rule against judging a profile model on metrics alone is satisfiable without new plumbing.

#### The chain is not in the repository

The entire PDZ2 chain -- about 110 scripts, including `pool_pdz2.py`, `split_pdz2.py`, `split_pdz2_ml.py`, `ml_from_preds.py`, `wdm_peaks_compare.py`, `pdz2_refine_prep.py`, `collect_rfree.py`, `unmerged_ccsym.py` and every `.sbatch` -- exists only in `/n/netscratch/.../laue_runs/pdz2_model/`.
There is no PDZ2 pipeline configuration, no PDZ2 step in `scripts/poly/run_pipeline.py`, and no figure script that reads CCsym.
That scratch filesystem has already purged this project's raw images once.

## The workflow

Two chains, one per mode, both ending at the same four numbers: merging statistics, refinement, anomalous peaks, and (EF-X only) CCsym.

### Monochromatic

```
integrator.train
  -> integrator.predict                  prediction parquets
  -> write_predictions.py                split per sweep, back into .refl
  -> dials.scale + dials.merge           reusing the reference integrated.expt
  -> iotbx.merging_statistics
  -> refine.py --rfree-from <ref mtz>    phenix, shared free set
  -> rs.find_peaks -f ANOM -p PANOM -z 5
```

Driver `scripts/sbgrid/cannon/05_integrator_pipeline.sh <run-dir> <epoch>`; side-by-side table `scripts/sbgrid/compare_arms.py --reference <dir> --integrator <dir>`.
Reusing the reference's `integrated.expt` is what keeps the crystal models identical between arms.

### Polychromatic (Laue)

```
integrator.train
  -> integrator.predict --write-mtz      preds_epoch_XXXX.mtz
  -> careless poly, config 4             env crls, A100
  -> phenix.refine TWICE                 eff1 on merged + shaken model;
                                         eff2 on refine1 output, reusing its R-free flags
  -> anomalous_peak_heights.py <refine2 mtz> 9b7c.pdb [S,I]
  -> careless.ccanom on *_xval_0.mtz
```

Driver `scripts/poly/run_pipeline.py --run-dir <dir> --cfg scripts/poly/poly_pipeline_cfg.yaml`, with `--steps` to resume partway and `--dry-run` to print the exact shell.

### careless configs

From `hewl_1118/laue-dials-careless/config{1..6}.sh`.
**Config 4 is the default**: anomalous plus positional encoding.
1 is the anomalous baseline, 2 and 3 are Friedel-split with double-Wilson, 5 and 6 vary `image-layers` and `mlp-width`.

Shared flags `--merge-half-datasets --half-dataset-repeats=3 --mc-samples=10 --mlp-layers=10 --studentt-likelihood-dof=64 --wavelength-key=wavelength --dmin=1.5`, metadata keys `BATCH,xcal,ycal,dHKL,wavelength` -- exactly what `write_mtz_from_preds` emits.

PDZ2 is the exception: one **joint** careless over 8 files (4 timepoints x {C2 shared-index, P1}) with `--separate-files`, **no** `--anomalous`, 12k iterations, test fraction 0.1, and `is_c/is_d/is_e/is_f` pass indicators in the metadata.

### Environments and their traps

`integrator-cuda-dev` for train and predict, `crls` for careless, phenix 1.20 `phenix_env.sh` for refinement.

Console scripts in `crls` and `ld` carry **dead shebangs** from the storage migration -- invoke through the env's python, `micromamba run -n crls python $(command -v careless)`.
careless's TensorFlow has **no kernels for Blackwell GPUs**, so pin it to A100 (`--gres=gpu:nvidia_a100-sxm4-80gb:1`); PyTorch is fine on either.
`laue.predict` deadlocks on fork-after-BLAS-threads unless `OMP_NUM_THREADS=1` is set (11.5 hours -> 92 seconds).

### Where artifacts land

`run_paths.yaml` in each run directory names the predictions directory.
Everything downstream hangs off it as `predictions/epoch_XXXX/scaling/config4_refine/`, holding `merging_stats.csv`, `peaks.csv`, `peak_heights.csv`, `ccanom.csv`, and the two `refine*/refined_*.log` files carrying the R values.

## Where the data and results live

Audited 2026-10-07 with `du -sh`.
**Almost everything is on netscratch, which is purged.**
The SBGrid raw images were already lost once, around 2026-09-14, and had to be re-fetched with `rsync -rl --size-only` -- plain rsync skipped files because the mtimes had drifted.

Total on netscratch: roughly **862 GB**.

### Netscratch - volatile

Root `/n/netscratch/hekstra_lab/Lab/laldama/` unless noted.

| what | size | files | path | tier |
|---|---|---|---|---|
| SBGrid 845 | 298G | 3,373 | `sbgrid/845` | dataset + results (290G under `integrator/`) |
| 6VYN (757-763) | 182G | 660 | `sbgrid/6vyn` | reference 93G + results 90G |
| nanoBragg DHFR sim | 180G | 4,708 | `/n/netscratch/hekstra_lab/Lab/data/move_from_tier_3/data` | raw (from Derek) |
| SBGrid 821 | 49G | 4,260 | `sbgrid/821` | dataset + results (47G under `integrator/`) |
| PDZ2 reference run | 42G | 2,258 | `laue_runs/pdz2_ref` | re-fetched images + results |
| SBGrid 834 | 33G | 146 | `sbgrid/834` | dataset (contents unverified) |
| 1118 screen (config sweep) | 22G | 1,419 | `laue_runs/screen` | **results** |
| SBGrid 140 | 22G | 452 | `sbgrid/140` | broken, contents unverified |
| 1118 HEWL 360 | 6.4G | 97 | `laue_runs/1118_hewl360` | dataset + laue-dials ref |
| 1117 DHFR | 6.3G | 304 | `laue_runs/1117_ldint` | dataset + results |
| PDZ2 (first pass) | 1.5G | 583 | `laue_runs/pdz2` | results |
| 1118 1440-frame comparison | 915M | 113 | `laue_runs/hewl1440_cmp` | **results** |
| 1118 fixed-basis (Oct 2026) | 743M | 393 | `laue_runs/1118_fixedbasis` | **results** |

Nested paths worth naming individually:

- `sbgrid/821/integrator/dataset` (6.6G) -- the extracted shoeboxes, `counts.npy` / `masks.npy` / `metadata.npy`
- `laue_runs/1118_hewl360/shoeboxes` (4.4G) -- same, for 1118
- `laue_runs/1118_hewl360/integrated.mtz` -- the `laue.integrate` baseline, i.e. the reference arm every 1118 comparison is read against

**Inodes, not bytes, are the binding constraint for the destination.**
The whole 862 GB is only about **18,800 files**, because the bulk is a handful of very large `.npy` arrays -- 821's entire shoebox dataset is 10 files, 1118's is 22.
rsync pays per directory entry, so this transfers at close to raw disk speed and needs no tarring.

The raw diffraction images are **already gone** from these trees (purged ~2026-09-14 and never re-fetched), so what remains is processed artifacts: 845's 298 GB is 290 GB of `integrator/` outputs against only 7.8 GB of `dials_reference`, and 6VYN splits 93 GB `ref/` and 90 GB `integrator/`.

### Lab storage - persistent

- `/n/lab_storage/hekstra_lab/people/aldama/integrator_data/hewl_9b7c/scaling_dataset` (6.1G) -- the 9b7c shoeboxes; the only dataset already on persistent storage
- `/n/lab_storage/hekstra_lab/people/aldama/integrator_data` (159G total) -- also holds data outside the netscratch inventory above; fold it into any consolidation
- `/n/lab_storage/hekstra_lab/people/aldama/software/integrator` (20G, 94,727 files) -- the checkout the pipeline scripts call as `INTEGRATOR_ROOT`

**That checkout is 97.5% stale conda environment.** `.conda/` holds 92,367 of its 94,727 inodes and was last modified 2026-06-19; nothing in the repo references it and every driver resolves environments by name through micromamba (`integrator-cuda-dev`, `crls`, `ld`, `refltorch`) in a different location. Deleting `.conda/` recovers 92k inodes and most of the 20 GB at no risk. The repository proper -- source, scripts, configs, docs and full history -- is ~2,300 inodes, with `.git` only 1,741 of them.

**Do not delete the checkout itself without diffing first.** It sits on branch `svae-integrator` at `eec2cba` (2026-09-01) with 23 modified tracked files and 74 untracked ones, including `src/integrator/io/mtz_io.py`, `profile_surrogates.py`, `base_integrator.py`, `wilson_loss.py` and `configs/ablation_likelihood/`. Some overlap the local working copy, but whether the cluster holds anything unique is unverified.

### What to move, in priority order

Priority is per gigabyte of irreplaceable content, not by size.

1. **Results** -- checkpoints, predictions, merged MTZs, refinement logs, peak CSVs.
   Small (a few GB in total) and **not reproducible without re-running everything**.
   `laue_runs/screen` (22G), `laue_runs/hewl1440_cmp`, `laue_runs/1118_fixedbasis`, `laue_runs/pdz2`, and the `integrator/` subtree inside each `sbgrid/<id>/`.
2. **Extracted shoebox datasets** -- `sbgrid/821/integrator/dataset`, `laue_runs/1118_hewl360/shoeboxes`, and the equivalents under 845 / 6vyn / 834.
   Regenerable from raw plus the DIALS reference, but each costs a full extraction.
3. **The laue-dials reference arms** -- `integrated.mtz` / `integrated.refl` and the `.expt` files under `1118_hewl360`, `1117_ldint`, `pdz2_ref`.
   Small, and every model comparison is read against them.
4. **There is no raw-image tier left to deprioritise.**
   The SBGrid frames were purged around 2026-09-14 and never re-fetched, so the 862 GB is entirely processed artifacts.
   Re-fetching, if ever needed, is `scripts/sbgrid/fetch.py` with `rsync -rl --size-only` -- plain rsync skips files because the mtimes have drifted.
   PDZ2 is the exception: its images WERE re-fetched into `pdz2_ref/data/` for the reference run.
5. **The nanoBragg simulation** (180G) came from Derek Mendez and is not on SBGrid; re-fetching means asking him.
   Its parent directory is literally named `move_from_tier_3`, so it may already be mid-migration and worth checking before copying.

## Gaps

Stated so the table is not read as complete.

- **Merging statistics are not recorded uniformly.** Only the Oct 2026 1118 fixed-basis arm has a `merging_stats.csv` from the current pipeline. For 9b7c, 821 and 845 the R values and peaks are documented but CC-half / R-pim / I/sigma per shell are not collected in one place.
- **1118 R-free has no frame-matched reference.** Ours is 0.217 at 1440 and 0.2166 at 360; `laue.integrate`'s 0.194 is quoted at 1440 only, and the 360-frame reference R-free was never extracted.
- **9b7c and 1118 merging statistics predate the current `merging_stats.csv` schema**, so the numbers in older notes are not directly comparable to new runs.
- **834 and 845 reference refinements undershoot their deposited values** (0.216 vs 0.139; 0.188 vs 0.154); the reference refinement needs redoing before those rows are quoted as replications.
- **140 is broken** and needs the single-sweep bug fixed before any model comparison.
- **1117 has no deposited model**, so MR of 1RX2 is needed for a refinement reference; the first phaser attempt produced no solution.
- **PDZ2 CCsym is now tabulated per arm above.** The quartile numbers in the notes still carry normal-theory standard errors and need restating on a paired bootstrap.
