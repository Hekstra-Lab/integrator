# The DIALS scale + merge pipeline, and its fit to a positive-support integrator

This document maps every stage of `dials.scale` followed by `dials.merge`, states
what each stage assumes, and gives a suitability verdict for intensities produced
by a variational integrator with strictly-positive (posterior-mean) output.

Verdicts are labelled by evidence: **measured** means checked numerically on the
HEWL A/B (`scaled.refl` = DIALS-integrated, `dials_scaled.refl` = model-integrated,
same crystal, same DIALS scaling); **reasoned** means argued from the code but not
yet measured.

The reference numbers throughout come from that A/B and from the port, which
reproduces DIALS to ~1e-15 (merge, scale) and 5e-5 (error model `a`, `b`).

## The two questions that decide suitability

Every verdict below turns on two properties of the model's intensities that
conventional DIALS-integrated intensities do not have.

1. **Strict positivity.** Model intensities are never negative (0.00% vs DIALS'
   8.6% at the observation level, 3.89% even after merging). Any DIALS step whose
   logic keys on the sign or the negative tail becomes a silent no-op -- it does
   not error, it just stops doing anything.
2. **Posterior-mean shrinkage.** A Gamma posterior mean `E[I | shoebox]` is a
   prior-informed estimate, not a maximum-likelihood one. Any DIALS step that
   assumes its input is an unbiased likelihood estimate, and then applies its own
   prior, double-counts.

## Stage-by-stage

### dials.scale

**S1. Intensity choice** (`intensity_choice=combine`, default).
Scores a grid of profile/summation crossover values (`Imid`) by Rmeas.
*Verdict: degenerate (measured).* The exporter writes `prf == sum`, so the scan
is meaningless; the model run selected `Imid=0` (pure profile). Set
`intensity_choice=sum` (or `profile`). Harmless but wasteful as-is.

**S2. Reflection selection for minimisation** (`quasi_random`, `Isigma_range=-5,0`).
Chooses a well-connected subset to refine the scale model on.
*Verdict: filter inert (measured, by inspection of the threshold).* `I/sigma > -5`
passes every positive reflection, so the cut that normally trims the noisy tail
does nothing. Not harmful -- it selects *more* data, not wrong data -- but the
selected population is no longer the "safely bright" subset the defaults assume.

**S3. Symmetry grouping** (`Ih_table`, ASU mapping).
Integrator-agnostic. *Verdict: fine.* (The port replaces cctbx with gemmi here;
the only subtlety, centric reflections not being split by Friedel parity, is a
symmetry fact, not an integrator one.)

**S4. Scale model** (physical: smooth scale + smooth decay/B + SH absorption).
Pure geometry -- rotation angle, resolution, crystal-frame scattering vectors.
*Verdict: fine.* Nothing in the parameterization sees the intensity distribution.
The port reproduces DIALS' `inverse_scale_factor` to 1.8e-15 from its refined
parameters. Note DIALS hardcodes the absorption reference axis as `(1,0,0)` and
quantizes the SH angles onto a 0.5-degree grid above 100k reflections -- both are
DIALS quirks, not integrator mismatches.

**S5. The scaling target** (`L = sum w (I - g*Ih)^2`, `w = 1/sigma^2`,
`Ih = sum(w g I) / sum(w g^2)`).
A Gaussian weighted-least-squares, i.e. an implicit Gaussian likelihood on `I`
with fixed per-reflection variance.
*Verdict: mostly fine, one real caveat (reasoned).* As a quasi-likelihood it stays
consistent whenever `E[I] = g*Ih`, which the data support (see S7: `a > 1` on both
integrators, so the mean model is not pathologically violated). It is written as
`I - g*Ih`, scaling the *model* up rather than dividing the observation, so it
never divides by a near-zero quantity -- a genuinely robust choice for weak data.
The caveat: `w = 1/sigma^2` is held fixed and is a function of the observation
itself. When `sigma` correlates with `I`, downward fluctuations get more weight,
biasing `Ih` low. This bias exists for conventional data too; whether it is worse
for posterior-mean intensities is the one **open, unmeasured** question in the
scaling half. A `weight_mode="unity"` A/B on `Ih` would settle it.

**S6. Outlier rejection** (`norm_dev`, symmetric `|z| > 6`).
Flags observations far from their group's leave-one-out mean.
*Verdict: one-sided in effect (reasoned).* The `abs()` makes it two-sided, but on
strictly-positive right-skewed data the lower tail is arithmetically unreachable,
so essentially every flagged outlier is a bright reflection. The phil floor
`value_min=3.0` prevents loosening it, and there is no asymmetric option in stock
DIALS (the port adds `zmax_lower`). Low-severity: at multiplicity ~40 few
reflections are near the boundary.

**S7. Error model** (`sigma'^2 = a^2 (sigma^2 + b^2 I^2)`, `a`/`b` fitted).
Inflates sigmas so normalized deviations have unit variance.
*Verdict: works, with two asterisks (measured).* It fits a sensible correction:
`a = 1.384, b = 0.034` for the model vs `a = 1.174, b = 0.039` for DIALS. `a > 1`
on both means both integrators are overconfident (observed scatter exceeds claimed
sigma); the model is ~18% more so. Crucially, the per-bin residual variances
flatten to ~1.0 across five orders of magnitude of intensity (spread 1.26x for the
model, *tighter* than DIALS' 1.35x), so **the miscalibration is uniform, not
concentrated at the weak end** -- a single global `a` is the right shape of
correction here. The `a/b` degeneracy feared for `sigma^2 ~ I^2` data does not
arise, because the model's variance is Poisson-like (`V ~ I`), keeping the two
terms separable.
Asterisk 1: DIALS' `calc_deltahl` applies an **inverted** multiplicity prefactor
(`sqrt((n-1)/n)` where it should be `sqrt(n/(n-1))`), so every DIALS `a` and output
sigma is low by `(n-1)/n`. It biases both integrators equally at matched
multiplicity, so the comparison stands, but both `a` values understate the true
miscalibration. See `dials_deltahl_prefactor_bug` memory.
Asterisk 2: the sign-based *input filters* (`intensities > 0.001`,
`avg_I_over_var > 0.85`) silently stop excluding anything on positive data. This
changes which reflections the fit sees, not the fit's validity.

### dials.merge

**M1. Export filtering** (`bad_for_scaling` flags, `partiality > 0.4`, `d` cuts).
*Verdict: fine.* Flag- and geometry-based, not intensity-sign based.

**M2. Un-scale** (`I/g`, `sigma/g`).
*Verdict: fine.* Treats `g` as exact (ignores `Var(g)`), same for both integrators.

**M3. Merge** (`merge_equivalents`: inverse-variance weighted mean, run twice --
Laue for IMEAN, point-group for I(+)/I(-)).
`I(hkl) = sum(w I)/sum(w)`, `sigma = sqrt(1/sum(w))`, `w = 1/sigma^2`.
*Verdict: fine, inherits S5's weight caveat (measured to 7e-15 vs cctbx).* The
merge is a sound inverse-variance combination. It carries the same
weight-correlation caveat as the target, and it propagates sigmas only (default
`use_internal_variance=False`), so the reported merged sigma trusts the error
model entirely -- which is why S7's miscalibration flows straight through to the
merged uncertainties.

**M4. French-Wilson / truncate** (`truncate=True`, default; Wilson-prior posterior
`J = E[I | I_obs]`, `F = E[|F| | I_obs]`).
*Verdict: partially mismatched (measured).* This is the step most worth scrutiny.
FW bundles three operations with different suitability:
- *Negative/rejection handling* (reject `I/sigma < -3.7`; pull negatives up toward
  the shell mean): a **no-op** on model data. 71 DIALS reflections rejected, 0 model;
  FW nudges DIALS' weak reflections `+0.04 sigma`, i.e. up. This is the same
  sign-based inertness seen in S2/S6.
- *Wilson shrinkage* of weak positive reflections: **double-counting (measured).**
  For the model's weak reflections (`I/sigma < 3`, ~16% of the set), FW applies an
  additional **`-0.24 sigma` downward** shrinkage toward the exponential Wilson
  prior -- on top of whatever prior the model's posterior mean already imposed.
  The direction is opposite to what FW does to DIALS data (`+0.04 sigma`),
  precisely because model intensities are already positive and posterior-like.
  How much of this is genuine double-counting depends on how Wilson-like the
  model's learned prior is; the per-shell intensity distribution is close to
  exponential for both (shape ratio 0.98 model, 0.95 DIALS), which is suggestive
  but not conclusive.
- *The sqrt bias correction* (`F = <sqrt(J)> != sqrt(<J>)`): **genuinely needed
  (measured).** Even for the model, `F` differs from naive `sqrt(J)` by `-3.8%` on
  weak reflections. This is the correct nonlinear I->F transform and is not a
  prior double-count.
Implication: if amplitudes are wanted, the sqrt-bias part must be kept, but the
Wilson-shrinkage part is arguably applying a prior twice. A defensible alternative
for a posterior-mean integrator is to skip FW's shrinkage and apply only the
positivity-respecting I->F moment transform, or to feed intensities (not
amplitudes) downstream and let refinement handle the prior once.

## Summary table

| Stage | Suitability | Evidence |
|---|---|---|
| S1 intensity choice | degenerate, set `intensity_choice=sum` | measured |
| S2 reflection selection | filter inert, not harmful | measured |
| S3 symmetry grouping | fine | -- |
| S4 scale model | fine | measured 1.8e-15 |
| S5 scaling target | fine; weight-correlation caveat | reasoned |
| S6 outlier rejection | one-sided in effect, low severity | reasoned |
| S7 error model | works; global `a`; prefactor bug; filter no-ops | measured |
| M1 export filtering | fine | -- |
| M2 un-scale | fine | -- |
| M3 merge | fine; inherits S5 caveat | measured 7e-15 |
| M4 French-Wilson | shrinkage double-counts; sqrt-bias needed; rejection no-op | measured |

## The one-line answer

`dials.scale` and `dials.merge` do not need to be modified to *accept*
positive-support intensities -- nothing errors, and the error model correctly
reports the model as overconfident (`a = 1.38`) rather than breaking. The steps
that are conceptually mismatched are not in the scaling target; they are the
prior-bearing and sign-bearing steps: French-Wilson shrinkage (double-counts a
prior) and the family of sign-based filters/rejections that silently no-op. The
scaling target's fixed inverse-variance weighting is the one place a real,
unmeasured bias could hide, and it is the natural next experiment.

## Open items

- **Weight-mode A/B on `Ih`** (S5/M3): does `w = 1/sigma^2` vs `unity` shift the
  merged intensities systematically? The only unmeasured scaling-side question.
- **CCanom with `n_repeats`** averaging: the headline 0.366 vs 0.232 A/B numbers
  are single random-split draws with per-shell sd ~0.03; they need averaging
  before any anomalous conclusion.
- **FW-free amplitude path**: quantify the downstream effect of skipping FW
  shrinkage while keeping the I->F moment transform.
