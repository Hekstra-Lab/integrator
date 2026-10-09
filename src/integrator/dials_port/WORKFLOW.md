# Scaling/merging investigation — what we tried, in order

The question: our integrator is Bayesian (positive-support Gamma posteriors),
but `dials.scale`/`dials.merge` are frequentist.
Does DIALS need modifying for our intensities, and can we do better?

Dataset throughout: HEWL iodide-SAD (9b7c), P 4₃ 2₁ 2, 1.038 Å, ~1.93M obs.
A/B pair: model-integrated vs DIALS-integrated reflections, same crystal.
Verification oracle: `816_sbgrid_HEWL/pass1` (P 4₁ 2₂), whose `scaled.expt`
carries DIALS' own refined parameters.

## 1. Audit the DIALS pipeline

Read the scaling error model, target/weighting, and outlier/merge machinery.
Found: nothing errors on positive-support data, but several steps are
sign-based proxies that silently no-op, and French-Wilson applies a prior the
posterior mean already carries.
Written up per-step in `PIPELINE.md`.

## 2. Build a verified torch port

Reimplemented DIALS scale + merge in torch float64, symmetry via gemmi, gradients
via autograd (no cctbx runtime).
Verified against DIALS/cctbx: merge + French-Wilson to 7e-15, physical scale
model to 2e-15, error model `(a,b)` to 5e-5, deterministic stats to ~1e-15.
This is the substrate that let every later claim be measured, not argued.

Found along the way: an inverted `calc_deltahl` multiplicity prefactor in DIALS
(`sqrt((n-1)/n)` instead of `sqrt(n/(n-1))`), so DIALS' `a` and all its output
sigmas are low by `(n-1)/n`.

## 3. Calibration

Refit the error model on each integrator (recovered DIALS' published values
exactly as a cross-check).
Model `a = 1.38` vs DIALS `a = 1.17`: our sigmas are ~18% more overconfident,
but uniformly across intensity (per-bin residual variance flatter than DIALS').
The feared `a`/`b` degeneracy did not appear — our variance is Poisson-like
(`V ≈ I`), not `∝ I²`.

## 4. The decisive metric — anomalous peaks

Refined both merged datasets with phenix (your `ref_phenix.eff`) and ran
`rs.find_peaks -f ANOM -p PANOM -z 5`.
DIALS arm reproduced `reference_peaks.csv` almost exactly → pipeline validated.
Model beat DIALS: mean peak z 17.07 vs 16.15 (+0.9σ), wins 10/13 sites, iodines
+2.8/+3.3σ.
It is S/N not sigma inflation (z is map-rms-normalised; gains concentrate on the
highest-f″ atoms).
Model refines to slightly worse R (0.181 vs 0.175) but better anomalous peaks —
the right trade for anomalous work.

## 5. French-Wilson: needed by DIALS, redundant for us

Compared three I→F treatments (phenix FW / our port FW / flat positivity-only).
Wilson shrinkage buys DIALS +1.73σ but the model only +0.31σ — our posterior
mean already did the regularization FW exists to do.
Our port FW matched phenix FW to ~0.07σ (independent cross-check).
Model-flat still beats DIALS-full-FW, so the edge survives dropping FW.

## 6. TorchRef as a fast substitute

Validated TorchRef against phenix: per-site peak z correlation 0.9997, same
model−DIALS margin (+0.84 vs +0.91), ~8× faster (31s vs 4min).
Used it to screen every variant below.

## 7. Fix the weight-correlation bias

DIALS' `w = 1/σ²` merge biases weak merged `Ih` low; measured 3–4× worse for us
(~15% at the weak end) because our σ tracks `I` more tightly.
Switching to the count/Poisson merge `ΣI/Σg` raises anomalous peaks ~+1σ — for
**both** integrators.
So it is a real free win, but a merge-level fix orthogonal to the integrator; the
margin stays ~+0.85σ.
Confirmed in phenix and TorchRef.

## 8. Toward fully Bayesian — three merge-level attempts

Motivation: Bayesian integration handed to a frequentist merge throws away the
posterior and re-applies the prior. Can the merge be made Bayesian?

**8a. Deconvolve + merge once (per-HKL Wilson prior).**
Recover per-obs likelihoods (Gamma cavity), apply Wilson once per HKL.
Well-conditioned for the model (0.7% fallback) but not DIALS (43% — its weak
posteriors carry too little info to deconvolve).
DESTROYED anomalous signal: iodines 35→10σ, DANO RMS 13→482 (37× noise).
Cause: `I(+)`/`I(-)` are separate groups, so a per-HKL prior shrinks each Friedel
mate toward the shell mean independently — differential shrinkage swamps the tiny
difference. (Confirms the "Wilson KL on the wrong coordinate" note.)

**8b. Difference-as-latent with an empirical-Bayes prior.**
Correct split: Wilson prior only on the common `(I⁺+I⁻)/2`, a per-shell
James-Stein prior centred at 0 on `δ = I⁺−I⁻`, centrics forced `δ=0`.
No per-hemisphere corruption this time, but peaks still dropped: 18.14→17.59,
4/13 sites lost.
Cause: the anomalous map is a coherent Fourier sum — even low-per-reflection-S/N
DANO adds constructively at atoms while noise averages down.
EB shrinkage minimises per-reflection MSE, which attenuates the amplitude the
coherent sum needs.
The optimal merge-level prior on `δ` is therefore **flat** — the count merge
already is the optimal difference-as-latent treatment.

**8c. Count merge (flat δ, unbiased common).** Best at the merge level: 18.14.

## Robustness: the prefactor bug in the reported sigmas

All peak experiments used DIALS' error-model sigmas from `scaled.refl`, which
carry the inverted `calc_deltahl` prefactor (sigmas ~4% too small on 9b7c; the
refit correction is model ×1.046, DIALS ×1.037, nearly common-mode).
Applying the exact correction shifts the count-merge margin from +0.79 to +0.70:
inflating sigmas helps the noisier DIALS data slightly more, but the model still
wins on every site.
Peaks are moderately sigma-sensitive (an exaggerated ×1.3 nearly closes the
margin), so this is a stated robustness check, not a negligible effect.
Note this is a post-hoc sigma rescale; a true prefactor-corrected re-scale+merge
would also perturb the scale refinement (second order, not propagated here).

Variance source matters more than the prefactor.
Everything above used DIALS' error-model-INFLATED variances
(`intensity.scale.variance`, ~1.95x the model's posterior variance, ~1.40x for
DIALS).
Using the RAW posterior variance instead (frame-matched
`conv^2 * intensity.prf.variance`, no error model) WIDENS the margin:
model 17.96, DIALS 16.83, margin +1.13 (vs +0.79 inflated).
It deflates the model more (1.40x) than DIALS (1.18x) because the model's
posterior was more overconfident, so trusting the Bayesian posterior sharpens the
model's map more.
Caveat: a=1.38 says those raw sigmas ARE too small (residual scatter exceeds
posterior variance), so the sharper map may be over-weighting under-trusted
reflections; report significance against a calibrated noise level, not raw z.
The count-merge INTENSITIES (hence DANO) are identical across all three variance
choices -- only the map weighting / FW shrinkage changes.

## Conclusions

- `dials.scale`/`dials.merge` need no modification to accept our intensities;
  the mismatched steps are the prior-bearing and sign-bearing ones, and they
  cost nothing at the anomalous sites.
- Our integration beats DIALS on anomalous peaks (+0.85σ, robust across two
  refinement engines), because the integrator has absorbed the regularization
  DIALS spreads across French-Wilson and inverse-variance weighting.
- Free merge-level win available to everyone: the count/Poisson merge `ΣI/Σg`
  over DIALS' `1/σ²` gains ~+1σ of anomalous signal.
- The merge is the wrong place for anomalous priors — every "more Bayesian"
  merge either corrupts (per-hemisphere) or attenuates (EB) the Bijvoet signal;
  the Fourier synthesis is already the optimal denoiser.
- The remaining Bayesian leverage is at **integration** (where the model wins) or
  a joint pixels→F model where the anomalous difference couples to the real-space
  structure — not at the merge.

## Artifacts

- Port: `src/integrator/dials_port/` (verified modules + `PIPELINE.md`).
- Merged MTZs, refinements, peak CSVs: session scratchpad.
- Reproduce from `Downloads/scaling_reference/{scaled,dials_scaled}.refl`
  (DIALS- and model-integrated, both DIALS-scaled) and their `_.refl` unscaled
  counterparts.
