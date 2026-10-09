# Open work

## Integration-quality metrics without ground truth or merging

The question: what tells us a single shoebox was integrated well, before any scaling or merging, with no truth to compare against?

This matters because every metric currently in use fails that test.
CC-half, cc_anom, R-free and peak heights all need a merge.
The gate-conditioned score added in `fast_eval.py` half-merges by HKL, so it needs multiplicity and inherits scale variation between images -- it cannot judge one shoebox either.

Ranked by how much they would actually tell us.

### 1. Held-out pixel predictive likelihood (build first)

Mask a random ~10% of pixels per shoebox, fit on the rest, score `log p(c_p | I*pi_p + b_p)` on the held-out ones.

A **proper scoring rule**: it cannot be improved by shrinking sigma, inflating I, or any reparameterisation, and it tests precisely what integration claims to do -- explain the pixels as intensity times profile plus background.
Per-shoebox, so pre-merging by construction.
Nothing else on this list is as close to a sufficient summary of integration quality.

### 2. Pearson residual structure

With `lambda_p = I*pi_p + b_p` and `c_p ~ Poisson(lambda_p)`, the residuals `r_p = (c_p - lambda_p)/sqrt(lambda_p)` must have mean 0 and variance 1.

- `Var(r) > 1`: the model is too rigid, or the counts are overdispersed -- gain != 1 ADU/photon is a live possibility and would also mean every Fisher sigma in the project is off by sqrt(gain).
- Radial trend in `r`: the profile shape is wrong. This is a direct test of the Q2 question, stronger than comparing `n_eff`, because it tests the profile against the data rather than describing it.
- Trend with `I`: the intensity scale is mis-estimated with signal.

### 3. Negative-intensity fraction

A background-subtracted intensity is a scaled Skellam, so a correct integrator MUST produce negatives at I/sigma ~ 1 -- roughly 15-40%.
Our exports have **0%**.
One number, no truth required, and it already caught a real defect: it is how the posterior export was proved not to be a measurement.

### 4. Wilson-plot linearity

Bin per-observation I by resolution; `log<I>` against `sin^2(theta)/lambda^2` should be straight with slope -2B.
Curvature indicates resolution-dependent bias.
Classic, truth-free, pre-merging.

### 5. Background against the border

The predicted `b` should match a robust border-pixel median where there is no signal.
Disagreement means intensity and background are trading off, which is the identifiability collapse documented for poly merging in [[poly_merging_collapse]].

### Status

None are implemented.
1 and 2 are the real answer; 3 and 4 are cheap one-number checks worth having permanently.
All five work on an existing prediction parquet with no careless run, so they apply to all 20 ablation arms immediately.
If the simulated data lands with ground truth, these should be validated against it -- that tells us how far to trust them when truth is absent.

## Also open

- **The sigma model.** Measured `Var(z)` is 15-265 against an honest 1.0 across every ablation arm, so sigma is understated everywhere. This is the largest single defect found so far and it is upstream of every weighted metric.
- **Poly SVAE in the noise regime.** Diagnosed as the documented background/profile identifiability collapse; the three fixes (peaked warm-start, fitted background prior, structured `ChebyshevSpectrum`) arrive with the scaling merge.
- **`intensity.sum.value` disagrees with our shoebox summation by 2.5x** (corr 0.58). Either our boxes clip signal or the two integrate different regions. Upstream of everything, and unexplained.
- **Seed floor unknown.** Re-predicting one checkpoint moved IOD 204 from 20.77 to 17.23 through an identical chain. Until replicates bound this, differences below ~3 sigma are not interpretable.
