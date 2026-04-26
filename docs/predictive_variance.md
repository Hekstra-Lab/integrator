# Posterior Predictive Variance for Intensity

## Background

Classical integration programs (AIMLESS, XDS, DIALS) report an intensity variance
$\sigma^2(I)$ that begins as Poisson counting noise from profile fitting, then gets
inflated by an empirical error model:

$$\sigma'^2 = a^2\bigl(\sigma^2 + (bI)^2\bigr)$$

The $(bI)^2$ term absorbs systematic effects — beam fluctuations, absorption
residuals, and critically, the covariance between scale/profile/background
estimation and the intensity estimate. These programs then optionally propagate
scale-factor uncertainty (DIALS, AIMLESS) for a per-reflection correction.

## Why `qi_var` is not enough

In our variational model the approximate posterior for intensity is

$$q(I) = \mathrm{Gamma}(\alpha, \beta)$$

with variance $\alpha / \beta^2$. This is the **marginal posterior variance of the
latent intensity under the mean-field factorisation** $q(I)\,q(p)\,q(b_g)$.
Because the mean-field assumption treats $I$, $p$, and $b_g$ as independent, `qi_var`
is computed _as if the profile and background were known_ (fixed at their posterior
means). It therefore misses the additional intensity uncertainty that arises when the
profile and background are themselves uncertain.

By the law of total variance:

$$
\mathrm{Var}[I \mid \text{data}]
= \underbrace{E_{p,\,b_g}\!\bigl[\mathrm{Var}[I \mid p, b_g]\bigr]}_{\approx\;\texttt{qi\_var}}
+ \underbrace{\mathrm{Var}_{p,\,b_g}\!\bigl[E[I \mid p, b_g]\bigr]}_{\text{missing from } \texttt{qi\_var}}
$$

The second term captures how much the _point estimate_ of intensity shifts when
you resample profile and background. For strong reflections this term dominates —
exactly the regime where classical programs need the $bI^2$ inflation.

## Computing the predictive variance from MC samples

The forward pass already draws joint samples:

```
zI  ~ qi     [batch, mc_samples, 1]
zp  ~ qp     [batch, mc_samples, n_pixels]
zbg ~ qbg    [batch, mc_samples, 1]
```

and computes the Poisson rate $\lambda = z_I \cdot z_p + z_{bg}$.

For each MC sample $s$, we run the **iterative Kabsch weighted profile-fitting
estimator** (matching the existing `calculate_intensities` implementation). The
pixel-level variance $v_i$ is refined over $K$ iterations (default $K = 3$):

1. Initialise $v_i^{(s)} = z_{bg}^{(s)}$ (background-only variance).
2. For each iteration $k = 1, \ldots, K$:

$$w_i^{(s)} = \frac{z_p^{(s)}_i}{v_i^{(s)}}$$

$$\hat{I}^{(s)} = \frac{\sum_i w_i^{(s)} \left(c_i - z_{bg}^{(s)}\right)}
                       {\sum_i w_i^{(s)}\, z_p^{(s)}_i}$$

$$v_i^{(s)} \leftarrow \overline{\hat{I}^{(s)} \cdot z_p^{(s)} + z_{bg}^{(s)}}$$

where the overline denotes the pixel-mean (following the existing Kabsch
implementation). The weighting by $1/v_i$ down-weights pixels dominated by
background, improving the estimate for weak reflections.

The **posterior predictive variance** is then:

$$
\sigma^2_{\text{pred}} = \mathrm{Var}_s\!\bigl[\hat{I}^{(s)}\bigr]
$$

This single number per reflection automatically includes:

| Source of uncertainty | Captured? |
|---|---|
| Poisson counting noise (via noisy $c_i$) | Yes — counts enter the estimator |
| Profile shape uncertainty | Yes — different $z_p^{(s)}$ change the weights |
| Background uncertainty | Yes — different $z_{bg}^{(s)}$ shift the baseline |
| Intensity–profile covariance | Yes — joint samples couple the two |
| Neural-network weight uncertainty | No — would require ensembles or MC-dropout |

### Practical behaviour

- **Weak reflections**: profile is poorly constrained, so the $z_p$ samples vary
  substantially and $\sigma^2_{\text{pred}} \gg \texttt{qi\_var}$.
- **Strong reflections**: background uncertainty shifts the baseline, contributing
  an intensity-proportional variance analogous to the classical $bI^2$ term.
- **Well-measured, isolated spots**: profile and background are tight, so
  $\sigma^2_{\text{pred}} \approx \texttt{qi\_var}$.

## Implementation

The function `predictive_intensity_variance` in
`integrator/model/integrators/integrator_utils.py` computes this quantity from the
`IntegratorBaseOutputs` dataclass and injects it into the output dict under the key
`"qi_pred_var"`. It is available as a prediction key alongside `qi_mean` and
`qi_var`.

```yaml
predict_keys:
  - qi_mean       # posterior mean of intensity
  - qi_var        # posterior variance (mean-field, ignores p/bg covariance)
  - qi_pred_var   # predictive variance (marginalises over p and bg)
```
