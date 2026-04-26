# Loss Models for Bayesian Integration

## Overview

Three loss functions implement the ELBO for variational inference over reflection intensities, profiles, and backgrounds. They differ in how the **intensity prior** is specified:

| Loss | Intensity Prior | BG Prior | Profile Prior | Learnable Prior Params |
|------|----------------|----------|---------------|----------------------|
| `Loss` | Single global prior | Single global | Single global | 0 |
| `PerBinLoss` | Per-bin empirical Bayes | Per-bin empirical Bayes | Per-bin empirical Bayes | 0 |
| `WilsonPerBinLoss` | Per-bin fully Bayesian Wilson | Per-bin empirical Bayes | Per-bin empirical Bayes | 4 |

All three share the same ELBO structure:

```
L = E_q[ log p(x | I, prf, bg) ]       -- Poisson NLL
  - KL( q(prf) || p(prf) )              -- profile
  - KL( q(I)   || p(I) )                -- intensity
  - KL( q(bg)  || p(bg) )               -- background
```

---

## 1. `Loss` — Single Global Prior

**File:** `src/integrator/model/loss/loss.py`

The baseline model. A single prior distribution is shared across all reflections regardless of resolution. Prior parameters are fixed from configuration (e.g., a single Gamma or Exponential for intensity).

**Limitation:** Does not account for the well-known resolution dependence of diffraction intensities. Low-resolution reflections are systematically stronger than high-resolution ones, so a single prior is a poor fit for both regimes simultaneously.

---

## 2. `PerBinLoss` — Per-Bin Empirical Bayes

**File:** `src/integrator/model/loss/per_bin_loss.py`

Reflections are grouped into resolution bins. Each bin gets its own fixed prior parameters estimated from the data:

- **Intensity:** `I ~ Exp(tau_k)` where `tau_k = 1 / mean_intensity_k` from DIALS estimates
- **Background:** `bg ~ Exp(lambda_k)` where `lambda_k = 1 / mean_bg_k` from border pixels
- **Profile:** `prf ~ Dir(alpha_k)` where `alpha_k` is fit per bin via method of moments

**Inputs (fixed buffers):**
- `tau_per_group.pt` — `(n_bins,)` Exponential rates for intensity
- `bg_rate_per_group.pt` — `(n_bins,)` Exponential rates for background
- `concentration_per_group.pt` — `(n_bins, 441)` Dirichlet concentrations for profile

**Limitation:** The intensity prior `tau_k` is derived from DIALS intensity estimates, creating a circularity — we use intensity estimates to set the prior on the quantity we're trying to estimate.

---

## 3. `WilsonPerBinLoss` — Fully Bayesian Wilson Intensity Prior

**File:** `src/integrator/model/loss/wilson_per_bin_loss.py`

Replaces the empirical intensity prior with a physics-based Wilson distribution whose hyperparameters are learned through variational inference. Background and profile priors remain per-bin empirical Bayes.

### The Wilson Model

The Wilson distribution describes how average diffraction intensity falls off with resolution:

```
Sigma_k = K * exp(-2B * s_k^2)
```

where:
- `K` — overall scale factor (global)
uata_dir = Path("/Users/luis/from_harvard_rc/")
data = {x.stem: x for x in list(data_dir.glob("*9b7c/*.pt"))}
- `B` — Wilson B-factor in Angstrom^2 (global), describes thermal motion / disorder
- `s_k^2 = 1 / (4 * d_k^2)` — resolution parameter per bin, purely geometric

The intensity prior rate for bin k is:

```
tau_k = 1 / Sigma_k = (1/K) * exp(2B * s_k^2)
```

This reduces `n_bins` free parameters to just 2 physically meaningful ones.

### Generative Model

```
log K ~ Normal(mu_K, sigma_K)           -- hyperprior (fixed, broad)
log B ~ Normal(mu_B, sigma_B)           -- hyperprior (fixed, broad)

For each resolution bin k:
    tau_k = (1/K) * exp(2B * s_k^2)     -- Wilson formula (deterministic given K, B)
    I   ~ Gamma(1, tau_k)               -- = Exp(tau_k)
    prf ~ Dir(alpha_k)                  -- empirical Bayes (fixed)
    bg  ~ Exp(lambda_k)                 -- empirical Bayes (fixed)
    counts ~ Poisson(I * prf + bg)
```

### Variational Posterior

```
q(log K) = Normal(mu_K_tilde, sigma_K_tilde)    -- 2 learnable scalars (loc, log_scale)
q(log B) = Normal(mu_B_tilde, sigma_B_tilde)    -- 2 learnable scalars (loc, log_scale)
q(I | x)   -- amortized, from encoder network
q(prf | x) -- amortized, from encoder network
q(bg | x)  -- amortized, from encoder network
```

### ELBO

```
L = E_q[ log p(x | I, prf, bg) ]                                    -- NLL
  - KL( q(prf) || Dir(alpha_k) )                                    -- profile (per-bin EB)
  - E_{q(K,B)}[ KL( q(I) || Gamma(1, tau_k(K,B)) ) ]               -- intensity (MC over K,B)
  - KL( q(bg) || Exp(lambda_k) )                                    -- background (per-bin EB)
  - [ KL(q(log K) || p(log K)) + KL(q(log B) || p(log B)) ] / N    -- hyperprior (amortized)
```

The intensity KL has a Monte Carlo outer loop (controlled by `n_wilson_samples`) because `tau_k` depends on the sampled hyperparameters K, B. The hyperprior KL is divided by dataset size N since K and B are global parameters shared across all observations.

### Key Properties

- **No circularity:** The only intensity-related input is `s_squared_per_group`, which is purely geometric (computed from d-spacings via unit cell + Miller indices). No DIALS intensity estimates are needed.
- **`tau_per_group` is optional:** If provided, it is used only to warm-start the variational parameters via linear regression of `log(tau)` on `s^2`. It does not define the prior.
- **Global B-factor:** The Wilson B is a property of the crystal as a whole, not per-atom. Per-atom B-factors are a downstream refinement quantity.
- **SBC compatible:** The per-bin formulation allows simulation-based calibration because the data-generating prior and model prior operate at the same granularity.

### Inputs

- `s_squared_per_group.pt` — `(n_bins,)` resolution parameter, purely geometric
- `bg_rate_per_group.pt` — `(n_bins,)` Exponential rates for background (empirical Bayes)
- `concentration_per_group.pt` — `(n_bins, 441)` Dirichlet concentrations (empirical Bayes)
- `tau_per_group.pt` — `(n_bins,)` **optional**, for initialization only

### Configuration

```yaml
loss:
  name: wilson_per_bin
  args:
    mc_samples: 100
    eps: 0.00001
    s_squared_per_group: s_squared_per_group.pt
    bg_rate_per_group: bg_rate_per_group.pt
    concentration_per_group: concentration_per_group.pt
    # tau_per_group: tau_per_group.pt  # optional, for init only
    hp_log_K_scale: 3.0               # hyperprior breadth on log K
    hp_log_B_scale: 1.0               # hyperprior breadth on log B
    n_wilson_samples: 4               # MC samples over (K, B)
    pprf_weight: 1.0                  # set to 1.0 for proper ELBO
    pbg_weight: 1.0
    pi_weight: 1.0
```

### Diagnostics

```python
loss.posterior_means()
# {'K_mean': 1.23, 'B_mean': 28.5, 'K_std': 0.05, 'B_std': 1.2}
```

---

## Comparison Strategy

All three losses can be evaluated on the same simulated dataset (same `counts.pt`, `masks.pt`, `reference.pt`). Only the loss section of the YAML config changes. This enables direct comparison of:

1. **Global prior** — baseline
2. **Per-bin empirical Bayes** — upper bound (uses oracle information from DIALS)
3. **Per-bin Wilson** — physics-based, no circularity, learns from data

For proper posterior calibration, KL weights must all be 1.0. Any weight != 1.0 produces a tempered ELBO that is not interpretable as a variational bound.
