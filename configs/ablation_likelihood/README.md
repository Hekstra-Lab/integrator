# Ablation: Negative Binomial vs Poisson pixel likelihood

Tests whether a Negative Binomial (NB) pixel likelihood beats the default Poisson
likelihood in the ELBO reconstruction term.

## What changes

The reconstruction term is a per-pixel count likelihood evaluated at the
integrator's predicted rate `rate = I * profile + background`.

- Poisson: `variance = mean`. The current default.
- Negative Binomial: `variance = mean + mean^2 / r`, with a learnable dispersion
  `r`. Recovers Poisson as `r -> inf`.
  It absorbs per-pixel counting noise *beyond* Poisson at a fixed rate -- detector
  gain (real detectors have `variance = gain * mean`, not `= mean`), pixel-to-pixel
  gain variation, and residual model mismatch.

The likelihood is a property of the loss, not the integrator: `rate` and `counts`
are produced by the integrator and passed to the loss unchanged, so the swap is a
single config knob (`loss.args.likelihood`) and touches no integrator.

Implementation: `src/integrator/model/loss/count_likelihood.py` (`CountLikelihood`),
wired into `WilsonLoss` and `GlobalPriorLoss`. Tests: `tests/test_count_likelihood.py`.

## The conjugacy caveat (which integrator to trust)

The NB dispersion parameter is a learnable scalar on the loss; it does not require
conjugacy. But the *integrator's* intensity posterior does interact with the
likelihood:

- Amortized integrators (`hierarchical`, base learned-head models) predict a Gamma
  `q(I)` from an encoder. There is no closed-form conjugate update, so nothing
  breaks -- Poisson vs NB is a clean, unconfounded swap.
  **Use `hierarchical_*` for the headline result.**

- The SVAE integrator forms `q(I)` by a closed-form Poisson-Gamma conjugate update
  (`svae_integrator.py:81-86`). That closed form is the *exact* posterior only under
  a Poisson likelihood. Under NB the ELBO is still valid, but the conjugate update
  becomes an approximate inference network mismatched to the likelihood, so an
  SVAE+NB result conflates "NB helps" with "conjugate q mismatched to NB".
  **Use `svae_*` only to probe SVAE robustness, not for the headline claim.**
  (Note the SVAE already has a learned responsibility gate `g_p`, so its update is
  not exactly conjugate even under Poisson.)

## Scale stabilization (all four configs)

All four configs set `stabilize_scale: true`. This is a separate fix for a
training instability where the Wilson scale G or B-factor collapses to its floor.

Root cause: the prior mean is a Wilson plot `log<I> = log G - 2B s^2`, with
`G = softplus(raw_G)`. A default `G ~ 1` against a true intensity scale of O(100)
means `raw_G` must travel a ~200-unit *linear* distance to recover; at `lr=1e-3`
that never happens, and the optimizer instead drives G and B into the region where
`softplus'(raw) -> 0` (dead gradient) and stalls there.

`stabilize_scale: true` fixes it, DIALS-free:

- G in log-space (`G = exp(raw_G)`), so the O(100) scale is a short multiplicative
  hop rather than a long linear trek.
- B = `b_min + softplus(raw_B)`: floored below at `b_min` (default 1 A^2, so it never
  collapses to 0) but unbounded above -- no cap to clip a large B or reintroduce a
  dead-gradient trap at the top.
- A one-time scale init on the first batch: fits G from a Wilson plot of the *raw
  shoebox counts* (summed counts above the empirical background) vs s^2. Uses only
  raw detector data + resolution -- never DIALS-integrated intensities.

In a controlled sim (true G=200, B=25, 5 seeds) the baseline collapses 5/5
(G=2.7, B=0.85); with stabilization it recovers 0/5 collapsed (G~200, B~25) and is
robust to a 3x error in the count-based init. Set consistently across all four arms
so it does not confound the NB-vs-Poisson comparison. Default `stabilize_scale: false`
reproduces the legacy softplus behavior and is checkpoint-compatible.

### Init knobs

`init_B` and `init_G` are both *physical* values -- the B factor in A^2 and the
scale G itself -- not raw pre-activation numbers. Each is inverted through whichever
parameterization is active, so `get_B()`/`get_G()` return exactly what you asked for
regardless of `stabilize_scale`.

| knob | default | meaning |
| --- | --- | --- |
| `init_B` | 30.0 | initial B factor (A^2). `B = softplus(raw_B) + b_min` |
| `init_G` | unset | initial scale G. **When set, it also pins G: the count-based Wilson fit is skipped**, so G starts exactly here |
| `b_min` | 0.0 | floor on B |
| `init_scale_from_counts` | true | run the one-time Wilson fit (only under `stabilize_scale`) |
| `wilson_init_bins` | 20 | resolution bins for that fit |

Leaving `init_G` unset keeps the count-based fit (the default for these four
configs) and falls back to a neutral `G = 1` if the fit is off. Set `init_G` when
you already know the scale -- e.g. resuming a sweep at a known G, or deliberately
testing a fixed scale -- and the data-driven estimate will not overwrite it.

Minor: the SVAE integrator reads `loss._get_tau` inside its own forward (for the
conjugate prior), which runs one step before the loss's first-batch init, so the
SVAE's prior uses the un-fitted G on batch 1 only. Amortized integrators have no
such lag.

## Configs

Each pair differs only in the `loss.args` likelihood block (stabilization is on in
all four).

| Config                     | Integrator            | Likelihood        |
| -------------------------- | --------------------- | ----------------- |
| `hierarchical_poisson.yaml`| amortized (learned)   | Poisson (control) |
| `hierarchical_nbinom.yaml` | amortized (learned)   | Negative Binomial |
| `svae_poisson.yaml`        | SVAE (conjugate)      | Poisson (control) |
| `svae_nbinom.yaml`         | SVAE (conjugate)      | Negative Binomial |

## Knobs (`loss.args`)

- `likelihood`: `poisson` | `negative_binomial`
- `nb_dispersion_init`: initial `r` (default 10.0). Larger starts nearer Poisson.
- `nb_dispersion_scope`: `global` (one shared `r`) | `per_bin` (one `r` per
  resolution bin, reusing the loss `n_bins` binning and `group_label`).

## Running (on the cluster)

```bash
uv run integrator.train --config configs/ablation_likelihood/hierarchical_poisson.yaml
uv run integrator.train --config configs/ablation_likelihood/hierarchical_nbinom.yaml
# and the svae_* pair
```

Compare on the reconstruction NLL and your downstream metrics (CC1/2, CCanom,
I/sigma, Rpim). Also log the learned dispersion `softplus(count_likelihood.raw_dispersion)`:
if it drifts large the data prefers Poisson; a small stable `r` is real overdispersion.

## Interpretation note

The MC-sampled `rate = I*profile + bg` already induces overdispersion in the
marginal counts (mixing Poisson over the Gamma `q(I)`). The NB `r` therefore
captures overdispersion *conditional on the sampled rate* -- it competes with, and
is identified separately from, the latent-intensity variance. A near-Poisson `r`
does not mean the counts are Poisson marginally; it means little extra noise remains
once `I`, `bg`, and profile are sampled.
