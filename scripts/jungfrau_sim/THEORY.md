# Theory notes: JUNGFRAU detector physics and the likelihood question

Companion to `README.md`, which states the results.
This note explains the physics, the derivations behind each likelihood, and why the study is built the way it is.
It is written to be read start-to-finish by someone who has not thought about detector electronics before.

Everything numeric here is either cited to a source or reproduced by `selftest.py`.

---

## 1. What a JUNGFRAU actually does

### 1.1 Photon to charge

A 12.4 keV X-ray photon is absorbed in the silicon sensor and dumps its energy into electron-hole pairs.
Silicon needs about 3.6 eV per pair, so one photon liberates

$$N_{e^-} = \frac{12400 \text{ eV}}{3.6 \text{ eV}} \approx 3444 \text{ electrons.}$$

The literature quotes "~3500 e⁻ for a 12.4 keV photon", which is the same number.
This is the single most important quantity in this whole study, and section 5 explains why.

### 1.2 Charge to voltage: integration, not counting

Here is the conceptual fork.

A **photon-counting** detector (PILATUS, EIGER) puts a discriminator on every pixel.
Each photon produces a pulse; if the pulse clears a threshold, a counter increments.
The output is an integer *by construction*, and the detector has literally counted photons.

A **charge-integrating** detector (JUNGFRAU) has no discriminator and no counter.
Each pixel has a charge-sensitive preamplifier that simply accumulates whatever charge arrives during the exposure.
At the end of the exposure, an ADC digitizes the accumulated voltage.

So on a JUNGFRAU, **integer photon counts exist in the physics but never in the electronics**.
Nothing in the readout chain ever represents "3 photons".
It represents "an amount of charge that happens to be about 3 × 3444 electrons".

This is the origin of the question this study answers.
If the detector never counted, can we recover the count?

### 1.3 The three gain stages

Integrating charge has a dynamic range problem.
An amplifier sensitive enough to resolve one photon saturates almost immediately on a bright Bragg peak.

JUNGFRAU solves this with three amplification settings per pixel, selected **automatically, per pixel, per shot** by a comparator that watches the accumulating charge:

| stage | role | saturates near |
|---|---|---|
| G0 | high gain, single-photon sensitive | ~25 photons at 10 keV |
| G1 | medium gain | ~800 photons |
| G2 | low gain | 8000–10000 photons |

The pixel starts in G0.
If the charge approaches G0 saturation, the comparator switches that pixel to G1, and later to G2.
The gain stage is therefore a **discrete latent variable that depends on the signal itself** — which is why the noise on a JUNGFRAU pixel is not just heteroscedastic but signal-dependent through a switch.

For this study that turns out to matter less than you would think, for a reason worth stating early: **weak data never leaves G0**.
The Poisson-vs-Normal distinction only bites at low counts, and low counts are exactly where G0 is in charge.
G1 and G2 only engage above ~25 photons, where Poisson noise ($\sqrt{25} = 5$) dwarfs any read noise.

### 1.4 The 16-bit word

The readout packs each pixel into 16 bits:

```
 bit  15 14 | 13 12 11 10 9 8 7 6 5 4 3 2 1 0
      gain  |            ADU (14 bits)
```

The top 2 bits say which stage fired: `00` = G0, `01` = G1, `11` = G2.
The bottom 14 bits are the ADC value, so ADU runs 0–16383.

psana decodes this with cuts at `1<<14`, `2<<14`, `3<<14`, which is the same thing expressed as thresholds on the raw integer.
`detector.py` packs with `adu | (bits << 14)` and unpacks with `raw & 0x3fff` / `raw >> 14`.

### 1.5 Pedestal and gain, explained

These are the two constants you asked about, and they are simply the **intercept and slope** of a straight line.

For a given pixel in a given gain stage, the readout is linear in deposited energy:

$$\text{ADU} = \underbrace{\text{pedestal}}_{\text{intercept}} + \underbrace{\text{gain}}_{\text{slope}} \times \text{energy}$$

**Pedestal** is what the pixel reads with *zero* photons.
It is not zero, because the amplifier chain has a DC baseline — an offset that exists whether or not anything hit the sensor.
It is *per-pixel* (every amplifier is slightly different) and *per-gain-stage* (each stage has its own baseline), so a module's pedestal array has shape `(3, 512, 1024)`.

It is measured from **dark frames**: close the shutter, read out, average.
For G1 and G2 this needs a dedicated run with the detector *forced* into those stages, since a dark pixel would never switch there on its own.
For G0 the trick is to start acquiring a few seconds before the shutter opens and use the leading dark frames.

**Gain** is the slope in ADU per keV.
It is calibrated at the factory by PSI and shipped as a constant, and it too is per-pixel and per-stage.
The psana defaults are:

| stage | gain (ADU/keV) | ADU per 12.4 keV photon |
|---|---|---|
| G0 | +41.5 | +515 |
| G1 | −1.39 | −17.2 |
| G2 | −0.11 | −1.4 |

**The negative signs are real and are not a typo.**
In G1 and G2 the digitized code *decreases* as the signal increases — a brighter pixel reads a *lower* number.
PSI ships the constants with the sign baked in ("Gain files in units of [ADU/keV] are supplied by PSI and account for correct scale orientation sign"), so that one formula works for all three stages without special-casing.

A practical consequence, and the reason `detector.py` uses pedestals `(2000, 15000, 15000)`: in G0 the pedestal sits near the *bottom* of the ADU range and signal counts *up* toward the rail at 16383; in G1/G2 the pedestal sits near the *top* and signal counts *down* toward 0.

I did not have published per-pixel pedestal constants, so I chose those three values and then *checked* them: the headroom between each pedestal and its rail implies a saturation point of 28 / 870 / 10997 photons, against the published 25 / ~800 / 8000–10000.
`selftest.py` asserts this. The agreement is a real (if modest) check that the model's geometry is right.

### 1.6 The calibration inverse

Undoing the line is trivial arithmetic — psana literally computes

$$\text{energy} = \frac{\text{code} - \text{pedestal} - \text{offset}}{\text{gain}}$$

and then photon-equivalents are `energy / 12.4 keV`.

That division is where the "data live on the real line" observation comes from, and it is worth being precise about *why*.
The code is an integer. The pedestal is a **float**. The gain is a **float**.
Integer minus float divided by float is a float — and on a background pixel it is frequently **negative**, because the noise is symmetric around a pedestal that represents zero photons.

The study measures this: at a background rate of 0.5 photons/pixel, **29.2%** of pixels calibrate to a negative value.
That is not a bug and not a detector fault — it is exactly what the model predicts.

The naive prediction is $P(N=0) \times \tfrac{1}{2} = e^{-0.5}/2 = 0.303$: the pixels that got no photon, half of which land below the pedestal by chance.
That is close but measurably wrong (the simulation says 0.292, and with 200k pixels the standard error is 0.001, so 0.303 is off by 11σ).

The missing piece is a preview of §5: **the ADU is itself an integer**, so it has its own rounding deadband.
An $N=0$ pixel only reads negative if the noise pushes the *code* below $-0.5$ ADU, not merely below $0$:

$$P(x < 0) = P(N=0) \cdot \Phi\!\left(\frac{-0.5}{\sigma_{\text{adu}}}\right) = e^{-0.5} \cdot \Phi\!\left(\frac{-0.5}{12.4}\right) = 0.2935$$

which matches the simulation.
Even here, at the very first quantization in the chain, a rounding deadband is quietly eating part of the noise.

---

## 2. Four reasons the output is not an integer

Worth enumerating, because they have very different sizes:

1. **Read noise.** Electronic noise in the amplifier and ADC. Gaussian, zero-mean. **~0.024 photons in G0.**
2. **Pedestal error.** The pedestal you subtract is not the pedestal that applied, because it drifts thermally between the dark run and the exposure. A *bias*, not noise. **~0.19 photons** (up to 100 ADU, measured).
3. **ADC quantization.** The code is an integer, so photon-equivalents come in steps of $1/515 = 0.002$ photons in G0. Negligible.
4. **Charge sharing.** A photon absorbed near a pixel boundary spreads its charge cloud across 2–4 pixels, so each gets a *fraction*. **This one is different in kind**: it means the per-pixel truth is not an integer *even before any noise*. It is not modelled here (see §9).

Note the ordering: the *bias* (#2) is roughly **8× larger** than the *noise* (#1).
Nearly everyone's intuition about detectors is anchored on read noise.
For JUNGFRAU that intuition points at the wrong term.

---

## 3. The statistical model

### 3.1 The generative chain

For one shoebox with $P$ pixels, intensity $I$, normalized profile $p$ (with $\sum_p p_p = 1$), and flat background $B$:

$$\lambda_p = I \, p_p + B$$
$$N_p \sim \text{Poisson}(\lambda_p)$$
$$\text{ADU}_p = \text{ped}[g] + \text{gain}[g] \cdot N_p E_\gamma + \varepsilon_p, \qquad \varepsilon_p \sim \mathcal{N}(0, \sigma_{\text{adu}}[g])$$

Calibrating back to photon-equivalents and writing $\sigma$ for the read noise *in photon units*:

$$x_p = N_p + \varepsilon_p, \qquad \varepsilon_p \sim \mathcal{N}(0, \sigma), \qquad N_p \sim \text{Poisson}(\lambda_p)$$

So after calibration, the detector's entire effect collapses to: **add Gaussian noise of 0.024 photons to a Poisson count**.
That is the whole problem, and stated that way the answer is nearly obvious.

### 3.2 The compound distribution

$x$ is a Poisson count *plus* Gaussian noise, so its density is a **convolution** — an infinite mixture of Gaussians, one per possible count, weighted by the Poisson pmf:

$$p(x \mid \lambda, \sigma) = \sum_{n=0}^{\infty} \underbrace{\frac{\lambda^n e^{-\lambda}}{n!}}_{\text{Poisson}(n;\lambda)} \cdot \underbrace{\frac{1}{\sigma\sqrt{2\pi}} e^{-(x-n)^2 / 2\sigma^2}}_{\mathcal{N}(x;\, n,\, \sigma)}$$

Two moments fall out immediately and are worth memorizing:

$$\mathbb{E}[x] = \lambda, \qquad \text{Var}[x] = \underbrace{\lambda}_{\text{Poisson}} + \underbrace{\sigma^2}_{\text{read}}$$

The variance is by the law of total variance: $\text{Var}(x) = \mathbb{E}[\text{Var}(x|N)] + \text{Var}(\mathbb{E}[x|N]) = \sigma^2 + \lambda$.

This distribution is the simulator's own law, so its likelihood is the **ceiling**: no estimator can do better, and every other rung is measured against it.
It is `exact` in `likelihoods.py`, computed by truncating the sum and using `logsumexp`.

`selftest.py` verifies it three ways: it integrates to 1.0 (to $10^{-8}$), its mean and variance match $\lambda$ and $\lambda + \sigma^2$, and as $\sigma \to 0$ it collapses onto the Poisson pmf to $4.4 \times 10^{-16}$.

---

## 4. The likelihood ladder

Each rung is a different claim about what happened to $\lambda$ on the way out.

### 4.1 `normal_free` — $\mathcal{N}(x; \lambda, s^2)$, $s$ fitted

The naive default: assume Gaussian noise of unknown constant size.
This is what you get by reaching for an MSE loss, or by putting a free `sigma` head on a decoder.

Its sin is **discarding the mean-variance coupling**.
Poisson data have $\text{Var} = \text{mean}$ — bright pixels are intrinsically noisier than dim ones.
A single fitted $s^2$ cannot express that. §7 shows exactly what it costs.

### 4.2 `normal_coupled` — $\mathcal{N}(x; \lambda, \lambda + \sigma^2)$

The *correct* Gaussian: plug the true variance from §3.2 in rather than fitting it.
This is the best a Normal likelihood can do, and it is a genuinely good model — it is right about both moments.
It is only wrong about *shape*, since a Poisson at $\lambda \approx 0.5$ is discrete and skewed and nothing like a Gaussian.

Note it needs no free parameters: $\sigma$ is a known detector constant and $\lambda$ is the thing you are fitting.

### 4.3 `poisson_counts` — $\text{Poisson}(\text{round}(\text{clamp}(x, 0)); \lambda)$

The integer route.
Round the calibrated value, then use the honest discrete likelihood.

It makes one **implicit and untrue** assertion: that $\sigma = 0$.
It treats the rounded value as *the* count, with no acknowledgement that read noise ever existed.
The clamp is a second sin, censoring the 29% of background pixels that came out negative.

Both sins turn out to be free, for the reason in §5.

### 4.4 `exact` — the convolution of §3.2

The ceiling. $O(n_{\max})$ per pixel, which is why the study keeps it to weak intensities.

### 4.5 `gat` — generalized Anscombe

A classical trick: apply a nonlinear transform that makes Poisson-ish noise approximately Gaussian with **constant variance 1**, then use a plain Normal.

$$\text{GAT}(x) = 2\sqrt{x + \tfrac{3}{8} + \sigma^2} \;\; \approx \;\; \mathcal{N}\!\left(2\sqrt{\lambda + \tfrac{3}{8} + \sigma^2},\; 1\right)$$

The $\sqrt{\cdot}$ is the variance-stabilizing transform for a Poisson (since $\text{Var} = \text{mean}$, the derivative $\propto 1/\sqrt{\lambda}$ exactly cancels the growth in spread).
The $3/8$ is the Anscombe correction; the $+\sigma^2$ is the *generalization* that handles added Gaussian read noise.

It is included because the integrator's data module already uses the Anscombe transform, so it is the closest thing to what the codebase does today.
The study finds it carries a **5–10% low bias on $I$** — the approximation degrades below $\lambda \approx 1$, which is precisely the background regime.

---

## 5. The deadband argument

This is the theoretical core of the whole study, and it is a one-line calculation.

### 5.1 Why the conversion is lossless

We want $\text{round}(x) = N$.
Since $x = N + \varepsilon$ and $N$ is an integer, rounding returns $N$ **iff** $|\varepsilon| < 0.5$.

Rounding has a **±0.5 photon deadband**, and anything that happens inside it is erased.

So the probability of getting the count wrong is just a Gaussian tail:

$$P(\text{round}(x) \neq N) = P(|\varepsilon| > 0.5) = 2\,\Phi\!\left(\frac{-0.5}{\sigma}\right)$$

Now put the numbers in. With $\sigma = 0.024$:

$$\frac{0.5}{0.024} = 20.8 \text{ standard deviations} \implies P \approx 10^{-96}$$

**That is the answer to "can we convert to integers".**
The read noise is a 40σ-resolved 0.024 photons; the deadband is 0.5 photons.
The noise is not merely smaller than the deadband, it is *twenty standard deviations* inside it.
You would not observe a single misrounded pixel in the lifetime of the universe.

There is a refinement for $N = 0$ pixels, where the clamp *helps*: any $\varepsilon < 0.5$ (including all negative $\varepsilon$) rounds and clamps to 0, which is the true count.
So $N=0$ fails only on the upper tail, giving the mixed rate

$$P(\text{fail}) = \Phi\!\left(\frac{-0.5}{\sigma}\right)\Big[2 - e^{-\lambda}\Big]$$

This closed form reproduces the simulation across the whole noise sweep:

| σ (photons) | theory %exact | simulated %exact |
|---|---|---|
| 0.024 | 100.000% | 100.00% |
| 0.150 | 99.940% | 99.94% |
| 0.300 | 93.341% | 93.36% |
| 0.500 | 77.892% | 77.67% |
| 0.800 | 62.936% | 62.46% |

The simulation is not a black box — a one-line formula predicts it.

### 5.2 Why rounding also kills the pedestal bias

Now the part that inverts the usual intuition.

Add a pedestal error $b$, so $x = N + b + \varepsilon$.
Rounding still returns $N$ iff $|b + \varepsilon| < 0.5$:

$$P(\text{round}(x) \neq N) = \Phi\!\left(\frac{-0.5 + b}{\sigma}\right) + \Phi\!\left(\frac{-0.5 - b}{\sigma}\right)$$

The deadband does not care whether the perturbation is noise or bias.
It erases **anything** smaller than half a photon.

With the measured drift $b = 0.194$ and $\sigma = 0.024$, the surviving tail is $\approx 10^{-37}$.
Even at the long-integration noise ($\sigma = 0.058$, 840 µs) with the full drift, it is $\approx 10^{-7}$ — one pixel in ten million.

So **rounding is a bias-immunity mechanism, not merely a lossless one**.
And the contrast with the Normal route is stark, because a Normal likelihood has no deadband: it sees the full $b$ on every pixel.
On a 507-pixel shoebox that per-pixel bias lands almost entirely in the background term, where it is multiplied by the pixel count.
The study measures exactly this:

```
pedestal error = 0 ADU            pedestal error = 100 ADU (+0.194 ph/px)
  normal_coupled  bias +0.22%       normal_coupled  bias -6.93%
  poisson_counts  bias +0.85%       poisson_counts  bias +0.85%   <- unchanged
```

This is the most useful thing in the study, because pedestal drift is the **dominant systematic** on a real JUNGFRAU (§2), and the integer route is immune to it *for free*.

### 5.3 Where it breaks

Rounding stops being free once $\sigma$ becomes comparable to the deadband — around $\sigma \gtrsim 0.3$, i.e. **~15× worse than a real G0 pixel**.

Above that, something initially confusing happens: `poisson_counts` picks up *lower* bias and RMSE than `exact` (at $\sigma = 0.5$: $-2.10\%$ / 80.1% vs $+7.35\%$ / 86.7%).
How can anything beat the true likelihood?

Because MLE optimality is a statement about **asymptotically unbiased** estimators.
Rounding is a nonlinear hard-thresholding operation — a denoiser — and like any shrinkage estimator it can buy MSE by trading bias.
It is destroying Fisher information while *improving* MSE, and both are true at once.

This is not a regime any real JUNGFRAU G0 pixel occupies, so it does not change the recommendation.
It is here to locate the boundary.

---

## 5.4 The information-theoretic answer: two currencies, and they disagree

§5 argued rounding is safe in G0 from the deadband. That argument does *not* extend to G1/G2, where $\sigma$ is 0.72 and 9.1 photons — far outside the deadband. So does the integer route collapse there?

No, and the reason is the most interesting thing in this study. It needs **two different information measures**, because "recover the count" and "estimate the intensity" are different goals and they come apart.

- $H(N \mid x)$ in **bits** — how much uncertainty about the latent count survives. This is the *recovery* currency. $H(N|x) \approx 0$ means $x$ pins $N$ down, so `round(x)` is a sufficient statistic.
- $\mathcal{J}(\lambda)$, **Fisher information** about the rate. This is the *estimation* currency: Cramér–Rao gives $\text{Var}(\hat\lambda) \geq 1/\mathcal{J}$, so losing Fisher information means wider error bars no matter how clever the estimator.

`information.py` computes both by direct numerical integration (validated against $\mathcal{J} \to 1/\lambda$ and $H(N|x) \to 0$ as $\sigma \to 0$):

| stage | λ | σ | $\mathcal{J}_{\text{exact}}/\mathcal{J}_{\text{Poisson}}$ | $\mathcal{J}_{\text{round}}/\mathcal{J}_{\text{exact}}$ | $H(N)$ | $H(N\mid x)$ | count recoverable? |
|---|---|---|---|---|---|---|---|
| G0 | 0.5 | 0.024 | 1.0000 | 1.0000 | 1.338 | 0.000 | **yes** |
| G0 | 20 | 0.024 | 1.0000 | 1.0000 | 4.202 | 0.000 | **yes** |
| G1 | 40 | 0.719 | 0.9872 | 0.9980 | 4.705 | 1.563 | **no** |
| G1 | 300 | 0.719 | 0.9983 | 0.9997 | 6.161 | 1.571 | **no** |
| G2 | 1000 | 9.091 | 0.9237 | 0.9999 | 7.030 | 5.174 | **no** |
| G2 | 5000 | 9.091 | 0.9837 | 1.0000 | 8.191 | 5.220 | **no** |

Read the two right-hand blocks against each other:

**Rounding costs essentially no Fisher information in any stage** — worst case 0.2%, in G1. Yet **the count is only recoverable in G0**; by G2 you have lost 5.2 bits of it forever.

Both are true because they answer different questions. In G2 you genuinely cannot say whether a pixel held 1000 or 1003 photons — but you don't care, because the Poisson noise is already $\sqrt{1000} = 32$. The quantization error of rounding has variance $1/12 = 0.083$, utterly negligible against a total variance of $\lambda + \sigma^2 = 1083$.

So **rounding is lossless in two opposite limits, for opposite reasons**:

- $\sigma \ll 0.5$ (**G0**): the noise is inside the deadband, so `round(x) = N` *exactly*. Rounding recovers the count.
- $\sigma \gg 1$ (**G2**): quantization is negligible against the noise. Rounding does not recover the count and does not need to.

The worst case is the **middle**, $\sigma \approx 0.5$, where rounding neither recovers the count nor is negligible. On a JUNGFRAU that middle is G1 ($\sigma = 0.72$) — and even there it costs 0.2%.

The column that *does* show a real cost is $\mathcal{J}_{\text{exact}}/\mathcal{J}_{\text{Poisson}}$ in G2: **7.6%** at $\lambda=1000$. That is the read noise itself, not the rounding — it is exactly $\lambda/(\lambda+\sigma^2) = 1000/1083 = 0.923$. No likelihood can recover it; the information was destroyed in the amplifier.

### So: pre-convert and assume Poisson?

**Yes.** And note it is really *two* decisions, only one of which was ever in doubt:

1. **Round the data.** Free everywhere — ≤0.2% of Fisher information, in every stage.
2. **Then assume Poisson** (i.e. assert $\text{Var}=\lambda$ when the truth is $\lambda+\sigma^2$). This is *exact* in G0 and wrong-but-negligible elsewhere: $\sigma^2$ is a 1.3% correction at G1 ($\lambda=40$) and 8% at G2 ($\lambda=1000$) — to a variance that, by then, nothing depends on.

The obvious refinement — route G0 to Poisson and G1/G2 to $\mathcal{N}(\lambda, \lambda+\sigma^2)$, which is near-exact in each regime — is implemented as the `hybrid` rung. **It buys nothing measurable.** `study.py --only gain_stages` ties it against plain `poisson_counts` and `exact` at intensities populating all three stages:

```
I_true = 200000   peak ~6806 ph   G0/G1/G2 = 57.5/28.5/14.0%   counts exact 72.63%
  normal_coupled   bias -0.01%   rmse 0.25%   z-std 1.13   cov95 0.900
  poisson_counts   bias -0.01%   rmse 0.26%   z-std 1.14   cov95 0.893
  hybrid           bias -0.01%   rmse 0.25%   z-std 1.12   cov95 0.900
  exact            bias -0.01%   rmse 0.25%   z-std 1.12   cov95 0.907
```

Only 72.63% of counts are recovered exactly, 14% of pixels are in G2 — and **every rung agrees to 0.01% bias**. The likelihood choice has stopped mattering, because by the time a pixel reaches G1 the reflection is bright enough that nothing else matters either.

That is the whole result in one line: **the likelihood choice only matters for weak data, and weak data is always in G0, where Poisson-on-rounded is exact.** The bright regime where the integer route degrades is precisely the regime where no choice matters.

(`hybrid` is kept in the ladder as the null result. It is the obvious idea and it is measurably unnecessary.)

## 6. How the estimation works

### 6.1 MLE with the profile held known

The study fits **only $(I, B)$**, with the profile $p$ given.

This is deliberate and is the main design decision.
A real integrator learns the profile, the encoder, and the intensity together, so if you compared likelihoods there, any difference could be the likelihood, the amortization gap, the optimizer, or profile misspecification.
Holding the profile known makes the comparison **attributable**: two rungs differ *only* in their likelihood, so any gap is caused by the likelihood and nothing else.

Parameterization is $I = e^{\log I}$, $B = e^{\log B}$, which keeps $\lambda > 0$ (required by Poisson) and is shared across all rungs so no rung gets an advantage.

Optimization is batched LBFGS over the summed NLL.
Shoeboxes are independent, so optimizing the sum over per-box parameters is *exactly* optimizing each box separately — no approximation, just vectorization.

### 6.2 Error bars from the observed information

$\sigma_I$ comes from the curvature of the NLL at the optimum (the observed information):

$$\hat{\sigma}_I = \sqrt{\left[\mathbf{H}^{-1}\right]_{II}}, \qquad \mathbf{H} = \nabla^2_{(I,B)} \left(-\log \mathcal{L}\right)\Big|_{\hat{I}, \hat{B}}$$

The same independence makes the full Hessian **block-diagonal**, one 2×2 block per shoebox.
That is why `fit()` can extract every block with two grad-of-grad passes rather than building an enormous matrix: differentiating $\sum_i \partial\mathcal{L}/\partial I_i$ with respect to $I$ picks out the diagonal, because box $i$'s gradient depends only on box $i$'s parameters.

The 2×2 inverse is done in closed form, and the curvature is taken in natural $(I, B)$ coordinates rather than log coordinates so $\sigma_I$ is directly on the intensity scale.

### 6.3 The diagnostics, and why calibration is the real test

Per cell the study reports four numbers:

- **bias%** — is $\hat{I}$ centered on the truth?
- **rmse%** — how far off is it typically?
- **z-std** — the standard deviation of $z = (\hat{I} - I)/\hat{\sigma}_I$. **Should be 1.**
- **cov95** — fraction of boxes with $|z| < 1.96$. **Should be 0.95.**

The last two are the ones that matter, and they are why the study bothers with the Hessian at all.

A likelihood can produce a perfectly good point estimate while lying about its uncertainty.
`normal_free` at $I=200$ has a bias of $-0.38\%$ — excellent — and a z-std of 2.10, meaning its error bars are **2.1× too small** and its "95% interval" actually covers 63% of the time.

For crystallography that is the whole ballgame.
Everything downstream — merging weights, scaling, $I/\sigma$ cutoffs, refinement targets — consumes $\sigma_I$ and trusts it.
A likelihood that is right about $I$ and wrong about $\sigma_I$ will silently corrupt all of it, and you will not see it in a plot of $\hat{I}$ vs truth.

---

## 7. Why `normal_free` is overconfident

Worth deriving, because the mechanism is general and the direction is predictable from theory alone.

Fisher information for $I$ under a Gaussian with pixel variance $v_p$, using $\partial \lambda_p / \partial I = p_p$:

$$\mathcal{J}_I = \sum_p \frac{p_p^2}{v_p}$$

The two models disagree about $v_p$:

- truth: $v_p = \lambda_p$ (each pixel's own variance)
- `normal_free`: $v_p = s^2$ for all $p$, where the fitted $s^2$ converges to roughly the *average* pixel variance

At $I = 200$, $B = 0.5$ on the study's 507-pixel shoebox, that average is $s^2 \approx 0.894$ — dominated by the many background pixels at $\lambda = 0.5$.
But the peak pixel has $\lambda = 7.31$.

So `normal_free` weights the peak pixel by $1/0.894$ when it deserves $1/7.31$: an **8.2× overweight**.

And this is maximally damaging, because the weights $p_p^2$ in $\mathcal{J}_I$ are *largest exactly where the overweighting is worst* — the peak pixels are simultaneously the most informative about $I$ and the most over-trusted.
Summing up:

$$\frac{\mathcal{J}_I^{\text{homoscedastic}}}{\mathcal{J}_I^{\text{true}}} = 3.83 \implies \hat{\sigma}_I \text{ understated by } \sqrt{3.83} = 1.96\times$$

The study measures **2.10×**.
Theory and simulation agree, so this is a mechanism, not a quirk.
(The residual gap is because $s^2$ is fitted rather than exactly the mean, and the log parameterization contributes a little.)

The general lesson: **a free-variance Gaussian on count data does not just lose efficiency, it manufactures false confidence**, and it does so worst on the brightest, most important pixels.

---

## 8. The research findings, and where each number came from

| quantity | value | source |
|---|---|---|
| e-h pair energy in Si | 3.6 eV | standard |
| electrons per 12.4 keV photon | 3444 (~3500) | derived / Struct. Dyn. |
| G0 read noise, 10 µs | 83 e⁻ RMS = **0.024 ph** | Struct. Dyn. 7, 014305 |
| G0 read noise, 840 µs | 200 e⁻ RMS = **0.058 ph** | Struct. Dyn. 7, 014305 |
| pedestal thermal drift | up to 100 ADU = 2.5 keV = **0.19 ph** | Struct. Dyn. 7, 014305 |
| gains G0/G1/G2 | +41.5 / −1.39 / −0.11 ADU/keV | psana defaults |
| ADU per photon, G0 | 515 | derived |
| gain bit encoding | 00 / 01 / 11 | XFEL docs, psana |
| calibration formula | `(code − ped − offset)/gain` | psana |
| dynamic ranges | 25 / ~800 / 8000–10000 ph | Struct. Dyn., XFEL docs |

The single most load-bearing finding is the **ratio** 0.024 : 0.19 : 0.5 (noise : drift : deadband).
Everything in §5 follows from it.

A useful sanity anchor: the `jungfrau-photoncounter` project exists specifically to convert JUNGFRAU data to photon counts.
Rounding to integers is not an exotic proposal, it is what the community already does — this study explains *why* it is safe and quantifies when it stops being.

---

## 9. What is modelled, what is assumed, what is missing

**Anchored to measurements:** G0 read noise; the gain constants; the drift magnitude; the bit encoding; the calibration formula.

**Modelled choices, documented at their definitions:**

- *Pedestals* `(2000, 15000, 15000)` ADU are not real per-pixel constants — real ones are per-pixel arrays from dark runs. They are chosen so the headroom reproduces the published dynamic ranges, which `selftest.py` checks (28 / 870 / 10997 vs published 25 / ~800 / 8000–10000).
- *G1/G2 read noise* is modelled as ADC-dominated (constant in ADU), absent published per-stage figures. In photon units this scales with the gain ratio: 0.024 / 0.72 / 9.1 photons. Inconsequential, because G1/G2 only engage above ~25 photons where Poisson noise dominates by orders of magnitude.

**Not modelled — and the one that could actually change the answer:**

- **Charge sharing.** A photon near a pixel boundary splits its cloud across 2–4 pixels, so pixel value $= \sum_{\text{photons}} f \cdot E_\gamma$ with $f \in [0,1]$ continuous. This means the per-pixel truth is **not an integer even before noise**, so "round to recover the count" is not merely noisy but asks for something that does not exist. It is the principled reason integer conversion can never be exactly right, and the obvious next extension (a Gaussian charge cloud rasterized onto the pixel grid, one toggle in `readout()`).

  Why it probably does not overturn the conclusion: charge sharing conserves total charge, so it redistributes *within* the shoebox rather than out of it, and the shoebox sums the profile back up. But this is an argument, not a measurement, and the extension is the way to settle it.

- Per-pixel gain/pedestal variation, gain-switching hysteresis and artifacts near the switching boundary, Fano noise in the e-h pair count (sub-Poisson, small), and the 16-memory-cell mode's degraded pedestals.

**Known caveat in the estimator, not the detector:** at $I = 2$ every rung is unusable (bias +60–90%, RMSE ~250%).
That regime is dominated by the $I = e^{\log I}$ positivity constraint, which cannot represent $I \le 0$ and so pushes $\mathbb{E}[\hat{I}]$ up.
Real weak-reflection handling wants to admit negative intensities.
The comparison there is measuring the parameterization, not the likelihood, and should not be read as a statement about either.

---

## 10. Reading the output

```
uv run python scripts/jungfrau_sim/selftest.py           # 17 checks on the model itself
uv run python scripts/jungfrau_sim/study.py              # all four studies (~15 min)
uv run python scripts/jungfrau_sim/study.py --only ladder --n 300
```

- **`selftest.py`** validates the detector and the `exact` likelihood. Run it first; if it fails nothing else means anything.
- **study 1, `integer_recovery`** — the pixel-level answer to "can we convert", with no fitting involved.
- **study 2, `ladder`** — the five likelihoods vs intensity at real G0 noise.
- **study 3, `pedestal`** — the drift result of §5.2.
- **study 4, `noise_sweep`** — locates the boundary of §5.3.

In the ladder tables: **bias%/rmse%** describe the point estimate, **z-std/cov95** describe honesty of the error bar, and `exact` is the ceiling. A rung matching `exact` is not approximating it — it *is* it.

---

## References

- [JUNGFRAU docs, European XFEL](https://rtd.xfel.eu/docs/jungfrau-detector-documentation/en/latest/general_introduction.html) — gain bit encoding, calibration procedure, pedestal measurement
- [JUNGFRAU detector for brighter x-ray sources, *Struct. Dyn.* **7**, 014305 (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7044001/) — noise figures, 3500 e⁻/photon, 100 ADU pedestal drift, dynamic ranges
- [LCLS psana JUNGFRAU calibration](https://confluence.slac.stanford.edu/display/PSDM/Jungfrau) — `(code − ped − offset)/gain`, gain defaults, gain-bit cuts
- [Fast and accurate data collection for MX using JUNGFRAU, *Nat. Methods* **15**, 799 (2018)](https://www.nature.com/articles/s41592-018-0143-7)
- [First operation in 16-memory-cell mode at European XFEL, *Front. Phys.* (2023)](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2023.1303247/full) — gain-switching region artifacts, pedestal difficulties
- [jungfrau-photoncounter](https://github.com/ComputationalRadiationPhysics/jungfrau-photoncounter) — prior art for converting JUNGFRAU data to counts
