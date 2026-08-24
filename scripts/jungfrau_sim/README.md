# JUNGFRAU likelihood study

Synthetic ground-truth study answering two questions for SFX data on a JUNGFRAU detector:

1. **Can we convert the detector's real-valued output back to integer photon counts?**
2. **If we can, does it matter which pixel likelihood we use?**

This file states the results.
**[`THEORY.md`](THEORY.md) explains the detector physics, derives each likelihood, and shows why the results come out the way they do** — start there if you want to understand rather than just use this.

The study is standalone.
It does not touch `dataset.yaml`, the data module, or any integrator.
Shoeboxes are generated with known $I$, profile and background, digitized through a simulated readout, calibrated back, and $I$ is recovered by maximum likelihood with the profile **held known**.
Only $(I, B)$ are fitted, so any gap between likelihoods is attributable to the likelihood and nothing else.

```
uv run python scripts/jungfrau_sim/selftest.py    # validate the detector + likelihood model
uv run python scripts/jungfrau_sim/study.py       # all four studies (~15 min)
uv run python scripts/jungfrau_sim/study.py --only ladder --n 300
```

## Answers

**Yes, convert to integers.**
At the real G0 read noise, `round(clamp((adu - pedestal)/gain, 0))` returns the exact true photon count for **100.00%** of pixels, and stays exact under a 200 ADU pedestal drift (double the measured worst case).
Fitting `poisson_counts` on the rounded data and fitting `exact` on the real-valued data give **identical** bias and RMSE to two decimals at every intensity tested.
Rounding is not an approximation here; it recovers the latent count.

**Use Poisson, and never use a Normal with a free variance.**
The ranking at real G0 noise is `poisson_counts == exact` > `normal_coupled` > `gat` > `normal_free`.
The point estimates are close, but the **error bars are not**: `normal_free` reports $\sigma_I$ that is 2.1x too small at $I=200$ (95% interval covers 63% of the time).
For crystallography that is the whole ballgame, since $I/\sigma$ drives downstream weighting.

## Generated datasets

`generate.py` emits 2D shoeboxes with ground-truth intensity, profile and background, each pixel given **twice**: as the detector's real-valued output and as rounded integers.

```
uv run python scripts/jungfrau_sim/generate.py --n 20000 --out data/jf_sim
uv run python scripts/jungfrau_sim/generate.py --n 20000 --pedestal-drift 100 --out data/jf_sim_drift
```

| file | shape | dtype | what it is |
|---|---|---|---|
| `counts_real.npy` | (N, 400) | float32 | **the detector's actual output.** Real line, negative on ~28% of pixels. Train a Normal likelihood on this. |
| `counts_poisson.npy` | (N, 400) | int32 | **the same pixels rounded**, `round(clamp(x, 0))`. Train a Poisson likelihood on this. |
| `counts_true.npy` | (N, 400) | int32 | the latent Poisson draw — what the detector *would* have counted. Scores the conversion directly. |
| `raw_adu.npy` | (N, 400) | uint16 | packed 16-bit words, so calibration can be redone with a different pedestal |
| `gain_stage.npy` | (N, 400) | uint8 | which stage fired per pixel |
| `profiles.npy` | (N, 400) | float32 | ground-truth normalized profile |
| `intensity.npy`, `background.npy` | (N,) | float32 | ground-truth $I$ and $B$ |
| `profile_params.npy` | (N, 5) | float32 | ground-truth $(c_x, c_y, \sigma_1, \sigma_2, \theta)$ |
| `rate_true.npy` | (N, 400) | float32 | ground-truth $\lambda = I p + B$ |

Generative model: $I \sim \text{Exp}(\text{mean}=200)$ (Wilson, acentric), $B \sim \text{Exp}(\text{mean}=0.5)$, profiles from `integrator.simulate.profiles` (elliptical 2D Gaussians, $h \sim \mathcal{N}(0, I_5)$), $N \sim \text{Poisson}(I p + B)$, then through the readout.
Each dataset is ~176 MB and `sim.json` records the full config plus integrity stats.

### What the default dataset shows

The Exponential intensity tail reaches $I = 1965$, giving peak pixels of 182 photons — which pushes 0.126% of pixels into **G1**, and that is where the whole story lives:

```
          %px  sig_read    %exact   rms err  <count>  read/Poisson
  G0   99.874%     0.024  100.000%     0.024      1.0         2.5%
  G1    0.126%     0.719   51.563%     0.718     36.7        11.9%
```

**Every misrounded pixel in the dataset is a G1 pixel.** G0 recovers the true count 100.000% of the time; G1 only 51.6%, because its read noise (0.72 photons) is larger than the ±0.5 deadband.

And it does not matter. A pixel only reaches G1 once it holds >25 photons, where Poisson noise ($\sqrt{37} = 6.1$) is 8x the read noise. Being off by ±1 on a count of 37 is noise you already had. **The integer conversion is exact exactly where precision matters, and lossy only where it doesn't.**

### The drift dataset

`jf_sim_drift` applies the measured ~100 ADU G0 pedestal drift. Same seed, so the latent truth is identical and the two are directly comparable:

- `counts_real` is shifted by **−0.194 photons on every pixel** — **−77.6 photons per shoebox**, against a mean intensity of 200. Negative pixels jump from 28% to 57%.
- `counts_poisson` is **bit-identical** to the clean dataset. 0 of 8,000,000 pixels differ.

That pair is the cleanest available demonstration of §5.2 in [`THEORY.md`](THEORY.md): the deadband erases the drift, so the integer route never sees the dominant systematic that the Normal route must absorb.

Note the drift is applied to **G0 only** by default. The published ~100 ADU is a G0 measurement (quoted as 2.5 keV = 100/41.5, the G0 gain). Whether G1/G2 pedestals drift by the same *ADU* is not settled by that figure, and it matters enormously: a fixed ADU error converts to photons through $1/\text{gain}$, so the same 100 ADU is 0.19 photons in G0 but **5.8 in G1**. Pass three values (`--pedestal-drift 100 100 100`) to explore that; it is a knob, not a constant.

## Why the detector is kinder than it looks

JUNGFRAU is charge-integrating, so integer counts exist in the physics but never in the electronics.
The natural worry is that the ADC destroys the counting information.
It does not, because the numbers are lopsided:

| quantity | photons | source |
|---|---|---|
| G0 read noise (10 µs) | **0.024** | 83 e⁻ RMS / 3444 e⁻ per 12.4 keV photon |
| G0 read noise (840 µs) | 0.058 | 200 e⁻ RMS |
| pedestal thermal drift | **0.19** | ~100 ADU, measured |
| rounding deadband | **0.5** | $\pm\tfrac{1}{2}$ photon |

One 12.4 keV photon is 515 ADU in G0 against ~12 ADU of noise — a 40σ separation between 0 and 1 photons.
Everything the detector does to a pixel is far inside the rounding deadband, which is why the conversion is lossless.

The counter-intuitive consequence is that **rounding is a bias-immunity mechanism, not just a lossless one**.
A 100 ADU pedestal drift is a per-pixel *bias*, and on a 507-pixel shoebox it lands almost entirely in the background term:

```
pedestal error = 0 ADU              pedestal error = 100 ADU (+0.194 ph/px)
  normal_coupled  bias  +0.22%        normal_coupled  bias  -6.93%
  poisson_counts  bias  +0.85%        poisson_counts  bias  +0.85%   <- unchanged
```

The Normal likelihood sees the full 0.194 ph/px bias and eats it.
Rounding maps it to zero, because 0.194 < 0.5.
So the integer route is *more* robust to the dominant systematic, not less.

## Where it breaks

Read noise would have to grow ~15x before the integer route degrades (`study.py --only noise_sweep`):

| σ_read (ph) | pixels exact | notes |
|---|---|---|
| 0.024 (real G0) | 100.00% | `poisson_counts` ≡ `exact` |
| 0.15 | 99.93% | still tracking |
| 0.30 | 93.27% | starts to peel away |
| 0.50 | 77.70% | rounding now destroys information |

Above σ ≈ 0.3 the rounded estimator starts to behave like a shrinkage estimator: it picks up lower *bias* and *RMSE* than `exact` in places, because hard rounding is a nonlinear denoiser.
That is a real effect and not a contradiction of MLE optimality — `exact` is efficient among asymptotically unbiased estimators, and rounding buys MSE by trading bias.
It is not a regime any real JUNGFRAU G0 pixel occupies, so it does not change the recommendation.

## Model and its assumptions

Forward chain in `detector.py`, one gain stage selected per pixel per shot by a comparator:

$$N \sim \text{Poisson}(\lambda), \quad
\text{adu} = \text{ped}[g] + \text{gain}[g]\cdot N E_\gamma + \varepsilon, \quad
\varepsilon \sim \mathcal{N}(0, \sigma_{\text{adu}}[g])$$

packed into 16 bits as 2 gain bits (00=G0, 01=G1, 11=G2) plus a 14-bit ADU.
Calibration inverts psana's `energy = (code - pedestal) / gain`.

Constants are the psana defaults: gains $(41.5, -1.39, -0.11)$ ADU/keV.
The negative signs are real — G1/G2 invert, so a brighter pixel reads a *lower* code.

Three things are modelled choices rather than published constants, all documented at their definitions:

- **Pedestals** `(2000, 15000, 15000)` ADU are not per-pixel calibration constants. They are picked so each stage's headroom reproduces the published dynamic ranges, which `selftest.py` verifies independently: 28 / 870 / 10997 photons against a published 25 / ~800 / 8000–10000.
- **G1/G2 read noise** is modelled as ADC-dominated (constant in ADU), absent published per-stage figures. This is inconsequential: G1/G2 only engage above ~25 photons, where Poisson noise dominates read noise by orders of magnitude. Only the G0 figure is anchored to a measurement, and only G0 matters for weak data.
- **Charge sharing is not modelled.** A photon landing near a pixel boundary splits its cloud across 2–4 pixels, so the per-pixel truth is not integer-valued even before noise. This is the one physical effect that could genuinely break the integer route, and it is the obvious next extension.

## Caveats

- At $I=2$ every likelihood is unusable (bias +60–90%, RMSE ~250%), and the comparison there is dominated by the $I = e^{\log I}$ positivity constraint rather than by the likelihood. Real weak-reflection handling wants to admit negative $I$; this study does not.
- `gat` carries a systematic 5–10% *low* bias on $I$. The integrator uses the Anscombe transform for input standardization rather than as a likelihood, so this does not directly indict it, but it is worth knowing before anyone promotes it to a loss.
- `normal_free` is given a per-shoebox fitted scale, which is the *generous* version of the naive baseline. It still comes last.
- $\sigma_I$ for `normal_free` uses the $(I,B)$ block of the observed information, treating the fitted scale as orthogonal. That is essentially exact for a Normal, where mean and variance parameters are orthogonal.

## References

- [JUNGFRAU docs, European XFEL](https://rtd.xfel.eu/docs/jungfrau-detector-documentation/en/latest/general_introduction.html) — gain bit encoding, calibration procedure
- [JUNGFRAU for brighter x-ray sources, Struct. Dyn. 7, 014305 (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7044001/) — 83/200 e⁻ noise, 3500 e⁻ per 12.4 keV photon, 100 ADU pedestal drift, dynamic ranges
- [LCLS psana JUNGFRAU calibration](https://confluence.slac.stanford.edu/display/PSDM/Jungfrau) — `(code - pedestal - offset)/gain`, gain defaults, gain-bit cuts
- [Fast and accurate MX with JUNGFRAU, Nat. Methods 15, 799 (2018)](https://www.nature.com/articles/s41592-018-0143-7)
- [jungfrau-photoncounter](https://github.com/ComputationalRadiationPhysics/jungfrau-photoncounter) — prior art: converting JUNGFRAU data to photon counts is standard practice
