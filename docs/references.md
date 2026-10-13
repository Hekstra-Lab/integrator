# References

Work this project builds on, and where each one enters the code.

Entries marked **[verify]** are ones whose exact venue, year or volume has not been checked against the published record — the content is right, the citation line needs confirming before it goes into a manuscript.

## Simplex maps for the profile

The profile is a distribution over the pixels of a shoebox: non-negative, summing to one.
How that constraint is imposed turned out to matter more than the basis being constrained.

**Martins, A. F. T. & Astudillo, R. F. (2016). From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification.** ICML. [arXiv:1602.02068](https://arxiv.org/abs/1602.02068)

Euclidean projection onto the simplex, giving exactly zero on the components it excludes.
The $\alpha = 2$ end of the family below.

**Peters, B., Niculae, V. & Martins, A. F. T. (2019). Sparse Sequence-to-Sequence Models.** ACL. [arXiv:1905.05702](https://arxiv.org/abs/1905.05702)

$\alpha$-entmax, interpolating softmax ($\alpha = 1$) and sparsemax ($\alpha = 2$).
At $\alpha = 1.5$ the solution has a closed form found by sorting, which is what we implement in `entmax15` / `masked_entmax15` in `model/distributions/profile_surrogates.py`, with the paper's closed-form backward rather than autograd through the sort.

Why it is in this codebase: softmax is strictly positive, so a profile built from it cannot place exactly zero mass on a pixel, and its gradient $\pi(1-\pi)$ stops pushing well before a small tail is gone.
Spread over a 625-pixel box, $2 \times 10^{-4}$ a pixel that nothing can remove is 12.5% of the profile.
Measured on hewl_1118, the softmax learned-basis family held 13.5% of its mass beyond 5 px from the spot centre; the same arm with entmax holds 0.26%, and its anomalous peak heights rose from 4.03 to 8.12.

## Profile models

**Refregier, A. (2003). Shapelets -- I. A method for image analysis.** MNRAS 338, 35. [arXiv:astro-ph/0105178](https://arxiv.org/abs/astro-ph/0105178)

Gauss--Hermite basis functions for decomposing galaxy images, built for weak-lensing shear measurement.
The basis is complete, orthonormal, and ordered so low orders carry the bulk — and because every element is a Gaussian times a polynomial, a finite expansion cannot have heavy tails.
That is the property a Bragg profile needs, and it is why `ShapeletProfileSurrogate` normalizes a squared amplitude rather than exponentiating a free logit.

Companion papers, for the shear application and the polar form: **Refregier & Bacon (2003), Shapelets -- II** and **Massey & Refregier (2005), Polar shapelets**. **[verify]**

**Sullivan, B., Archibald, R., Langan, P. S., Dobbek, H., Bommer, M., McFeeters, R. L., Coates, L., Wang, X., Gallmeier, F., Carpenter, J. M., Lynch, V. & Langan, P. (2018). Improving the accuracy and resolution of neutron crystallographic data by three-dimensional profile fitting of Bragg peaks in reciprocal space.** Acta Cryst. D74, 1085--1095. [doi:10.1107/S2059798318013347](https://doi.org/10.1107/S2059798318013347)

The closest published analogue to what we are doing, in a neighbouring field and the same data regime: weak peaks, high background, long wavelengths.
Four things in it bear directly on our design.

* The profile is **parametric and physics-indexed**: an Ikeda--Carpenter function along time-of-flight times a bivariate Gaussian in the two angular coordinates, $\psi = A(\psi_{IC} \times \psi_{BVG}) + B$.
  Three parameters describe the angular part — $\sigma_{2\theta}$, $\sigma_{\varphi_{az}}$ and a covariance $\rho$ setting the orientation.
* Their Figure 5 shows those parameters varying **smoothly over the detector**: $\sigma_{2\theta}$ falls monotonically with scattering angle, $\sigma_{\varphi_{az}}$ is nearly constant, $\rho$ oscillates with azimuth.
  Profile shape is a low-dimensional, smooth function of detector position — the premise a pre-trained, fine-tunable model needs.
* Weak peaks **borrow from a library of strong ones**: peaks with fewer than 250 events take their angular profile from the nearest peak with more than 500, in $(q_{\varphi_{az}}, q_{2\theta})$.
  Fit the shape where there is signal, apply it where there is not — which is what amortization does in our encoder.
* Their variance includes a **fit term**, $\mathrm{var}(I) = \sigma^2_{\mathrm{obs}} + \sigma^2_{\mathrm{bg}} + \sigma^2_{\mathrm{fit}}$ with $\sigma^2_{\mathrm{fit}} = \sum[N_{\mathrm{obs}}(N_{\mathrm{obs}} - N_{\mathrm{model}})^2] / \sum N_{\mathrm{obs}}$.
  They budget for the profile being fitted rather than known; our conjugate posterior does not, which is the overconfidence measured in `scripts/check_posterior_variance.py`.

Results: R-free down 2.3%, 2.3% and 2.7% on three proteins, CC$_{1/2}$ up most at high resolution.

Earlier profile-fitting work it cites, for the history: Alexander & Smith (1962), Diamond (1969), Rossmann (1985), Leslie (2006), Kabsch (2006). **[verify]**

## Statistics and merging

**Wilson, A. J. C. (1949).** The intensity statistics the prior is built on: an acentric $|F|^2$ is exponential, a centric one is $\Sigma$ times a $\chi^2$ with one degree of freedom — same mean, twice the variance. **[verify]**

Implemented in `model/loss/wilson_loss.py` (`intensity_prior_params`) and, since `1d9174b`, used by the conjugate SVAE integrator rather than its own acentric-only copy.

**French, S. & Wilson, K. (1978). On the treatment of negative intensity observations.** Acta Cryst. A34, 517--525. **[verify]**

The $I \to F$ transform with a positivity prior; `--columns intensity` in the refinement chain turns it on, `--columns amplitude` off.

**Karplus, P. A. & Diederichs, K. (2012). Linking crystallographic model and data quality.** Science 336, 1030--1033. **[verify]**

CC$_{1/2}$ as a resolution criterion.
Worth reading alongside our own finding that CC$_{1/2}$ computed on a merging model's output is close to meaningless: splitting already-merged batch values gave 0.996 for a model whose intensities were 800x mis-scaled and drifted 7.27x across resolution.
For such a model the half-dataset split has to happen in the **data** (see `image_parity`), as careless does with its xval files.

**Dalton, K. M., Greisman, J. B. & Hekstra, D. R. (2022). A unifying Bayesian framework for merging X-ray diffraction data.** Nature Communications. **[verify]**

`careless`; the scaling and merging step our integrate-then-scale arm uses, and the variational framing the merging models depart from.

**Winter, G. et al. (2018). DIALS: implementation and evaluation of a new integration package.** Acta Cryst. D74, 85--97. **[verify]**

The reference integration pipeline; `src/integrator/dials_port` is a torch reimplementation of its scaling and merging, verified to ~1e-15 against cctbx.

**laue-dials** — the Laue extension used for the polychromatic baselines. **[verify: citation]**

## Simulation

**nanoBragg / diffBragg** (Mendez et al.) — the physics simulator ported to torch in `src/integrator/physics/nanobragg.py`, matching the C implementation to float32, and the source of the synthetic ecDHFR Laue data. **[verify: citation]**

**Ikeda, S. & Carpenter, J. M. (1985).** The moderator pulse-shape function Sullivan et al. fit along time-of-flight. **[verify]**

## Where these show up in the code

| reference | code |
|---|---|
| Peters et al. 2019 | `model/distributions/profile_surrogates.py` — `entmax15`, `masked_entmax15` |
| Refregier 2003 | `ShapeletProfileSurrogate` in the same file |
| Sullivan et al. 2018 | `PhysicalProfileSurrogate`; the open $\sigma_{\mathrm{fit}}$ question |
| Wilson 1949 | `model/loss/wilson_loss.py`, `scripts/fit_wilson_scale.py` |
| Karplus & Diederichs 2012 | `scripts/merging_stats_from_preds.py`, `image_parity` |
| DIALS | `src/integrator/dials_port/` |
