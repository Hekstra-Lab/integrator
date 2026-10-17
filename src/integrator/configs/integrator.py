from dataclasses import dataclass
from typing import Literal


@dataclass
class IntegratorCfg:
    """Architecture and inference hyperparameters for an integrator `LightningModule`.

    Attributes:
        data_dim: Shoebox dimensionality, either `2d` or `3d`.
        d: Shoebox depth in pixels (number of z-slices); ignored for `2d` data but still validated positive.
        h: Shoebox height in pixels; must be positive.
        w: Shoebox width in pixels; must be positive.
        encoder_out: Width of the encoder embedding consumed by the surrogates.
        mc_samples: Monte Carlo samples drawn from each surrogate per forward pass; must be `>= 1`.
        predict_keys: Output columns to emit at predict time, or `default` for the built-in set.
        profile_support: Voxels the profile softmax is normalized over. `valid`
            spreads the profile across every valid voxel (default, unchanged
            behavior); `foreground` restricts it to the DIALS foreground bit,
            the learned-model analog of foreground summation. Ragged integrators
            only; falls back to `valid` per reflection when a shoebox has no
            foreground voxel.
        rb_gate_half_signal: Signal photons at which the profile-uncertainty term of
            the Rao-Blackwellized sigma is half-weighted. `None` applies it in full.
            Only consulted when an `i_rb`/`var_rb` column is requested; the term is
            unreliable where the data cannot constrain the profile, so a few hundred
            photons is a reasonable guard on weak data.
    """

    data_dim: Literal["2d", "3d"]
    d: int
    h: int
    w: int
    encoder_out: int = 64
    mc_samples: int = 4
    predict_keys: Literal["default"] | list[str] = "default"
    profile_support: Literal["valid", "foreground"] = "valid"
    rb_gate_half_signal: float | None = None
    profile_coord_channels: bool = False
    """Give the profile encoder two extra channels holding each pixel's offset
    in the beam-centred radial/tangential frame. A conv encoder is translation
    equivariant and so cannot otherwise know the azimuth that sets the streak
    direction. Requires `xyzcal.px.0/1` in the metadata and a beam centre."""
    beam_center: list[float] | None = None
    """Detector beam centre in pixels; filled from dataset.yaml when unset."""

    # Coset (background-only) reflection handling. Two independent behaviors:
    # whether the rate is forced to background only, and whether the
    # intensity/profile KL is dropped for cosets (background KL is always kept).
    #   "override"       - rate -> background only; full KL (legacy default).
    #   "override_no_kl" - rate -> background only; drop intensity/profile KL.
    #   "supervised"     - rate stays I*prf + bg (no override) so background-only
    #                      counts train the intensity head to report ~0; drop
    #                      intensity/profile KL.
    #   "aux"            - rate -> background only; drop intensity/profile KL;
    #                      add a log-space L2 penalty pulling coset intensity to
    #                      a floor (loss: coset_aux_weight, coset_i_floor).
    coset_mode: Literal[
        "override", "override_no_kl", "supervised", "aux"
    ] = "override"

    # Scaling model: per-HKL structure factor lookup table
    n_hkl: int | None = None
    scaling_init_mu: float = 1.0
    scaling_init_fano: float = 1.0
    scaling_init_k: float = 1.0
    scaling_init_rate: float = 1.0
    scaling_eps: float = 1e-6
    scaling_k_min: float = 0.1
    scaling_rate_min: float = 0.001
    scaling_fano_min: float = 0.0
    scaling_mu_constraint: str = "exp"
    scaling_lr: float | None = None
    # Warm-start: path to a checkpoint to partially load (matching, shape-checked
    # keys) into the integrator before training. Use a previous conjugate scaling
    # model (every key transfers, incl. the scale field) or a converged per-obs
    # integrator (encoders + surrogates transfer; scale stays at init). Applied
    # in construct_integrator and skipped under skip_warmstart (prediction).
    init_from_checkpoint: str | None = None
    # Transplant the background modules (k_bg/r_bg encoders + qbg surrogate) from
    # another run's checkpoint (architectures must match), then freeze them. Takes
    # the background -- the field the merging models keep collapsing -- out of the
    # optimization, using a proven bg from a good integration run. ABSOLUTE path.
    bg_init_from_checkpoint: str | None = None
    bg_freeze: bool = True
    merge_weight: float = 1.0
    merge_kl_weight: float = 1.0
    # Cross-observation scaling-consistency loss weight (ConjugateMergingIntegrator):
    # penalizes disagreement between symmetry-equivalent observations after scaling
    # (DIALS-style internal consistency, data-only), giving the per-obs scale a
    # direct gradient the ELBO under-identifies. 0.0 = off. Start ~0.1-1.0.
    consistency_weight: float = 0.0
    # Compute the consistency-loss WLS target over the Friedel-POOLED group
    # (`nonanom_id`) instead of the anomalous `asu_id`. The per-obs scale is then
    # identified against the +/- pooled mean and cannot lower its residual by
    # splitting the mates, so it stops absorbing the Bijvoet difference; the real
    # anomalous survives in the merge and is regularized by double_wilson_weight.
    # Needs `nonanom_id` in metadata (scripts/add_friedel_metadata.py) and a
    # `group_by_key: nonanom_id` loader so both mates share a batch. Requires
    # consistency_weight > 0 to have any effect.
    consistency_pool_friedel: bool = False
    # Centric anchoring weight (AmortizedMergingIntegrator). Centric reflections
    # have I(+)==I(-) by symmetry, so any Bijvoet difference the scale makes on
    # them is pure scale error. The penalty is the mean squared FRACTIONAL
    # +/- difference of the sign-split data-only WLS intensity on centrics, with
    # only the per-obs scale carrying gradient -> pins the scale on the
    # population where the truth is zero. 0.0 = off; the logged `centric_anchor`
    # is the RMS fake-anomalous-on-centrics, so raise the weight until it drops.
    # Needs centric/friedel_plus/nonanom_id metadata + a nonanom_id loader.
    centric_anchor_weight: float = 0.0
    # Double-Wilson coupling weight (AmortizedMergingIntegrator). Adds the
    # conditional factor of the double-Wilson prior (Dalton, Greisman & Hekstra,
    # Nat. Commun. 2024) on top of the per-HKL marginal Wilson KL: a zero-mean
    # Normal prior on the log-ratio of paired mates' merged means, penalty =
    # mean_pairs (log E[I_+] - log E[I_-])^2. Shrinks noise-driven Bijvoet
    # differences toward zero (the likelihood keeps the real signal); weight ~
    # 1/(2 sigma_anom^2). 0.0 = off. Needs nonanom_id/friedel_plus metadata + a
    # nonanom_id loader so both mates share a batch. Tune up watching CCanom /
    # peak count: too high flattens the real anomalous, too low does nothing.
    double_wilson_weight: float = 0.0
    # Use the centric Wilson prior for centric reflections (AmortizedMerging-
    # Integrator). Acentric I is exponential = Gamma(alpha_W, tau) [mean 1/tau];
    # centric I is chi^2_1 = Gamma(alpha_W/2, alpha_W/2 * tau) -- half the shape
    # and a rate scaled to keep the same mean, so centrics get the correct
    # heavier tail (twice the normalized variance) instead of the acentric
    # exponential. 0.0/False = acentric prior for all (legacy). Needs `centric`
    # in metadata (scripts/add_friedel_metadata.py).
    wilson_centric_prior: bool = False
    # End-of-train-epoch scatter of the model's per-observation intensity
    # (scale * I_h) vs the DIALS intensity (`intensity.sum.value` in metadata),
    # log-log, with the log-space correlation. A per-epoch merge-quality readout;
    # logged to wandb (`intensity_vs_dials`) and saved under the run's
    # `intensity_scatter/`. Off by default.
    log_intensity_scatter: bool = False
    # Per-epoch model background (qbg.mean * n_pixels) vs DIALS background.sum.value
    # log-log scatter (`background_vs_dials` / `background_scatter/`). Off by default.
    log_background_scatter: bool = False
    # HierarchicalScalingIntegrator: freeze the warm-started integration
    # (encoders + qp/qbg/qi) so only the scale + merge head trains. Use with
    # init_from_checkpoint pointing at a trained HierarchicalIntegrator.
    freeze_integration: bool = False
    ema_momentum: float = 0.95

    # Amortized merging head (AmortizedMergingIntegrator):
    #   "mean" - legacy DeepSets mean-pool of features -> qi surrogate.
    #   "sum"  - amortized conjugate update: per-observation positive potentials
    #            summed in natural-parameter space (alpha_h = alpha_W + sum dalpha,
    #            beta_h = tau_h + sum s*prf), scale/geometry-conditioned. Sum (not
    #            mean) is what makes precision scale with multiplicity, mirroring
    #            the conjugate sufficient statistics.
    merge_aggregation: Literal["mean", "sum"] = "mean"
    # sum-mode refinements (ignored for "mean"):
    #   merge_attention     - self-attention over an HKL's observations emits a
    #                         per-obs trust gate (soft outlier rejection).
    #   merge_overdispersion- learned per-HKL variance inflation from the spread
    #                         of per-obs intensities (the error model / random
    #                         effect the conjugate model cannot represent).
    merge_attention: bool = False
    merge_overdispersion: bool = False
    wilson_alpha: float = 1.0
    # SVAEMergingIntegrator: bias of the per-observation responsibility gate,
    # r_p = sigmoid(profile_logit_p + gate_i). Negative so r starts low (-2.0
    # gives r ~ 0.12) and signal emerges as the profile sharpens.
    responsibility_gate_init: float = -2.0
    # SVAEMergingIntegrator: give the signal encoder one output per pixel
    # instead of one per observation. With a scalar the responsibility's shape
    # is the profile's shape and only its level is learned, so the signal
    # encoder cannot say a pixel is background where the profile puts mass.
    responsibility_per_pixel: bool = False
    sample_I_h: bool = True
    # HierarchicalSVAEIntegrator (Option 3 + 3a, GIG-exact): the Gamma random
    # effect J_i | I_h ~ Gamma(nu, nu/I_h) linking per-obs intensities to the
    # per-HKL merged I_h. nu is the dispersion (fixed hyperparameter; larger nu =
    # tighter around I_h = less over-dispersion). The GIG order p = alpha_W -
    # nu*N_h stays constant in nu, which keeps the Bessel order-derivative out of
    # the gradient. n_cavi_iters = unrolled two-level CAVI steps (J_i <-> I_h
    # coupling); a handful suffices once the fixed point is near.
    link_nu: float = 1.0
    n_cavi_iters: int = 8
    # Where the per-observation signal potential lives:
    #   False (default) - PIXEL level: a per-pixel responsibility gate g_{i,p},
    #          signal count = sum_p g_{i,p} c_{i,p} (full SVAE; pixel conjugacy).
    #   True  - OBSERVATION level: a single per-obs responsibility g_i, signal
    #          count = g_i * sum_p c_{i,p}. Mirrors the winning amortized merger /
    #          standalone integrator (potentials at the obs level, encoder still
    #          reads pixels), photon-conserving (g_i in (0,1)). The merge (GIG)
    #          and exposure are unchanged. Switches resp_head's output dim, so it
    #          is checkpoint-incompatible with the pixel-level mode.
    obs_level_potential: bool = False
    # Only when obs_level_potential: how the per-obs signal count is formed.
    #   False (default) - GATED: sigma_i = sigmoid(head) * sum_p c_p, a per-obs
    #          responsibility in (0,1) on the total counts (photon-conserving,
    #          sigma_i <= total counts).
    #   True  - FREE: sigma_i = softplus(head), the head emits the signal count
    #          directly, untied from the photon total (the "just output the
    #          variational parameter" analogue of the standalone integrator; can
    #          exceed the counts). The random-effect prior (nu) and the I_h
    #          coupling (b_i = nu*E[1/I_h] + exposure) are UNCHANGED either way.
    obs_potential_free: bool = False
    # Only when obs_level_potential: also emit the EXPOSURE freely (fully
    # gammaB-like -- the encoder controls BOTH Gamma potentials). The exposure
    # keeps the scale explicit, e_i = scale * softplus(head), so J stays de-scaled
    # for the merge (a fully scale-free exposure would either break the merge or
    # relearn the scale field). False (default) = analytic e_i = scale * sum prof.
    # Set with obs_potential_free for the full "emit both parameters" model; the
    # nu prior and I_h coupling are still conjugate-added.
    obs_exposure_free: bool = False
    # Parameterization of the fully-free (gammaB-like) emission. Requires
    # obs_level_potential + obs_potential_free + obs_exposure_free.
    #   "shape_rate" (default) - heads emit the potentials directly:
    #          sigma_i = softplus(head_a), e_i = scale*softplus(head_b).
    #   "fano" - heads emit the OBSERVED (mean m, Fano phi); the de-scaled
    #          likelihood Gamma(sigma_i, e_i) is recovered as sigma_i = m/phi,
    #          e_i = scale/phi (FanoGamma reparam, scale kept so J stays de-scaled
    #          for the merge). Same graphical model + I_h coupling; only the head
    #          parameterization (and its gradient Jacobian) changes.
    potential_param: str = "shape_rate"
    # HierarchicalSVAEIntegrator merge inference:
    #   True  (default) - FULLY AMORTIZED single pass. A learned per-obs merge
    #          head emits the GIG natural parameter b directly (anchored on the
    #          prior-free MLE), so q(I_h) is still GIG-exact but the J_i<->I_h
    #          coupling is amortized (one Bessel call, no fixed-point loop).
    #   False - the derived two-level CAVI fixed point (n_cavi_iters), the
    #          "purist" conjugate inference; slower (a Bessel call per iter).
    amortize_merge: bool = True
    # Learn the random-effect dispersion nu instead of fixing it at link_nu.
    # nu = softplus(nu_raw) + nu_floor, clamped to nu_max. The J_i|I_h link has
    # CV = 1/sqrt(nu); nu_init ~ 50 gives CV ~14% (recommended). The old
    # link_nu=1.0 is a 100%-CV over-dispersed link that lets J_i float off I_h and
    # absorb scale/background error. Learning nu makes the GIG order p = alpha_W -
    # nu*N_h depend on nu, so the Bessel order-derivative is now needed (supplied
    # by gig._d_p_log_besselk). nu_per_bin models resolution-dependent merge error
    # (AIMLESS/careless error model) on equal-volume 1/d^2 shells. A weak
    # log-Normal restraint pins nu near nu_init so it cannot drift to absorb
    # mis-scaling; nu_warmup_epochs holds nu frozen at nu_init (detached in the
    # forward) while scale/background converge first. link_nu is kept as the fixed
    # value when learn_nu=False (full back-compat).
    learn_nu: bool = False
    nu_init: float = 50.0
    # Dedicated LR for nu_raw (its own Adam group), like scaling_lr for the
    # scale. nu is ONE scalar: at the base lr under global gradient clipping (the
    # conv encoders dominate the norm) its effective step is tiny and it barely
    # moves. A larger nu_lr (~0.05-0.5) lets it actually equilibrate. None = base
    # lr (and nu will crawl). Only used when learn_nu.
    nu_lr: float | None = None
    nu_per_bin: bool = False
    nu_n_bins: int = 10
    nu_floor: float = 1.0e-3
    nu_max: float = 200.0
    nu_restraint_weight: float = 0.0
    nu_restraint_log_sigma: float = 0.5
    nu_warmup_epochs: int = 0
    # Resolution range for nu(d) binning (1/(4 d^2) equal-volume shells); only
    # used when nu_per_bin.
    nu_d_min: float = 1.0
    nu_d_max: float = 60.0
    # Inner EM for ConjugateIntegrator: max responsibility iterations and the
    # relative-change tolerance for early stopping at the fixed point.
    n_em_iters: int = 40
    em_tol: float = 1e-3
    # Calibrated exact-posterior export (ConjugateIntegrator.predict_step). Only
    # computed when a `qi_exact_*` key is in predict_keys. n_nuisance<=1 = Fix A
    # (quadrature at the nuisance means); >1 also propagates q(profile)/q(bg).
    exact_posterior_n_nuisance: int = 16
    exact_posterior_n_grid: int = 1024
    # Amplitude parameterization: "gamma" (default), "normal", or "folded_normal"
    scaling_amplitude: str = "gamma"
    scaling_init_sigma_frac: float = 0.05
    scaling_init_from_wilson: str | None = None

    # Scaling model: Chebyshev scale s(frame) or s(frame, radius)
    # scale_none disables the scale entirely (s=1, no LP) -> rate = prof*I + bg,
    # so the conjugate intensity I is the raw integrated estimate. Takes
    # precedence over scale_mlp / scale_spatial.
    scale_none: bool = False
    scale_degree: int = 5
    scale_frame_min: float = 0.0
    scale_frame_max: float = 1000.0
    scale_mlp: bool = False
    scale_mlp_hidden: int = 64
    scale_mlp_layers: int = 2
    # ResNetScale: a deep residual scale that takes detector x, y DIRECTLY
    # rather than as a radius, so the azimuthal structure separating a Friedel
    # pair survives, and takes lp as an input feature rather than dividing by
    # it. Takes precedence over scale_mlp / scale_spatial; reuses
    # scale_mlp_hidden, scale_beam_center, scale_r_max, scale_frame_min/max and
    # the scale_sh_lmax absorption columns. Needs no scale_extra_features.
    scale_resnet: bool = False
    scale_resnet_blocks: int = 4
    scale_resnet_use_xy: bool = True
    # off for Laue data, which carries no lp column
    scale_resnet_use_lp: bool = True
    # LaueMLPScale only. Off drops wavelength from the scale's features, so a
    # polychromatic Wilson prior's G(lambda) is the only spectral component.
    # The two are one gauge: left both free on hewl_1118 they settled on a
    # prior anti-correlated with the beam and a scale 3x too steep.
    scale_laue_use_wavelength: bool = True
    # `svae_merging` only. On, the Wilson rate is evaluated at EACH
    # observation's own wavelength and resolution and the resulting rates are
    # averaged over the reflection, instead of evaluating the rate once at the
    # group-mean wavelength. The prior is then a statement about measurements,
    # which is what Wilson's is, while the latent stays a crystal quantity.
    wilson_prior_per_obs: bool = False
    # MLPScale output-layer weight init std. 0.0 (default) keeps the legacy
    # zero-init (flat constant scale, zero gradient to hidden layers at step 0);
    # a small value (e.g. 0.05) lets the scale's spatial structure develop from
    # the first step. Bias is always 0 so softplus(0)~0.69 at init regardless.
    scale_head_init_std: float = 0.0
    # Feed the MLP scale the crystal-frame spherical-harmonic absorption features
    # (`absorption_sh` in metadata; reuses scale_sh_lmax for the basis size) as
    # extra inputs, so the flexible MLP can fit the crystal-frame absorption
    # surface it cannot build from lab-frame inputs alone. `even_only` keeps only
    # even-l harmonics -- Friedel-symmetric (Y_l(-r)=Y_l(r)), so they cannot form
    # the odd-l anomalous-gating band (safe by construction; the parity diagnostic
    # showed the true odd content is ~0). Needs the data loader's metadata
    # reference to be the absorption_sh-augmented file (metadata_sh.pt).
    scale_mlp_absorption: bool = False
    scale_mlp_absorption_even_only: bool = True
    # Drop d-spacing and detector radius from the scale MLP's inputs. Both are
    # the same for every symmetry mate of a reflection, so a scale built from
    # them is constant within each merging group, and `s -> c(d) s`,
    # `I_h -> I_h / c(d)` changes neither the likelihood nor the merging
    # consistency. Merging cannot see that direction, so only the Wilson prior
    # fixes it, and the merged intensities end up on the prior's line with the
    # data's real falloff absorbed into the scale. Default False so existing
    # scaling runs are untouched; set True for the hierarchical SVAE.
    scale_mlp_exclude_hkl_constant: bool = False
    # Extra per-observation metadata columns fed to the scale MLP, each
    # standardized by the matching entry of scale_extra_loc / scale_extra_scale.
    # The June svae_merging runs passed the scattering vector as
    # [s1.0, s1.1, s1.2]; frame, lp, radius and d give the scale no direction
    # that distinguishes h from -h within an image, and that is the direction
    # the anomalous difference sits in.
    scale_extra_features: list[str] | None = None

    # Read the per-observation scale straight from a metadata column instead of
    # evaluating the scale model. For diagnosis: pinning the scale to an
    # external reference (careless's own `Scale`, or the identified G(lambda))
    # separates "our scale is wrong" from "our merge is wrong" without
    # retraining, since the merge is closed-form given the scale. None keeps
    # the learned scale, so no existing run changes.
    scale_from_metadata: str | None = None

    # Read the per-observation background from a metadata column instead of
    # inferring it from the (1-r) counts. The scale and the background are
    # competing per-observation nuisances: `b_i` is one free parameter per
    # observation, fitted from the same pixels as the signal under a prior
    # worth one pixel in 561, so a per-observation multiplicative error has
    # somewhere to hide. Measured on synthetic shoeboxes, a conditional fit
    # recovers 0.162 of a known per-observation tilt of 0.5 with `b_i` free
    # and 0.544 with the background held at its true value -- and that does not
    # improve with bigger shoeboxes (0.31 at 100 pixels, 0.32 at 1323).
    # careless never faces this because DIALS fits the background from the
    # shoebox's background region before any scaling. Pinning laue-dials' own
    # per-observation background tests that account on real data. None keeps
    # the conjugate readout, so no existing run changes.
    bg_from_metadata: str | None = None

    # The Laue scale as physics, in an order that keeps its terms apart:
    # G(lambda) + log P(2theta, phi) + K(image) - 2 B(image) s^2 + crystal-frame
    # absorption harmonics, with NOTHING free in detector position or
    # resolution -- on a still lambda = 2 d sin(theta(r)), so a free function of
    # (r, d) absorbs 73% of a realistic spectrum and G stops being identified.
    # Polarisation is supplied, not fitted: its azimuthal signature
    # cos(2(phi - phi0)) lies inside the l=2, m=+-2 harmonics, which ARE
    # cos 2phi and sin 2phi, so fitting both lands the plane 22.5 deg off a
    # known truth and returns the term with CC -0.33 -- the right physics with
    # the sign inverted -- where supplying it gives CC +1.00.
    # Off by default, so no existing run changes.
    scale_laue_physical: bool = False
    # a key of its own, because `scale_spectrum_degree` already exists for
    # `LaueSpectralScale` and defaults to 40 there; reusing it would silently
    # change that class for every config that does not name it
    scale_laue_spectrum_degree: int = 6
    scale_image_degree: int = 8
    scale_polarisation_tau: float = 0.95
    scale_polarisation_phi0_deg: float = 0.0
    scale_deg_per_image: float = 1.0

    # Inflate the exported sigma by the between-observation dispersion the
    # Poisson merge cannot see. sigma^2/I^2 = 1/alpha + eps^2/n_eff + bg, where
    # eps is the per-observation multiplicative scatter estimated from the
    # data's own symmetry-equivalent residuals, as AIMLESS's SDadd and
    # cctbx.xfel's ev11 do. Off by default, so existing exports are unchanged.
    merge_error_model: bool = False
    merge_error_model_shells: int = 10

    # Save the per-observation merge terms (signal counts and exposure) that a
    # conditional Bijvoet estimator needs. A path, or None for no dump.
    merge_dump_terms: str | None = None

    # Refine the responsibility gate toward the exact Poisson E-step,
    # r = s I p / (s I p + b), using the model's own detached estimates. The
    # amortized gate is a function of the shoebox alone, so it cannot know s_i,
    # I_h or b_i and is the correct E-step only by coincidence: measured on
    # hewl_1118 it keeps 45% of the signal below I/sigma 0.5 and 149% near
    # I/sigma 2. 0 keeps the amortized gate alone, so existing runs are
    # unchanged.
    gate_em_iters: int = 0
    scale_extra_loc: list[float] | None = None
    scale_extra_scale: list[float] | None = None
    scale_spatial: bool = False
    scale_degree_radius: int = 5
    scale_beam_center: list[float] | None = None
    scale_r_min: float = 0.0
    scale_r_max: float = 1500.0

    # LaueMLPScale (polychromatic / Laue stills): a wavelength-aware MLP scale.
    # Replaces the rotation-series scales for stills data --
    #   log s = MLP([lambda, x, y, d]) + per-image log-scale,  output exp(.)
    # so the MLP owns the whole correction (incident spectrum G(lambda), detector
    # geometry, resolution falloff). The caller does NOT divide by lp (there is no
    # lp column for Laue stills). Reuses scale_mlp_hidden / scale_mlp_layers /
    # scale_beam_center / scale_r_max / dmin / scale_head_init_std. Takes
    # precedence over scale_physical / scale_mlp / scale_spatial. Pairs with the
    # polychromatic_data loader (group_by_asu_id) + monochromatic_wilson loss
    # (pi_weight 0): the Wilson prior stays resolution-only, the spectrum lives in
    # the scale (F^2 is wavelength-independent). Needs `wavelength` and (if the
    # per-image term is on) `image_num` in the batch metadata.
    scale_laue_mlp: bool = False
    scale_lambda_min: float = 0.95
    scale_lambda_max: float = 1.25
    # Number of distinct images/shots for the optional per-image log-scale (the
    # standard Laue per-shot scale factor). None = no per-image term (pure MLP).
    # Must exceed the max image_num in the data (out-of-range indices clamp).
    scale_n_images: int | None = None

    # LaueSpectralScale (structured Laue scale). G(lambda) as a low-order
    # ChebyshevSpectrum (the form the working polychromatic_wilson integrator
    # uses) in the per-observation scale, plus optional physical Lorentz/
    # polarization and a learned geometry residual. Replaces the black-box
    # LaueMLPScale, which stalls at a near-constant scale (cannot learn G(lambda)
    # against the merge gauge). Takes precedence over scale_laue_mlp. Reuses
    # scale_lambda_min/max, scale_beam_center, scale_r_max, scale_mlp_hidden/
    # layers, dmin. Pairs with the polychromatic_data loader + monochromatic_wilson
    # (resolution-only) prior; F^2 (I_h) is wavelength-independent.
    scale_laue_spectral: bool = False
    scale_spectrum_degree: int = 40
    # Warm-start G(lambda) from a trained polychromatic_wilson model: an ABSOLUTE
    # path to a file with key 'c' (scripts/extract_spectrum.py from its
    # checkpoint's loss.spectrum.c). None = zero-init (learned from scratch).
    scale_spectrum_init_from: str | None = None
    # Freeze the spectrum (de-risking test: warm-start a correct G(lambda) and
    # hold it while the merge settles the gauge; unfreeze afterwards to refine).
    scale_freeze_spectrum: bool = False
    # Physical corrections folded into the scale (Ren & Moffat); fixed, not learned.
    scale_lorentz: bool = False
    scale_polarization: bool = False
    scale_polarization_fraction: float = 0.99
    # Optional learned geometry/absorption residual on [x, y, d] (zero-init no-op).
    scale_residual: bool = False

    # PhysicalScale (DIALS-style): smooth scale(frame) x decay(frame, d) x
    # crystal-frame spherical-harmonic absorption. Takes precedence over
    # scale_mlp / scale_spatial. Needs precomputed `absorption_sh` in the
    # metadata (scripts/extract_crystal_frame_sh.py); scale_sh_lmax MUST match
    # that script's --lmax (n_sh = (lmax+1)^2 - 1). scale_degree sets the
    # rotation scale order; scale_degree_decay the B-factor(frame) order.
    scale_physical: bool = False
    scale_sh_lmax: int = 4
    scale_degree_decay: int = 2
    scale_absorption_init_std: float = 0.0
    # L2 restraint weight on the PhysicalScale absorption (l(l+1)-weighted) and
    # decay coefficients, added to the loss. 0.0 = OFF: the surface is then
    # bounded only by being low-dimensional -- that stops the gross collapse but
    # not the slow run-away the Beer-Lambert absorption showed. Start ~1e-2 and
    # tune on the Bijvoet diagnostic: too low re-absorbs the anomalous signal,
    # too high flattens the real ~4% absorption.
    scale_absorption_restraint: float = 0.0

    # Manual gradient clipping (for manual-optimization integrators)
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = "norm"

    # Refinement model: SFcalculator-based structure factors
    pdb_path: str | None = None
    dmin: float = 2.0
    wavelength: float = 1.0
    anomalous: bool = True
    asu_id_to_hkl_path: str | None = None
    restraint_w_xyz: float = 0.01
    restraint_w_biso: float = 0.001
    atom_lr: float | None = None

    # Variational refinement: isotropic Gaussian position posteriors
    atom_sigma_prior: float | None = None
    kl_atom_weight: float = 1.0

    # Geometry restraints from monomer library (bond lengths, angles)
    geometry_restraints: bool = False
    geometry_w_bond: float = 1.0
    geometry_w_angle: float = 1.0

    # Bulk solvent model: F_total = F_protein + k_sol * exp(-B_sol * s^2) * F_mask
    bulk_solvent: bool = False
    k_sol_init: float = 0.35
    B_sol_init: float = 46.0

    def __post_init__(self):
        if self.data_dim not in ("2d", "3d"):
            raise ValueError(
                f"data_dim must be '2d' or '3d', got {self.data_dim!r}"
            )

        for name in ("d", "h", "w"):
            v = getattr(self, name)
            if v <= 0:
                raise ValueError(f"{name} must be positive, got {v}")

        if self.mc_samples < 1:
            raise ValueError(f"mc_samples must be >= 1, got {self.mc_samples}")

        if self.coset_mode not in (
            "override",
            "override_no_kl",
            "supervised",
            "aux",
        ):
            raise ValueError(
                "coset_mode must be 'override', 'override_no_kl', "
                f"'supervised', or 'aux', got {self.coset_mode!r}"
            )

        if self.merge_aggregation not in ("mean", "sum"):
            raise ValueError(
                "merge_aggregation must be 'mean' or 'sum', got "
                f"{self.merge_aggregation!r}"
            )

        if self.learn_nu or self.nu_per_bin:
            if self.nu_init <= 0:
                raise ValueError(f"nu_init must be > 0, got {self.nu_init}")
            if self.nu_n_bins < 1:
                raise ValueError(f"nu_n_bins must be >= 1, got {self.nu_n_bins}")
            if self.nu_floor <= 0:
                raise ValueError(f"nu_floor must be > 0, got {self.nu_floor}")
            if self.nu_max <= self.nu_init:
                raise ValueError(
                    f"nu_max ({self.nu_max}) must be > nu_init ({self.nu_init})"
                )
            if self.nu_restraint_weight < 0:
                raise ValueError(
                    "nu_restraint_weight must be >= 0, got "
                    f"{self.nu_restraint_weight}"
                )
            if self.nu_restraint_log_sigma <= 0:
                raise ValueError(
                    "nu_restraint_log_sigma must be > 0, got "
                    f"{self.nu_restraint_log_sigma}"
                )
            if self.nu_warmup_epochs < 0:
                raise ValueError(
                    "nu_warmup_epochs must be >= 0, got "
                    f"{self.nu_warmup_epochs}"
                )
            if self.nu_d_max <= self.nu_d_min:
                raise ValueError(
                    f"nu_d_max ({self.nu_d_max}) must be > nu_d_min "
                    f"({self.nu_d_min})"
                )


@dataclass
class IntegratorConfig:
    name: str
    args: IntegratorCfg

    def __post_init__(self):
        from integrator.registry import REGISTRY

        valid = REGISTRY["integrator"].keys()
        if self.name not in valid:
            raise ValueError(
                f"Unknown integrator '{self.name}'. "
                f"Available integrators: {sorted(valid)}"
            )
