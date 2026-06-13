from dataclasses import dataclass
from typing import Literal


@dataclass
class IntegratorCfg:
    data_dim: Literal["2d", "3d"]
    d: int
    h: int
    w: int
    lr: float = 0.001
    encoder_out: int = 64
    weight_decay: float = 0.0
    decoder_weight_decay: float | None = None
    qp_smoothness_weight: float | None = None
    qp_orthogonality_weight: float | None = None
    lr_schedule: Literal["cosine_warmup", "step_linear_warmup"] | None = None
    warmup_epochs: int = 5
    warmup_steps: int = 0
    lr_min: float = 1.0e-5
    mc_samples: int = 4
    predict_keys: Literal["default"] | list[str] = "default"

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
    sample_I_h: bool = True
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

        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")

        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be non-negative, got {self.weight_decay}"
            )

        if (
            self.decoder_weight_decay is not None
            and self.decoder_weight_decay < 0
        ):
            raise ValueError(
                "decoder_weight_decay must be non-negative, got "
                f"{self.decoder_weight_decay}"
            )

        for name in (
            "qp_smoothness_weight",
            "qp_orthogonality_weight",
        ):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise ValueError(f"{name} must be non-negative, got {v}")

        if self.warmup_epochs < 0:
            raise ValueError(
                f"warmup_epochs must be non-negative, got {self.warmup_epochs}"
            )
        if self.lr_min < 0:
            raise ValueError(f"lr_min must be non-negative, got {self.lr_min}")
        if self.lr_min > self.lr:
            raise ValueError(
                f"lr_min ({self.lr_min}) must be <= lr ({self.lr})"
            )

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
