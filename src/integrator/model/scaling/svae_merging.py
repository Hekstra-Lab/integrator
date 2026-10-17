"""Per-HKL merging via a per-pixel signal responsibility (structured VAE).

Ported from the `amortized-merging` branch (`3f114d8`, Jun 2026), which is
where the best merging result on 9b7c was measured: R-work 0.1508, R-free
0.1687, against a DIALS reference of 0.1738. `ConjugateMergingIntegrator`
reaches 0.1930 at best on the same data, so the difference below is the one
worth carrying forward.

Sibling of `AmortizedMergingIntegrator`. Instead of a learned per-observation
potential head and a background surrogate, the profile logits are SHARED: a
per-observation gate turns them into a per-pixel signal responsibility

    r_p = sigmoid(profile_logit_p + gate_i)

(Cemgil's Poisson data augmentation, in one pass). The SAME r drives both
conjugate readouts:

    I_h ~ Gamma(alpha_W + sum_i sum_p r*c,   tau_h + sum_i s_i*sum_p prf)
    b_i ~ Gamma(a_bg    + sum_p (1-r)*c,      b_bg + P_i)

so signal + background = total count. Photons are conserved and the split is a
PARTITION rather than two heads competing for the same counts, which is what
makes it robust to the background eating the signal. `ConjugateMergingIntegrator`
pairs a conjugate intensity with an amortized background surrogate, and nothing
ties the two together.

The background prior is the loss's per-resolution-bin Gamma, which this model
was designed around: a scalar `Gamma(1, 1)` measured worse (R-free 0.2147 vs
0.1930 for conjugate merging on the same data). The gate is a small head on the
`k_i` features, initialized negative so r starts low and signal emerges as the
profile sharpens.

That gate is amortized, and it is the model's weakest joint. The split it
parameterises is exact only when `gate_i = log(s_i I_h / b_i)`, and a head on
the shoebox cannot see the scale, the merged intensity or the background it
would need. Measured on hewl_1118 it keeps 45% of the signal below `I/sigma`
0.5 and 149% near 2, and since brightness within a reflection tracks the
spectrum, that error is wavelength-shaped and corrupts any scale fitted from
these counts. `gate_em_iters` solves the `(I_h, b_i)` fixed point instead; see
`_responsibility_merge`.

Two names changed in the port. The June code merged on
`miller_idx_unfriedelized` and grouped batches by `miller_idx_friedelized`;
here those are `asu_id` (Friedel-separate, what the parent merges on) and
`nonanom_id` (Friedel-pooled, what the loader should set `group_by_key` to so
both mates land in one batch and `finalize_merge`'s completeness guard holds).

Subclasses `AmortizedMergingIntegrator` to reuse the scale field, the per-HKL
Wilson KL with its centric branch, the anomalous-preserving auxiliary losses
and the train/finalize plumbing; it overrides only the forward and the merge,
and needs just the profile + k_i encoders and the qp surrogate.
"""

import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Gamma

from integrator import configs
from integrator.model.integrators.hierarchical_integrator import (
    _add_group_outputs,
    _sample_profile,
)
from integrator.model.integrators.integrator_utils import (
    IntegratorBaseOutputs,
    _assemble_outputs,
)
from integrator.model.scaling.amortized_merging import (
    AmortizedMergingIntegrator,
)
from integrator.model.scaling.conjugate_merging import _scatter_sum_compact


logger = logging.getLogger(__name__)

# the per-HKL sums an error model needs, all data-only (no prior terms)
_ERROR_TERMS = (
    "alpha_sig", "beta_sum", "ss", "se", "ee", "v_bg", "n_obs", "d_mean",
)


def _split(
    i_per_obs: Tensor,
    scale: Tensor,
    profile_mean: Tensor,
    mask: Tensor,
    bg_rate: Tensor,
) -> Tensor:
    """The Binomial share of each pixel's count that belongs to the signal.

    `bg_rate` is counts per pixel, the same units as the background the ELBO
    adds to `s I p`.
    """
    sig_rate = scale.reshape(-1, 1) * i_per_obs.reshape(-1, 1) * profile_mean * mask
    return (sig_rate / (sig_rate + bg_rate).clamp(min=1e-12)) * mask


class SVAEMergingIntegrator(AmortizedMergingIntegrator):
    """Per-HKL merging via a shared-logit per-pixel responsibility.

    See the module docstring. Background is a conjugate readout rather than a
    surrogate, so only the profile + k_i encoders and the qp surrogate are
    needed.
    """

    REQUIRED_ENCODERS = {
        "profile": configs.ProfileEncoderArgs,
        "k_i": configs.IntensityEncoderArgs,
    }

    def __init__(self, cfg, loss, encoders, surrogates):
        # `merge_aggregation='mean'` makes the parent require a `qi` surrogate
        # for its own merge; this model never calls that path, so take the
        # branch that builds a head we drop below instead of demanding a
        # surrogate the config has no reason to carry.
        cfg.merge_aggregation = "sum"
        super().__init__(cfg, loss, encoders, surrogates)
        self._error_model_setup(cfg)

        # The parent's potential head is replaced by the responsibility gate;
        # drop it so it does not sit unused in the optimizer.
        if hasattr(self, "alpha_head"):
            del self.alpha_head
        # The background is a conjugate readout of the (1-r) counts, so a qbg
        # surrogate would sit in the optimizer contributing nothing. The
        # factory supplies one from its defaults; drop it.
        if "qbg" in self.surrogates:
            del self.surrogates["qbg"]
        if "qi" in self.surrogates:
            del self.surrogates["qi"]

        d = cfg.encoder_out
        # The June code emitted ONE scalar per observation here, so the
        # responsibility could only slide up or down as a whole and its SHAPE
        # came entirely from the profile logits: no pixel could be called
        # signal unless the profile already said so. Per-pixel lets the signal
        # encoder disagree with the profile about which pixels carry signal,
        # which is what the architecture diagram draws. Default stays scalar so
        # a config that does not ask for it reproduces the June runs exactly.
        self.responsibility_per_pixel = bool(
            getattr(cfg, "responsibility_per_pixel", False)
        )
        self.wilson_prior_per_obs = bool(
            getattr(cfg, "wilson_prior_per_obs", False)
        )
        n_gate = (
            int(math.prod(self.shoebox_shape))
            if self.responsibility_per_pixel
            else 1
        )
        # Named `responsibility_head`, not `gate_head`: the parent already uses
        # that name for its attention gate, which is initialized to +5.0 and
        # means the opposite thing.
        self.responsibility_head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Linear(d, n_gate)
        )
        nn.init.constant_(
            self.responsibility_head[-1].bias,
            float(getattr(cfg, "responsibility_gate_init", -2.0)),
        )

    def _gate(self, x_k_i: Tensor) -> Tensor:
        """Signal-encoder gate: `(B,)` scalar, or `(B, P)` when per-pixel."""
        g = self.responsibility_head(x_k_i)
        return g if self.responsibility_per_pixel else g.squeeze(-1)

    @staticmethod
    def _profile_logits(qp) -> Tensor:
        """Pre-normalization profile logits at the posterior mean.

        `ProfileSurrogate` reports them directly. Surrogates that warp the
        profile after decoding leave `mean_logits` as `None`, because the
        logits no longer line up with the warped grid; there, the log of the
        normalized profile serves instead. The two differ by a per-observation
        constant (the normalizer), and the responsibility gate is exactly a
        per-observation shift, so it absorbs the difference.

        The clamp matters for a sparse normalizer such as `entmax15`, which
        places exact zeros: `log(0)` would otherwise put `-inf` into the
        gradient rather than the zero responsibility intended.
        """
        logits = getattr(qp, "mean_logits", None)
        if logits is not None:
            return logits
        return qp.mean_profile.clamp_min(1e-12).log()

    def _bg_prior(
        self, group_labels: Tensor | None, n: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Per-observation background Gamma prior `(concentration, rate)`.

        Reuses the loss's background prior buffers, which `prepare_priors`
        fills per resolution bin unless the config names `bg_rate` and
        `bg_concentration` itself. Indexes by `group_label` when per-bin, else
        broadcasts the shared scalar to all `n` observations.
        """
        conc = self.loss.bg_concentration.to(device)
        rate = self.loss.bg_rate.to(device)
        if conc.ndim == 1:  # per-resolution-bin priors
            if group_labels is None:
                g = torch.zeros(n, dtype=torch.long, device=device)
            else:
                g = group_labels.to(device).long()
            return conc[g], rate[g]
        return conc.expand(n), rate.expand(n)

    def _per_obs_tau(
        self,
        d_per_obs: Tensor,
        wl_per_obs: Tensor | None,
        scale: Tensor,
        asu_ids: Tensor,
        cnt: Tensor,
    ) -> Tensor:
        """Wilson rate evaluated per observation, then averaged over the group.

        Wilson's statement is about a MEASUREMENT. The expected intensity of
        observation `i` is `G(lambda_i) exp(-2 B s^2)`, so the prior on the
        predicted observed intensity `s_i I_h` is `Gamma(alpha_W, tau_i)`, and
        the implied prior on `I_h` alone is `Gamma(alpha_W, s_i tau_i)`. One
        latent takes one prior, so a reflection's `N` rates are averaged.

        Averaging the RATES is the whole difference from the default path,
        which averages the wavelengths and evaluates `tau` once at the mean:
        `exp` and `1/G` do not commute with a mean, and `G` varies ~15x across
        this band, so the two differ by much more than rounding.

        The construction is also self-correcting. When the scale carries the
        spectrum, as a per-observation factor should, `s_i` tracks
        `G(lambda_i)` and the wavelength cancels out of `s_i tau_i`, leaving
        the lambda-free prior that a crystal-level latent wants. When it does
        not, the prior supplies what is missing.

        `scale` is detached: the rate is a statement about where the data was
        measured, not a quantity the scale should be pushed around by. Left
        attached, the KL could be reduced by shrinking the scale, which is a
        gauge direction with no physical content. The scale still enters
        `beta_h` through the likelihood with its gradient intact.
        """
        s_sq = 1.0 / (4.0 * d_per_obs.clamp(min=1e-6).pow(2))
        meta: dict[str, Tensor] = {"d": d_per_obs}
        if wl_per_obs is not None:
            meta["wavelength"] = wl_per_obs
        tau_obs = self.loss._get_tau(meta, s_sq, d_per_obs.device)
        tau_sum, _, _ = _scatter_sum_compact(
            scale.detach().reshape(-1) * tau_obs.reshape(-1), asu_ids
        )
        return (tau_sum / cnt.clamp(min=1.0)).clamp(min=1e-12)

    def _error_model_setup(self, cfg) -> None:
        """Read the error-model keys. Called from `__init__`."""
        self.merge_error_model = bool(getattr(cfg, "merge_error_model", False))
        self.merge_error_model_shells = int(
            getattr(cfg, "merge_error_model_shells", 10)
        )
        self.merge_dump_terms = getattr(cfg, "merge_dump_terms", None)
        self.gate_em_iters = int(getattr(cfg, "gate_em_iters", 0))

    def _responsibility_merge(
        self,
        logits: Tensor,
        gate: Tensor,
        counts: Tensor,
        scale: Tensor,
        profile_mean: Tensor,
        mask: Tensor,
        asu_ids: Tensor,
        d_per_obs: Tensor,
        group_labels: Tensor | None,
        wl_per_obs: Tensor | None = None,
        extras: dict | None = None,
    ) -> tuple[Gamma, Tensor, Tensor, Tensor, Tensor, Tensor, Gamma, Tensor]:
        """Conjugate merge via a per-pixel signal responsibility.

        `r_p = sigmoid(profile_logit_p + gate_i)` splits each pixel's count into
        signal (`r_p`) and background (`1 - r_p`). The same r feeds both
        conjugate readouts (signal to `alpha_h`, background to a per-observation
        Gamma); signal + background = total count, so photons are conserved.

        Returns `(qi_h, alpha_h, beta_h, inverse, unique, tau_h, qbg, r)`.
        """
        device = counts.device
        b = counts.shape[0]
        m = mask.reshape(b, -1).float()
        c = counts.clamp(min=0).reshape(b, -1).float() * m
        # (B,) scalar gate broadcasts over pixels; (B, P) per-pixel gate
        # is already aligned with the logits.
        g = gate if gate.ndim == 2 else gate.unsqueeze(-1)
        r = torch.sigmoid(logits + g) * m  # (B, P), masked

        total_counts = c.sum(dim=-1)
        sig_counts = (r * c).sum(dim=-1)  # sum_p r*c       (signal)
        bg_counts = total_counts - sig_counts  # sum_p (1-r)*c   (conservation)
        n_pix = m.sum(dim=-1).clamp(min=1.0)  # background exposure P_i

        # Per-HKL intensity posterior: signal counts give the shape, the
        # scale-weighted profile mass the rate.
        d_sum, inverse, unique = _scatter_sum_compact(d_per_obs, asu_ids)
        cnt, _, _ = _scatter_sum_compact(torch.ones_like(d_per_obs), asu_ids)
        if self.wilson_prior_per_obs:
            tau_h = self._per_obs_tau(
                d_per_obs, wl_per_obs, scale, asu_ids, cnt
            )
        else:
            extra = None
            if wl_per_obs is not None:
                wl_sum, _, _ = _scatter_sum_compact(wl_per_obs, asu_ids)
                extra = {"wavelength": wl_sum / cnt.clamp(min=1.0)}
            tau_h = self._wilson_tau(
                (d_sum / cnt.clamp(min=1.0)).clamp(min=1e-6), extra
            )
        delta_beta = scale * (profile_mean * m).sum(dim=-1)
        alpha_sig, _, _ = _scatter_sum_compact(sig_counts, asu_ids)
        beta_sum, _, _ = _scatter_sum_compact(delta_beta, asu_ids)
        alpha_h = self.alpha_W + alpha_sig
        beta_h = tau_h + beta_sum
        qi_h = Gamma(alpha_h.clamp(min=1e-6), beta_h.clamp(min=1e-12))

        # Per-observation background posterior from the (1-r) counts.
        bg_a0, bg_b0 = self._bg_prior(group_labels, b, device)
        qbg = Gamma(
            (bg_a0 + bg_counts).clamp(min=1e-6),
            (bg_b0 + n_pix).clamp(min=1e-12),
        )

        if self.gate_em_iters:
            # The exact E-step. A Poisson count whose rate is a sum splits
            # Binomially, with r = s I p / (s I p + b) -- and
            # `sigmoid(log p + g)` equals that precisely when
            # g = log(s I / b), which the amortized gate has no way to know.
            #
            # Iterating the split is an EM map, and in log I_h its derivative
            # is K = V_h / alpha_h with V_h = sum_p c r (1-r): strictly inside
            # (0, 1), since r(1-r) < r makes V_h < S_h <= alpha_h. At 95%
            # background it sits just under 1 and a plain iteration moves I_h
            # by about 1.5% a step, so a few of them are a no-op; the
            # implicit-function correction (1 - K)^-1 -- the same 1/(1-K)
            # `ConjugateMergingIntegrator._conjugate_em_merged` applies -- gets
            # there in two or three. The step has to be taken in log I_h: the
            # linear-space derivative V_h / (I_h beta_h) is not bounded by 1
            # below the root, and accelerating by it walks away from the
            # solution it is supposed to find.
            #
            # The split depends on the background as much as on I_h, so both
            # sides are re-estimated: holding `b` at the amortized gate's value
            # hands the solve the gate's own bias back.  `qbg`'s rate is
            # `bg_b0 + n_pix`, so its mean is already counts per pixel -- the
            # quantity the ELBO adds to `zI_scaled * zp`.
            bg_rate = qbg.mean.detach().reshape(-1, 1)
            i_hat = (alpha_h / beta_h.clamp(min=1e-12)).detach()
            beta_c = beta_h.detach().clamp(min=1e-12)
            bg_exposure = (bg_b0 + n_pix).clamp(min=1e-12)
            with torch.no_grad():
                for _ in range(self.gate_em_iters):
                    r = _split(i_hat[inverse], scale, profile_mean, m, bg_rate)
                    sig = (r * c).sum(dim=-1)
                    bg_rate = (
                        (bg_a0 + (total_counts - sig)).clamp(min=0.0) / bg_exposure
                    ).reshape(-1, 1)
                    s_h, _, _ = _scatter_sum_compact(sig, asu_ids)
                    v_h, _, _ = _scatter_sum_compact(
                        (c * r * (1.0 - r)).sum(dim=-1), asu_ids
                    )
                    a_h = self.alpha_W + s_h
                    step = (a_h / beta_c).clamp(min=1e-30).log() - i_hat.clamp(
                        min=1e-30
                    ).log()
                    k = (v_h / a_h.clamp(min=1e-12)).clamp(max=1.0 - 1e-3)
                    i_hat = (i_hat.clamp(min=1e-30).log() + step / (1.0 - k)).exp()
            # One differentiable pass at the converged intensity: the scale and
            # the profile keep their gradients, I_h and b do not, so this
            # refines the split rather than opening a path the KL can exploit.
            r = _split(i_hat[inverse], scale, profile_mean, m, bg_rate)
            sig_counts = (r * c).sum(dim=-1)
            bg_counts = total_counts - sig_counts
            alpha_sig, _, _ = _scatter_sum_compact(sig_counts, asu_ids)
            alpha_h = self.alpha_W + alpha_sig
            qi_h = Gamma(alpha_h.clamp(min=1e-6), beta_h.clamp(min=1e-12))
            qbg = Gamma(
                (bg_a0 + bg_counts).clamp(min=1e-6),
                (bg_b0 + n_pix).clamp(min=1e-12),
            )
        if extras is not None:
            # The terms an error model needs, all per HKL and all from this
            # batch, which holds complete groups. Prior contributions are kept
            # out: the dispersion is a property of the data, not of tau or
            # alpha_W. `sig_counts` is S_i, `delta_beta` is the exposure E_i,
            # so mu_i = E_i * I_h and the Pearson sum follows from the three
            # cross-sums without needing a second pass.
            r_sq = (r * r).sum(dim=-1)
            bg_var = qbg.concentration / qbg.rate.pow(2)
            v_bg = r_sq * bg_var
            for key, value in (
                ("alpha_sig", alpha_sig),
                ("beta_sum", beta_sum),
                ("ss", _scatter_sum_compact(sig_counts.pow(2), asu_ids)[0]),
                ("se", _scatter_sum_compact(sig_counts * delta_beta, asu_ids)[0]),
                ("ee", _scatter_sum_compact(delta_beta.pow(2), asu_ids)[0]),
                ("v_bg", _scatter_sum_compact(v_bg, asu_ids)[0]),
                ("n_obs", cnt),
                ("d_mean", d_sum / cnt.clamp(min=1.0)),
            ):
                extras[key] = value
            # `S_i` is not a clean signal measurement: it attributes a share of
            # EVERY pixel's count, so background enters with weight r_p and
            # leaves a pedestal `b_i sum_p r_p` that is nearly common inside a
            # reflection. Measured at the solve's own fixed point that pedestal
            # is 93% of S_i at 95% background and 98.5% at 99%, and an additive
            # pedestal dilutes exactly the multiplicative pattern a scale fit
            # reads -- a known per-observation tilt of 0.5 comes back as 0.011.
            # So also dump the background-subtracted estimate and its matching
            # exposure, for which `E[S] = I_h e` holds exactly:
            #
            #   S~_i = sum_p r_p (c_p - b_i),   e~_i = s_i sum_p r_p p_p.
            #
            # That keeps the conditional likelihood's exact profiling-out of
            # I_h, which taking the pedestal as an additive offset would
            # destroy, since the rate would stop being multiplicative in I_h.
            r_sum = r.sum(dim=-1)
            bg_mean = qbg.mean.detach()
            extras["per_obs"] = {
                "sig_counts": sig_counts.detach(),
                "exposure": delta_beta.detach(),
                # the scale this pass used, so a refit can recover the base
                # exposure e_i = E_i / s_i and solve for a new scale offline
                "scale": scale.detach().reshape(-1),
                "sig_counts_bgsub": (sig_counts - bg_mean * r_sum).detach(),
                "exposure_eff": (
                    scale.reshape(-1) * (r * profile_mean * m).sum(dim=-1)
                ).detach(),
                "bg_pedestal": (bg_mean * r_sum).detach(),
            }
        return qi_h, alpha_h, beta_h, inverse, unique, tau_h, qbg, r

    def _forward_impl(
        self,
        counts: Tensor,
        shoebox: Tensor,
        mask: Tensor,
        metadata: dict,
    ) -> dict[str, Any]:
        counts = torch.clamp(counts, min=0)
        b = shoebox.shape[0]
        device = shoebox.device
        shoebox_reshaped = (shoebox * mask).reshape(b, 1, *self.shoebox_shape)

        x_profile = self.encoders["profile"](shoebox_reshaped)
        x_k_i = self.encoders["k_i"](shoebox_reshaped)
        qp = self.surrogates["qp"](x_profile, mc_samples=self.mc_samples)
        profile_mean = qp.mean_profile

        scale = self._get_scale(metadata, device)
        asu_ids = metadata["asu_id"].long().to(device)
        d_obs = metadata["d"].to(device).float()
        group_labels = (
            metadata["group_label"].long().to(device)
            if "group_label" in metadata
            else None
        )

        # Shared profile logits plus a per-observation gate give the per-pixel
        # responsibility; I_h and the background are both conjugate readouts.
        gate = self._gate(x_k_i)
        (
            qi_h,
            alpha_h,
            beta_h,
            inverse,
            unique_hkls,
            tau_h,
            qbg,
            responsibility,
        ) = self._responsibility_merge(
            self._profile_logits(qp),
            gate,
            counts,
            scale,
            profile_mean,
            mask,
            asu_ids,
            d_obs,
            group_labels,
            metadata["wavelength"].to(device).float()
            if "wavelength" in metadata
            else None,
        )

        # Sample I_h once per HKL, broadcast to its observations.
        zI_h = qi_h.rsample([self.mc_samples]).clamp(min=1e-10)
        zI = zI_h[:, inverse]
        zI_scaled = (scale.unsqueeze(0) * zI).unsqueeze(-1).permute(1, 0, 2)
        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = _sample_profile(qp, self.mc_samples)
        rate = zI_scaled * zp + zbg

        if "is_coset" in metadata:
            coset = metadata["is_coset"].bool().view(-1, 1, 1)
            rate = torch.where(coset, zbg, rate)

        qi = Gamma(
            alpha_h[inverse].clamp(min=1e-6),
            beta_h[inverse].clamp(min=1e-12),
        )

        out = IntegratorBaseOutputs(
            rates=rate,
            counts=counts,
            mask=mask,
            qbg=qbg,
            qp=qp,
            qi=qi,
            zp=zp,
            zbg=zbg,
            metadata=metadata,
        )
        out = _assemble_outputs(out)
        out["asu_id"] = asu_ids
        if "group_label" in metadata:
            _add_group_outputs(out, metadata, self.loss)

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
            "qi_h": qi_h,
            "alpha_h": alpha_h,
            "beta_h": beta_h,
            "tau_h": tau_h,
            "inverse": inverse,
            "unique_hkls": unique_hkls,
            "scale": scale,
            "responsibility": responsibility,
        }

    def _apply_error_model(self, terms: dict, seen: Tensor) -> None:
        """Inflate the exported posterior by the dispersion the merge cannot see.

        The merge is Poisson-conjugate, so `alpha/beta^2` is the variance of a
        pure counting process.  What the data actually show is
        `Var(S_i) = mu_i + eps^2 mu_i^2`: a per-observation multiplicative
        scatter, which is the scale's own error plus anything else that
        multiplies an observation.  Pooling the symmetry equivalents of a
        resolution shell gives it by the method of moments, which is what
        AIMLESS's `SDadd` and `cctbx.xfel`'s `ev11` estimate and what careless
        gets for free from a stochastic scale.

        With `mu_i = E_i * I_h`, the Pearson sum follows from the three
        cross-sums already accumulated, so no second pass over the data is
        needed:

            sum_i (S_i - mu_i)^2 = ss - 2 I_h se + I_h^2 ee.

        The inflation is applied as `alpha/(1+phi)`, `beta/(1+phi)`, which
        holds the posterior mean exactly and multiplies its variance by
        `1 + phi` -- the same device as the parent's overdispersion branch,
        which this model's merge never reaches.
        """
        idx = seen.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return
        alpha_sig = terms["alpha_sig"][idx]
        beta_sum = terms["beta_sum"][idx].clamp(min=1e-12)
        i_hat = (alpha_sig / beta_sum).clamp(min=0.0)
        pearson = (
            terms["ss"][idx]
            - 2.0 * i_hat * terms["se"][idx]
            + i_hat.pow(2) * terms["ee"][idx]
        )
        mu_sum = i_hat * beta_sum
        mu_sq_sum = i_hat.pow(2) * terms["ee"][idx].clamp(min=0.0)
        # Poisson and background-subtraction variance removed; what is left is
        # the multiplicative part, and only reflections with enough signal to
        # see it carry information about it
        excess = pearson - mu_sum - terms["v_bg"][idx]
        n_obs = terms["n_obs"][idx].clamp(min=2.0)
        # (n-1)/n for the plug-in I_h; negligible at ~100 observations, exact here
        excess = excess * (n_obs / (n_obs - 1.0))

        d_mean = terms["d_mean"][idx].clamp(min=1e-6)
        shells = max(int(self.merge_error_model_shells), 1)
        order = torch.argsort(d_mean, descending=True)
        shell_of = torch.zeros_like(idx)
        bounds = torch.linspace(
            0, idx.numel(), shells + 1, device=idx.device
        ).long()
        for s in range(shells):
            shell_of[order[bounds[s] : bounds[s + 1]]] = s

        eps_sq = torch.zeros(shells, device=idx.device)
        for s in range(shells):
            sel = shell_of == s
            denom = mu_sq_sum[sel].sum()
            if float(denom) <= 0:
                continue
            eps_sq[s] = (excess[sel].sum() / denom).clamp(min=0.0)

        n_eff = (beta_sum.pow(2) / terms["ee"][idx].clamp(min=1e-30)).clamp(min=1.0)
        rel_var = eps_sq[shell_of] / n_eff + terms["v_bg"][idx] / mu_sum.clamp(
            min=1e-12
        ).pow(2)
        phi = (self.alpha_buffer[idx] * rel_var).clamp(min=0.0)
        self.alpha_buffer[idx] = self.alpha_buffer[idx] / (1.0 + phi)
        self.beta_buffer[idx] = self.beta_buffer[idx] / (1.0 + phi)
        self._error_model_eps_sq = eps_sq.detach().cpu()
        logger.info(
            "error model: eps per shell (low to high resolution) %s; "
            "median sigma inflation %.3fx",
            " ".join(f"{float(e)**0.5:.3f}" for e in eps_sq),
            float((1.0 + phi).sqrt().median()),
        )

    def _write_merge_dump(self, dump: dict) -> None:
        """Save the per-observation merge terms a conditional estimator needs.

        `scripts/stratified_bijvoet.py` reads this: the signal counts `S_i` and
        the exposure `E_i` are what a Poisson model of one observation needs,
        and the keys beside them say which reflection, which Friedel mate and
        which wavelength it belongs to.
        """
        out = {}
        for key, chunks in dump.items():
            if chunks:
                out[key] = torch.cat(chunks)
        path = Path(self.merge_dump_terms)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, path)
        logger.info(
            "merge terms for %d observations -> %s",
            len(out.get("sig_counts", [])),
            path,
        )

    @torch.no_grad()
    def finalize_merge(self, dataloader) -> None:
        """Recompute the per-HKL posterior over the full dataset.

        Like the parent, needs a loader that yields COMPLETE HKL groups per
        batch (`predict_dataloader(grouped=True)`); a guard raises otherwise.
        The exported intensities are therefore a whole-dataset posterior, not
        per-batch values.
        """
        self.eval()
        device = self.alpha_buffer.device
        seen = torch.zeros(self.n_hkl, dtype=torch.bool, device=device)
        self.alpha_buffer.fill_(self.alpha_W)
        self.beta_buffer.fill_(1.0)
        terms = {
            key: torch.zeros(self.n_hkl, device=device) for key in _ERROR_TERMS
        }
        dump: dict[str, list] = {} if not self.merge_dump_terms else {}

        for batch in dataloader:
            counts, shoebox, mask, metadata = batch
            counts = counts.to(device)
            shoebox = shoebox.to(device)
            mask = mask.to(device)
            b = shoebox.shape[0]
            sr = (shoebox * mask).reshape(b, 1, *self.shoebox_shape)
            scale = self._get_scale(metadata, device)
            asu_ids = metadata["asu_id"].long().to(device)
            d_obs = metadata["d"].to(device).float()
            qp = self.surrogates["qp"](
                self.encoders["profile"](sr), mc_samples=1
            )
            group_labels = (
                metadata["group_label"].long().to(device)
                if "group_label" in metadata
                else None
            )
            gate = self._gate(self.encoders["k_i"](sr))
            want_extras = self.merge_error_model or self.merge_dump_terms
            extras: dict | None = {} if want_extras else None
            _, alpha_h, beta_h, _, unique, _, _, _ = (
                self._responsibility_merge(
                    self._profile_logits(qp),
                    gate,
                    counts,
                    scale,
                    qp.mean_profile,
                    mask,
                    asu_ids,
                    d_obs,
                    group_labels,
                    metadata["wavelength"].to(device).float()
                    if "wavelength" in metadata
                    else None,
                    extras=extras,
                )
            )
            if extras is not None:
                # each HKL is complete in one batch, so a flat per-HKL buffer
                # is enough -- no accumulation across batches is needed
                for key in _ERROR_TERMS:
                    terms[key][unique] = extras[key]
                if self.merge_dump_terms:
                    for name, value in extras["per_obs"].items():
                        dump.setdefault(name, []).append(value.cpu())
                    # detector position and the diffracted beam direction:
                    # polarisation needs the azimuth, and the crystal-frame
                    # absorption basis needs s1 rotated by -phi(image)
                    # refl_ids is the reflection table's own row id, so a
                    # consumer can align the dump to the metadata exactly
                    # instead of matching on rounded coordinates
                    for key in ("refl_ids", "asu_id", "nonanom_id", "wavelength",
                                "d", "centric", "H", "K", "L", "image_num",
                                "xyzcal.px.0", "xyzcal.px.1",
                                "s1.0", "s1.1", "s1.2"):
                        if key in metadata:
                            v = metadata[key]
                            v = v if torch.is_tensor(v) else torch.as_tensor(v)
                            dump.setdefault(key, []).append(v.reshape(-1).cpu())
            if bool(seen[unique].any()):
                raise RuntimeError(
                    "finalize_merge requires a grouped (group_by_asu_id) loader "
                    "so each HKL is complete in one batch; found an HKL spanning "
                    "batches. Use predict_dataloader(grouped=True)."
                )
            self.alpha_buffer[unique] = alpha_h
            self.beta_buffer[unique] = beta_h
            seen[unique] = True
        self.buffer_seen.copy_(seen)
        if self.merge_dump_terms:
            self._write_merge_dump(dump)
        if self.merge_error_model:
            self._apply_error_model(terms, seen)
