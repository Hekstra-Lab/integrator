"""SVAE with q(I) and q(b) built per profile sample: q(I, b, eta) = q(eta) q(I | eta) q(b | eta)."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Distribution, Gamma, kl_divergence, register_kl

from integrator.model.integrators.svae_integrator import SVAEIntegrator


class PerSampleGamma(Distribution):
    """One Gamma per profile sample, (S, B) parameters, batch shape (B,).

    `rsample([S])` returns one draw from each component, aligned with the
    profile samples it was built from; `mean` and `variance` are those of the
    equal-weight mixture over the S components (the variance by the law of
    total variance, so it includes the spread of the conditional means across
    profile samples). KL to a Gamma prior is the average of the components'
    closed-form KLs, which is the exact E_{q(eta)} KL(q(I | eta) || p(I)) of the
    structured family, estimated on the S samples.
    """

    arg_constraints: dict = {}
    has_rsample = True

    def __init__(self, concentration: Tensor, rate: Tensor):
        self.concentration, self.rate = torch.broadcast_tensors(concentration, rate)
        self._components = Gamma(self.concentration, self.rate)
        super().__init__(batch_shape=self.concentration.shape[1:], validate_args=False)

    @property
    def n_components(self) -> int:
        return self.concentration.shape[0]

    @property
    def mean(self) -> Tensor:
        return self._components.mean.mean(0)

    @property
    def variance(self) -> Tensor:
        m = self._components.mean
        return self._components.variance.mean(0) + m.var(0, unbiased=False)

    def rsample(self, sample_shape=torch.Size()) -> Tensor:
        sample_shape = torch.Size(sample_shape)
        if sample_shape not in (torch.Size(), torch.Size([self.n_components])):
            raise ValueError(f"PerSampleGamma draws one value per component ({self.n_components}), got {tuple(sample_shape)}")
        return self._components.rsample()


@register_kl(PerSampleGamma, Gamma)
def _kl_per_sample_gamma(q: PerSampleGamma, p: Gamma) -> Tensor:
    return kl_divergence(q._components, p).mean(0)


class SVAEPerSampleIntegrator(SVAEIntegrator):
    """`svae` with the conjugate q(I) and q(b) conditioned on each profile sample.

    In `svae` the responsibility gate, and through it q(I) and q(b), is built
    from the MEAN profile, and I, b and the profile are then drawn
    independently: q(I) q(b) q(eta). On SBC data that family misses the
    dependence between the intensity and the profile -- the joint
    log-likelihood of the true latents beats every draw for bright reflections
    -- and q(I) is too narrow by about 2x in variance.

    Here, for each profile sample p_s (s = 1..S, the surrogate's own samples),

        r_s = i0_s p_s / (i0_s p_s + b0)                 the responsibility at p_s
        q(I | p_s) = Gamma(alpha_I + sum r_s c, tau + sum p_s m)
        q(b | p_s) = Gamma(alpha_b + sum (1 - r_s) c, beta_b + sum m)

    and the rate's draws pair (I_s, b_s) with the p_s they were built from,
    so the family is q(eta) q(I | eta) q(b | eta). The KL is
    KL(q(eta) || p(eta)) + E_q(eta)[KL(q(I | eta) || p(I)) + KL(q(b | eta) || p(b))],
    the last two averaged over the S samples. The exported qi_mean/qi_var
    and qbg_mean/qbg_var are the mixture's, so the profile's uncertainty adds
    to the conjugate variance.

    Only `gate: responsibility` with the conjugate intensity mean and variance
    is supported. `gate_detach` detaches each p_s inside its responsibility,
    as it detaches the mean profile in `svae`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.cfg.gate != "responsibility":
            raise ValueError("svae_per_sample supports gate: responsibility only")
        if self.cfg.intensity_variance != "conjugate" or self.cfg.intensity_mean != "conjugate" \
                or self.cfg.intensity_posterior == "amortized":
            raise ValueError("svae_per_sample supports the conjugate q(I) only")

    def _get_pixel_probs(self, qp, x_signal, raw_counts=None, mask=None):
        prf = qp.zp  # (S, B, P), the surrogate's profile samples
        m = mask.squeeze(-1) if mask is not None else torch.ones_like(prf[0])
        c = raw_counts * m
        b0 = (raw_counts.to(prf.dtype).masked_fill(m <= 0, float("nan"))
              .nanmedian(dim=-1, keepdim=True).values)
        b0 = torch.nan_to_num(b0, nan=0.0)  # (B, 1), as in `svae`
        prf_r = prf.detach() if getattr(self.cfg, "gate_detach", False) else prf
        s2 = (prf_r.pow(2) * m).sum(-1, keepdim=True).clamp_min(1e-12)
        i0 = ((prf_r * (c - b0) * m).sum(-1, keepdim=True) / s2).clamp_min(0.0)
        lam_s = i0 * prf_r
        signal_prob = lam_s / (lam_s + b0.clamp_min(1e-6) + 1e-12) * m
        return signal_prob, (1.0 - signal_prob) * m, prf

    def _get_qi(self, signal_prob, raw_counts, metadata, mean_prf, mask, x_signal=None):
        alpha_I, beta_I = self._intensity_prior(metadata, raw_counts.device)
        m = mask.squeeze(-1)
        alpha_ = alpha_I + (signal_prob * raw_counts * m).sum(-1)  # (S, B)
        beta_ = beta_I + (mean_prf * m).sum(-1)  # mean_prf holds the samples here
        return PerSampleGamma(alpha_, beta_)

    def _get_qbg(self, bg_prob, raw_counts, metadata, mask):
        alpha_bg, beta_bg = self._background_prior(metadata, raw_counts.device)
        m = mask.squeeze(-1)
        alpha_ = alpha_bg + (bg_prob * raw_counts * m).sum(-1)
        beta_ = beta_bg + m.sum(-1)
        return PerSampleGamma(alpha_, beta_.expand_as(alpha_))

    def _forward_impl(self, counts, shoebox, mask, metadata):
        out = super()._forward_impl(counts, shoebox, mask, metadata)
        fo = out["forward_out"]
        # the exported gate is per reflection; report its mean over the profile samples
        if fo.get("signal_prob") is not None and fo["signal_prob"].dim() == 3:
            fo["signal_prob"] = fo["signal_prob"].mean(0)
        return out

    def _add_deconvolved(self, forward_out, metadata):
        if {"i_deconv", "var_deconv"} & set(self.predict_keys):
            raise NotImplementedError("the deconvolved export assumes one Gamma q(I) per reflection")
