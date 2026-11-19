"""Three SVAE variants that differ in how the signal gate is built.

The gate's job is the Poisson split. For `c_i = s_i + b_i` with
`s_i ~ Poisson(I p_i)` and `b_i ~ Poisson(B)`, the posterior share of a
count belonging to signal is

    r_i = I p_i / (I p_i + B) = sigmoid( log(I p_i) - log B )

so the responsibility IS the profile put through a sigmoid, and the only
question is where `I` and `B` come from. `scripts/gate_study.py` measures
that on simulated streaky Gaussians: the form is unbiased to 0.1% given
the true I and B, and everything worse comes from estimating them.

`SVAEIntegrator` is left exactly as it is. These are separate classes with
their own registry entries, so every run already on disk keeps the
architecture it trained with.

What each one is for:

- `svae_conjugate` -- ONE encoder. Nothing reads `x_signal` once the gate
  is computed and q(I) is conjugate, so the second encoder is not built.
  Measured on the base class: its 387,008 parameters receive zero gradient
  under a computed gate, against 14 gradient tensors under the learned
  head. This is the arm that asks whether the second encoder is needed.

- `svae_bg` -- two encoders, the second predicting the BACKGROUND rather
  than the gate. A free per-pixel gate has no tie to the profile and
  measured +54% bias; `B` is one positive number per reflection that the
  gate genuinely needs and that the box median estimates badly (a Poisson
  median is 0 below rate 0.7, which saturates the gate). This asks whether
  amortizing the quantity the gate needs works where amortizing the gate
  itself did not.

- `svae_calib` -- two encoders, the second predicting a slope and offset
  on the log-odds, `sigmoid(a * log-odds + b)`, initialised at `a = 1`,
  `b = 0`, which is exactly the responsibility. A tied, two-parameter
  correction to a form we have reason to believe. The simulation expects
  it to stay near the identity: tempering the slope off 1 was worse in
  both directions, +88% at 0.7 and +40% at 1.3 against +23% at 1.0. It is
  therefore a falsification test, not a likely win.
"""

from __future__ import annotations

import torch
from torch import nn

from integrator import configs
from integrator.model.integrators.svae_integrator import SVAEIntegrator

__all__ = [
    "SVAEBackgroundIntegrator",
    "SVAECalibIntegrator",
    "SVAEConjugateIntegrator",
]


def _joint_fit(prf, counts, mask):
    """Least squares for (I, B) together against the design `[p, 1]`.

    Every background estimator that is an order statistic is wrong in its
    own direction: the median quantises to zero on a faint background, and
    the mean of the dim half sits below the mean of the whole by
    construction. `c = I p + B` is linear in both and `p` is known, so the
    2x2 normal equation needs neither. Measured, it takes the median bias
    at low SNR from +91% to -1.7%.
    """
    m = mask
    c = counts * m
    n = m.sum(-1, keepdim=True).clamp_min(1.0)
    spp = (prf.pow(2) * m).sum(-1, keepdim=True)
    sp = (prf * m).sum(-1, keepdim=True)
    spc = (prf * c).sum(-1, keepdim=True)
    sc = c.sum(-1, keepdim=True)
    det = (spp * n - sp * sp).clamp_min(1e-12)
    i0 = ((n * spc - sp * sc) / det).clamp_min(0.0)
    b0 = ((spp * sc - sp * spc) / det).clamp_min(0.0)
    return i0, b0


def _profile_for_gate(cfg, prf):
    """The profile the gate is evaluated at, detached where asked.

    Leaving it attached lets `KL(q(I) || prior)` pull the profile down
    where the counts are large, because the responsibility rises with `p`.
    """
    return prf.detach() if getattr(cfg, "gate_detach", False) else prf


class SVAEConjugateIntegrator(SVAEIntegrator):
    """One encoder. The gate is computed; nothing else reads the shoebox.

    `x_signal` has three consumers in the base class: `gate_head`, the qi
    surrogate under `intensity_posterior: amortized`, and
    `_background_posterior`, which accepts it and ignores it. With a
    computed gate and a conjugate q(I) none of them fires, so the encoder
    is not built at all rather than built and left at its initialisation.
    """

    REQUIRED_ENCODERS = {
        "profile": ("profile_encoder", configs.ProfileEncoderArgs),
    }

    def __init__(self, cfg, loss, encoders, surrogates, optimizer=None):
        super().__init__(cfg, loss, encoders, surrogates, optimizer)
        # `gate_head` is 28,665 weights that a computed gate never uses.
        self.gate_head = None
        # The base `_forward_impl` indexes `encoders["signal"]`
        # unconditionally. An identity keeps that call valid and costs
        # nothing, and no parameter is created.
        if "signal" not in self.encoders:
            self.encoders["signal"] = nn.Identity()

    def _get_pixel_probs(self, qp, x_signal, raw_counts=None, mask=None):
        if raw_counts is None:
            raise ValueError(
                "svae_conjugate derives the gate from the data and needs "
                "raw_counts"
            )
        prf = qp.mean_profile
        m = mask.squeeze(-1) if mask is not None else torch.ones_like(prf)
        prf_r = _profile_for_gate(self.cfg, prf)
        i0, b0 = _joint_fit(prf_r, raw_counts, m)
        lam = i0 * prf_r
        signal_prob = (lam / (lam + b0.clamp_min(1e-6) + 1e-12)) * m
        return signal_prob, (1.0 - signal_prob) * m, prf


class SVAEBackgroundIntegrator(SVAEIntegrator):
    """Two encoders, the second predicting the background.

    The gate still has no free parameters: it is `I p / (I p + B)` with `B`
    from the encoder and `I` from profile-weighted least squares given that
    `B`. What is amortized is the quantity the gate needs and the box
    median estimates worst.
    """

    REQUIRED_ENCODERS = {
        "profile": ("profile_encoder", configs.ProfileEncoderArgs),
        "signal": ("intensity_encoder", configs.IntensityEncoderArgs),
    }

    def __init__(self, cfg, loss, encoders, surrogates, optimizer=None):
        super().__init__(cfg, loss, encoders, surrogates, optimizer)
        self.gate_head = None
        # one positive number per reflection
        self.bg_head = nn.Linear(self.cfg.encoder_out, 1)
        # SMALL, not zero. With zero weights `d(out)/d(x) = W = 0`, so no
        # gradient reaches the encoder and it never starts learning -- the
        # dead second encoder this class exists to avoid, reintroduced by
        # the initialisation. Measured: 748,180 of 1,135,448 parameters
        # received gradient with zeros, all of them with this.
        nn.init.normal_(self.bg_head.weight, std=1e-3)
        # softplus(0) = 0.693, so the background starts near two thirds of
        # a count rather than at zero, which is where the gate saturates
        nn.init.zeros_(self.bg_head.bias)

    def _get_pixel_probs(self, qp, x_signal, raw_counts=None, mask=None):
        if raw_counts is None:
            raise ValueError("svae_bg needs raw_counts for the intensity fit")
        prf = qp.mean_profile
        m = mask.squeeze(-1) if mask is not None else torch.ones_like(prf)
        prf_r = _profile_for_gate(self.cfg, prf)
        b0 = nn.functional.softplus(self.bg_head(x_signal)).clamp_min(1e-6)
        # I by profile-weighted least squares GIVEN that background
        c = raw_counts * m
        s2 = (prf_r.pow(2) * m).sum(-1, keepdim=True).clamp_min(1e-12)
        i0 = ((prf_r * (c - b0) * m).sum(-1, keepdim=True) / s2).clamp_min(0.0)
        lam = i0 * prf_r
        signal_prob = (lam / (lam + b0 + 1e-12)) * m
        return signal_prob, (1.0 - signal_prob) * m, prf


class SVAECalibIntegrator(SVAEIntegrator):
    """Two encoders, the second calibrating the log-odds.

    `sigmoid(a * (log(I p) - log B) + b)` with `(a, b)` from the encoder,
    initialised at `a = 1`, `b = 0`, which is the responsibility exactly.
    The gate can only rescale and shift a quantity derived from the
    profile; it cannot invent a shape, which is what the free per-pixel
    head does.
    """

    REQUIRED_ENCODERS = {
        "profile": ("profile_encoder", configs.ProfileEncoderArgs),
        "signal": ("intensity_encoder", configs.IntensityEncoderArgs),
    }

    def __init__(self, cfg, loss, encoders, surrogates, optimizer=None):
        super().__init__(cfg, loss, encoders, surrogates, optimizer)
        self.gate_head = None
        self.calib_head = nn.Linear(self.cfg.encoder_out, 2)
        # Small rather than zero, for the same reason as `bg_head`: zero
        # weights cut the encoder out of the graph entirely. The bias below
        # is what starts the model at the exact responsibility; the weights
        # only have to be small enough not to move it far.
        nn.init.normal_(self.calib_head.weight, std=1e-3)
        with torch.no_grad():
            # softplus(0.5413) = 1.0, so slope starts at exactly 1
            self.calib_head.bias.copy_(torch.tensor([0.5413, 0.0]))

    def _get_pixel_probs(self, qp, x_signal, raw_counts=None, mask=None):
        if raw_counts is None:
            raise ValueError("svae_calib needs raw_counts for the log-odds")
        prf = qp.mean_profile
        m = mask.squeeze(-1) if mask is not None else torch.ones_like(prf)
        prf_r = _profile_for_gate(self.cfg, prf)
        i0, b0 = _joint_fit(prf_r, raw_counts, m)
        odds = (torch.log((i0 * prf_r).clamp_min(1e-30))
                - torch.log(b0.clamp_min(1e-30)))
        ab = self.calib_head(x_signal)
        a = nn.functional.softplus(ab[..., :1])
        b = ab[..., 1:]
        signal_prob = torch.sigmoid(a * odds + b) * m
        return signal_prob, (1.0 - signal_prob) * m, prf
