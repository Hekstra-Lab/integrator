"""Polychromatic Wilson loss whose intensity prior mean is set by empirical Bayes in closed form, separable in wavelength and resolution."""

import torch
from torch import Tensor

from integrator.model.loss.polychromatic_wilson_loss import PolychromaticWilsonLoss


class PolychromaticWilsonEBLoss(PolychromaticWilsonLoss):
    """`polychromatic_wilson` with the prior mean of I a free function of (lambda, d), refitted every epoch.

    The prior keeps the Wilson shape -- Gamma(1, tau) for an acentric,
    Gamma(1/2, tau/2) for a reflection flagged `centric` -- and replaces the
    mean 1/tau = G(lambda) exp(-B / 2d^2) by

        log E[I] = g_a + h_k

    with a the wavelength bin (`eb_lambda_bins` equal bins on
    [lambda_min, lambda_max], clamped) and k the resolution bin
    (`metadata["group_label"]`, the `n_bins` quantile bins of d the data module
    injects). At the end of every epoch the M-step of variational EM sets
    (g, h) to maximise the expected log prior over the epoch's q(I): for a
    Gamma prior of known shape the maximiser of the mean is the cell's mean of
    E_q[I] (for acentrics and centrics alike), and for the separable form it
    is found by alternating

        g_a = log mean_{j in a} E_q[I_j] exp(-h_k(j)),   h_k = log mean_{j in k} E_q[I_j] exp(-g_a(j))

    (`eb_iterations` sweeps; h is then centred to weighted mean 0, so g carries
    the scale). The step is damped in log space by `eb_damping`. Until the
    first M-step the prior is the parent's Wilson prior at its initial G and
    B. Nothing in the prior takes gradient: G(lambda) and B are frozen, and the
    parent's forward detaches tau because `refit_enabled` is set.

    Why: on mos010 photon counting one exponential in 1/d^2 cannot follow the
    true intensities. Under B 30.5 the ratio of mean I_true to the prior mean
    ran 1.00 at 1.7-2.0 A, 0.83 at 2.2-3.0 A, 1.45 at 3-4 A and 3.08 below 4 A;
    under the converged B 40.6 it was 1.45-1.73 at 1.7-2.0 A, which is where the
    outer-shell sums fell to 0.71 (scripts/sim_truth/prior_check.py).

    Caveat: the M-step is fitted to q, so where q collapses onto the prior (the
    weakest cells) the update moves slowly; the fixed point is the type-II
    maximum of the ELBO, biased wherever q is.
    """

    def __init__(
        self,
        *,
        eb_lambda_bins: int = 36,
        eb_damping: float = 1.0,
        eb_iterations: int = 10,
        eb_min_count: int = 30,
        **kwargs,
    ):
        kwargs["learn_G"] = False
        kwargs["learn_B"] = False
        super().__init__(**kwargs)
        if self._apply_polarization or self._apply_lorentz:
            raise ValueError("polychromatic_wilson_eb fits the whole prior mean; polarization and lorentz are not supported")
        n_d = int(kwargs.get("n_bins", 1))
        if n_d < 2:
            raise ValueError("polychromatic_wilson_eb needs n_bins >= 2 resolution bins (group_label)")
        if not 0 < eb_damping <= 1:
            raise ValueError("eb_damping must be in (0, 1]")
        self.eb_lambda_bins, self.eb_n_d = int(eb_lambda_bins), n_d
        self.eb_damping, self.eb_iterations, self.eb_min_count = float(eb_damping), int(eb_iterations), int(eb_min_count)
        self.register_buffer("eb_g", torch.zeros(self.eb_lambda_bins, dtype=torch.float64), persistent=True)
        self.register_buffer("eb_h", torch.zeros(n_d, dtype=torch.float64), persistent=True)
        self.register_buffer("eb_ready", torch.tensor(False), persistent=True)
        # the epoch's (lambda bin, d bin) cell sums of E_q[I] and counts
        self.register_buffer("_eb_sum", torch.zeros(self.eb_lambda_bins, n_d, dtype=torch.float64), persistent=False)
        self.register_buffer("_eb_n", torch.zeros(self.eb_lambda_bins, n_d, dtype=torch.float64), persistent=False)
        # makes the parent call _accumulate_fit in training, refit_prior at epoch end, and detach tau
        self.refit_enabled = True
        self.refit_residual = False

    def _cells(self, metadata: dict, device) -> tuple[Tensor, Tensor]:
        if "group_label" not in metadata:
            raise ValueError("polychromatic_wilson_eb needs metadata['group_label'] (the resolution bins)")
        lam = metadata["wavelength"].to(device).reshape(-1).double()
        lo, hi = float(self.spectrum.lam_mid - self.spectrum.lam_scale), float(self.spectrum.lam_mid + self.spectrum.lam_scale)
        a = ((lam - lo) / (hi - lo) * self.eb_lambda_bins).long().clamp(0, self.eb_lambda_bins - 1)
        k = metadata["group_label"].to(device).reshape(-1).long().clamp(0, self.eb_n_d - 1)
        return a, k

    def _get_tau(self, metadata: dict, s_sq: Tensor, device: torch.device) -> Tensor:
        tau = super()._get_tau(metadata, s_sq, device)
        if not bool(self.eb_ready):
            return tau
        a, k = self._cells(metadata, device)
        log_mean = self.eb_g.to(device)[a] + self.eb_h.to(device)[k]
        return torch.exp(-log_mean.clamp(-30.0, 30.0)).to(tau.dtype).reshape(tau.shape)

    def _accumulate_fit(self, counts, mask, s_sq, qi, metadata, device) -> None:
        with torch.no_grad():
            a, k = self._cells(metadata, device)
            m = qi.mean.detach().reshape(-1).double()
            idx = a * self.eb_n_d + k
            self._eb_sum.view(-1).scatter_add_(0, idx, m)
            self._eb_n.view(-1).scatter_add_(0, idx, torch.ones_like(m))

    def refit_prior(self, epoch: int) -> dict[str, float]:
        S, N = self._eb_sum, self._eb_n
        if float(N.sum()) == 0:
            return {}
        with torch.no_grad():
            ok = N >= self.eb_min_count
            Sw, Nw = torch.where(ok, S, torch.zeros_like(S)), torch.where(ok, N, torch.zeros_like(N))
            g = torch.log((Sw.sum(1) / Nw.sum(1).clamp_min(1)).clamp_min(1e-8))
            h = torch.zeros(self.eb_n_d, dtype=torch.float64, device=S.device)
            for _ in range(self.eb_iterations):
                num_h = (Sw * torch.exp(-g)[:, None]).sum(0)
                h = torch.where(Nw.sum(0) > 0, torch.log((num_h / Nw.sum(0).clamp_min(1)).clamp_min(1e-8)), h)
                num_g = (Sw * torch.exp(-h)[None, :]).sum(1)
                g = torch.where(Nw.sum(1) > 0, torch.log((num_g / Nw.sum(1).clamp_min(1)).clamp_min(1e-8)), g)
            wk = Nw.sum(0)
            shift = float((h * wk).sum() / wk.sum().clamp_min(1))
            h, g = h - shift, g + shift
            # a bin with no usable cell keeps its previous value
            g = torch.where(Nw.sum(1) > 0, g, self.eb_g)
            h = torch.where(Nw.sum(0) > 0, h, self.eb_h)
            eta = self.eb_damping if bool(self.eb_ready) else 1.0
            self.eb_g.copy_((1 - eta) * self.eb_g + eta * g)
            self.eb_h.copy_((1 - eta) * self.eb_h + eta * h)
            self.eb_ready.fill_(True)
            S.zero_()
            N.zero_()
        return {"eb_h_min": float(self.eb_h.min()), "eb_h_max": float(self.eb_h.max()),
                "eb_g_mean": float(self.eb_g.mean()), "eb_cells": float(ok.sum())}
