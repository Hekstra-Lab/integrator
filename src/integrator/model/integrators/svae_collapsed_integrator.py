"""SVAE trained on the collapsed bound: full-covariance q(eta), and (I, b) | eta integrated rather than approximated."""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor, nn

from integrator import configs
from integrator.model.integrators.base_integrator import _log_loss
from integrator.model.integrators.integrator_utils import IntegratorBaseOutputs, _assemble_outputs
from integrator.model.integrators.svae_integrator import SVAEIntegrator, _add_group_outputs
from integrator.model.distributions.profile_surrogates import ProfileSurrogateOutput
from integrator.model.posterior.collapsed import collapsed_log_z


class SVAECollapsedIntegrator(SVAEIntegrator):
    """`svae` with q(I), q(b) replaced by the exact (I, b) posterior given the profile, trained on

        L = E_{q(eta)}[ log Z(eta) ] - KL( q(eta) || p(eta) ),
        log Z(eta) = log integral integral p(c | I p(eta) + b) p(I) p(b) dI db

    (`integrator.model.posterior.collapsed.collapsed_log_z`: I on a grid, b
    integrated by Laplace), estimated with `collapsed_eta_samples` reparameterised
    draws eta_s = mu + L z. q(eta) = N(mu, L L^T): mu and the diagonal of L from
    the profile surrogate's own heads (`eta_delta`, `eta_scale`, as in
    `streak_kmix_v2_profile`), the strict lower triangle L_ij = sigma_i o_ij(x)
    from a head of this class started at zero, so training starts from the
    diagonal q(eta). p(eta) = N(theta0 + f(m), diag sigma^2), the surrogate's
    prior. There is no gate, no conjugate q(I) or q(b), and no `loss.forward`:
    the Wilson loss is used only for its priors (`intensity_prior_params`,
    `_background_prior`).

    On the mos010 prior SBC set the conjugate q(I) of `svae` has 12 var(u) 2.15
    because it conditions on the gate's split at its mean; the collapsed (I, b)
    posterior with the amortized q(eta) was 1.03 (I), 1.00 (b), and with a
    full-covariance Laplace q(eta) every quantity was 0.98-1.02. A tight bound is
    also the standard remedy for hyperparameter bias under a loose one (Turner &
    Sahani 2011), which may matter for the narrowing of a learned profile prior.

    Needs `streak_kmix_v2_profile` with `shape_posterior: amortized` and
    `residual: none`. One encoder (the profile's).
    """

    REQUIRED_ENCODERS = {
        "profile": ("profile_encoder", configs.ProfileEncoderArgs),
    }

    def __init__(self, cfg, loss, encoders, surrogates, optimizer=None):
        super().__init__(cfg, loss, encoders, surrogates, optimizer)
        qp = self.surrogates["qp"]
        if not hasattr(qp, "profile_from_eta") or getattr(qp, "post", None) != "amortized" \
                or getattr(qp, "residual", "") != "none":
            raise ValueError("svae_collapsed needs streak_kmix_v2_profile, shape_posterior amortized, residual none")
        self.gate_head = None
        if "signal" not in self.encoders:
            self.encoders["signal"] = nn.Identity()
        ne = qp.NE
        self.eta_offdiag = nn.Linear(qp.eta_delta.in_features, ne * (ne - 1) // 2)
        nn.init.zeros_(self.eta_offdiag.weight)
        nn.init.zeros_(self.eta_offdiag.bias)
        self.register_buffer("_tril", torch.tril_indices(ne, ne, -1), persistent=False)
        self.n_eta = int(getattr(cfg, "collapsed_eta_samples", 2))
        self.grid = int(getattr(cfg, "collapsed_grid", 64))

    def _q_eta(self, x_profile: Tensor, metadata: dict, mask: Tensor):
        qp = self.surrogates["qp"]
        m = mask.reshape(mask.shape[0], -1).to(x_profile.dtype)
        phi, fxy, eta_mean = qp.eta_prior(metadata, m)
        loc, sd = qp.q_eta_from_raw(eta_mean, qp.eta_delta(x_profile), qp.eta_scale(x_profile))
        sig = qp.eta_sd.to(loc)
        B, ne = loc.shape
        L = torch.diag_embed(sd)
        off = self.eta_offdiag(x_profile) * sig[self._tril[0]]
        L = L.index_put((torch.arange(B, device=loc.device)[:, None].expand(-1, off.shape[1]),
                         self._tril[0].expand(B, -1), self._tril[1].expand(B, -1)), off)
        return phi, fxy, eta_mean, loc, L, sig, m

    def _bound(self, counts, shoebox, mask, metadata):
        qp = self.surrogates["qp"]
        counts = torch.clamp(counts, min=0)
        b = shoebox.shape[0]
        sb = (shoebox * mask).reshape(b, 1, *self.shoebox_shape)
        x = self.encoders["profile"](self._profile_encoder_input(sb, metadata))
        phi, fxy, eta_mean, loc, L, sig, m = self._q_eta(x, metadata, mask)
        dev = counts.device
        a_I, r_I = self.loss.intensity_prior_params(metadata, dev)
        a_b, r_b = self._background_prior(metadata, dev)
        z = torch.randn((self.n_eta,) + loc.shape, device=dev, dtype=loc.dtype)
        etas = loc[None] + torch.einsum("bij,sbj->sbi", L, z)
        c = counts.reshape(b, -1)
        log_z, posts, profiles = [], [], []
        for s in range(self.n_eta):
            p = qp.profile_from_eta(phi, fxy, etas[s], m)
            lz, q = collapsed_log_z(c, m, p, a_I, r_I, a_b, r_b, grid=self.grid)
            log_z.append(lz)
            posts.append(q)
            profiles.append(p)
        e_log_z = torch.stack(log_z).mean(0).to(loc.dtype)
        # KL( N(loc, L L^T) || N(eta_mean, diag sig^2) )
        Ls = L / sig[None, :, None]
        d = (loc - eta_mean) / sig
        kl = 0.5 * ((Ls**2).sum((-1, -2)) + (d**2).sum(-1) - loc.shape[1]
                    - 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2) / sig).sum(-1))
        return e_log_z, kl, posts, profiles, loc, L

    def _forward_impl(self, counts, shoebox, mask, metadata) -> dict[str, Any]:
        e_log_z, kl, posts, profiles, loc, L = self._bound(counts, shoebox, mask, metadata)
        with torch.no_grad():
            mI = torch.stack([q.mean_var_I()[0] for q in posts])
            vI = torch.stack([q.mean_var_I()[1] for q in posts])
            mb = torch.stack([q.mean_var_b()[0] for q in posts])
            vb = torch.stack([q.mean_var_b()[1] for q in posts])
            i_mean, i_var = mI.mean(0), vI.mean(0) + mI.var(0, unbiased=False)
            b_mean, b_var = mb.mean(0), vb.mean(0) + mb.var(0, unbiased=False)
            f = loc.dtype
            qi = torch.distributions.Gamma((i_mean**2 / i_var).to(f), (i_mean / i_var).to(f))
            qbg = torch.distributions.Gamma((b_mean**2 / b_var).to(f), (b_mean / b_var).to(f))
            zp = torch.stack(profiles, 1).to(f)  # (B, S, P)
            S = zp.shape[1]
            zI = qi.sample([S]).permute(1, 0).unsqueeze(-1)
            zbg = qbg.sample([S]).permute(1, 0).unsqueeze(-1)
        qp_out = ProfileSurrogateOutput(zp=zp.permute(1, 0, 2), mean_profile=zp.mean(1), loc=loc.detach(),
                                        scale=L.detach().pow(2).sum(-1).sqrt(), prior_scale=None, kl=kl.detach())
        out = IntegratorBaseOutputs(rates=zI * zp + zbg, counts=counts, mask=mask, qbg=qbg,
                                    qp=qp_out, qi=qi, zp=zp, zbg=zbg, metadata=metadata)
        out = _assemble_outputs(out)
        out["eta_loc"] = loc.detach()
        out["eta_chol"] = L.detach()
        _add_group_outputs(out, metadata, self.loss)
        return {"forward_out": out, "qp": qp_out, "qi": qi, "qbg": qbg}

    def _step(self, batch, step: Literal["train", "val"]):
        counts, shoebox, mask, metadata = batch
        e_log_z, kl, *_ = self._bound(counts, shoebox, mask, metadata)
        nll = -e_log_z.mean()
        klm = kl.mean()
        total = nll + klm
        _log_loss(self, kl=klm, nll=nll, total_loss=total, step=step, kl_components={"kl_prf": klm})
        zero = torch.zeros((), device=total.device)
        return {"loss": total, "forward_out": {},
                "loss_components": {"loss": total.detach(), "nll": nll.detach(), "kl": klm.detach(),
                                    "kl_prf": klm.detach(), "kl_i": zero, "kl_bg": zero}}
