"""Per-observation conjugate Bayesian integration with a Wilson prior.

Generative model. For shoebox i (HKL h(i)), pixel p:

    counts_{i,p} ~ Poisson(rate_{i,p})
    rate_{i,p}   = s_i * I_i * profile_{i,p} + bg_i
    I_i          ~ Gamma(alpha_W, tau_{h(i)})       (Wilson prior)

The per-pixel rate is *affine* in I_i (the +bg_i term shifts it), not
linear, so the I-likelihood is not Poisson-in-I and the exact posterior on
I_i is not Gamma. Introducing a per-pixel signal/background split (Poisson
thinning / data augmentation) makes the model conditionally conjugate, and
the resulting Gamma is the mean-field (CAVI) variational factor q(I_i)
given (profile, bg, scale) — *exact only in the bg -> 0 limit*, and an
excellent approximation when signal and background are well separated:

    pi_{i,p} = s_i * I_i_hat * profile_{i,p} /
               (s_i * I_i_hat * profile_{i,p} + bg_i)        (responsibility)

    alpha_i = alpha_W + sum_p  pi_{i,p} * c_{i,p}      * mask_{i,p}
    beta_i  = tau_h   + sum_p  s_i      * profile_{i,p} * mask_{i,p}
    q(I_i)  = Gamma(alpha_i, beta_i)

beta_i is independent of I_i; only pi (hence alpha_i) depends on it. We
solve the fixed point I_i_hat = alpha_i / beta_i by iterating the
responsibility update to convergence (cold start at the Wilson mean
1 / tau_h) and obtain the training gradient by one-step implicit
differentiation — see `_conjugate_em`. `n_em_iters` is a max iteration
count with early stop at `em_tol`; bright, low-background spots converge
slowest.

What the neural net does:
    - Encoders predict q(profile_i), q(bg_i) per shoebox.
    - Scale s_i from physics (LP + optional learnable correction).
    - I_i is *derived* per shoebox — no encoder for I, no embedding.

The loss supplies the matching Wilson KL(q(I_i) || Gamma(alpha_W, tau_h))
at pi_weight = 1; __init__ asserts that coupling (pi_weight == 1,
alpha_W == 1 to match the loss's hard-coded acentric prior shape, and no
learned concentration prior).

Unlike the merging variant, each observation is processed independently —
no per-HKL aggregation, no EMA buffer, no grouped sampler required. The
per-obs q(I_i) is a Bayesian, DIALS-style profile-fitted intensity, not a
merged crystallographic posterior over the shared I_h.
"""

from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Gamma

from integrator import configs
from integrator.model.integrators.base_integrator import (
    BaseIntegrator,
    _log_loss,
)
from integrator.model.integrators.hierarchical_integrator import (
    _add_group_outputs,
    _get_normalized_position,
    _sample_profile,
)
from integrator.model.integrators.integrator_utils import (
    IntegratorBaseOutputs,
    _assemble_outputs,
)
from integrator.model.scaling.chebyshev_scale import (
    ChebyshevScale,
    MLPScale,
    SpatialChebyshevScale,
)


class ConjugateIntegrator(BaseIntegrator):
    """Per-observation conjugate intensity. DIALS-style Bayesian integration."""

    REQUIRED_ENCODERS = {
        "profile": configs.ProfileEncoderArgs,
        "k_bg": configs.IntensityEncoderArgs,
        "r_bg": configs.IntensityEncoderArgs,
    }

    def __init__(
        self,
        cfg: configs.IntegratorCfg,
        loss: nn.Module,
        encoders: dict[str, nn.Module],
        surrogates: dict[str, nn.Module],
    ):
        super().__init__(cfg, loss, encoders, surrogates)

        self.alpha_W = float(cfg.wilson_alpha)
        self.n_em_iters = int(cfg.n_em_iters)
        self.em_tol = float(cfg.em_tol)
        self.sample_I = bool(cfg.sample_I_h)
        self.exact_posterior_n_nuisance = int(cfg.exact_posterior_n_nuisance)
        self.exact_posterior_n_grid = int(cfg.exact_posterior_n_grid)

        # Consistency guard. The per-obs conjugate posterior
        # Gamma(alpha_W + ., tau + .) is the exact mean-field minimizer of the
        # per-obs ELBO only if the loss supplies the matching intensity term
        # KL(q(I) || Gamma(alpha_W, tau)) at full weight. MonochromaticWilsonLoss
        # hard-codes the prior shape to alpha = 1 (acentric Wilson) and scales
        # KL_i by pi_weight, so require pi_weight == 1, alpha_W == 1, and no
        # learned/continuous concentration prior. (The merging variant builds
        # its own KL and uses the opposite pi_weight == 0 convention.)
        pi_weight = float(getattr(self.loss, "pi_weight", 1.0))
        if abs(pi_weight - 1.0) > 1e-8:
            raise ValueError(
                "ConjugateIntegrator requires loss pi_weight == 1.0 so the "
                "loss's KL(qi || Wilson) is exactly the per-obs closed-form "
                f"Gamma-Gamma KL of the conjugate posterior; got {pi_weight}."
            )
        if abs(self.alpha_W - 1.0) > 1e-8:
            raise ValueError(
                "ConjugateIntegrator requires wilson_alpha == 1.0: the Wilson "
                "loss hard-codes the intensity prior shape to alpha = 1, so the "
                f"conjugate update must match it; got alpha_W={self.alpha_W}."
            )
        if (
            getattr(self.loss, "concentration_fn", None) is not None
            or getattr(self.loss, "learn_concentration", False)
        ):
            raise ValueError(
                "ConjugateIntegrator requires the default fixed Wilson prior "
                "shape (alpha = 1). A learned/continuous concentration prior "
                "makes the loss KL_i inconsistent with the conjugate update."
            )

        # Scale function (optional). With scale_mlp=false and scale_spatial=false,
        # the default ChebyshevScale with degree=0 gives a learnable constant
        # divided by lp (i.e. just LP correction).
        if cfg.scale_mlp:
            self.scale_fn = MLPScale(
                hidden_dim=cfg.scale_mlp_hidden,
                n_layers=cfg.scale_mlp_layers,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
                beam_center=cfg.scale_beam_center,
                r_max=cfg.scale_r_max,
                d_min=getattr(cfg, "dmin", 1.0),
                d_max=60.0,
            )
        elif cfg.scale_spatial:
            self.scale_fn = SpatialChebyshevScale(
                degree_frame=cfg.scale_degree,
                degree_radius=cfg.scale_degree_radius,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
                beam_center=cfg.scale_beam_center,
                r_min=cfg.scale_r_min,
                r_max=cfg.scale_r_max,
            )
        else:
            self.scale_fn = ChebyshevScale(
                degree=cfg.scale_degree,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
            )

    # %%
    def _get_scale(self, metadata: dict, device: torch.device) -> Tensor:
        frame = metadata["xyzcal.px.2"].to(device).float()
        lp = metadata["lp"].to(device).float().clamp(min=1e-8)
        if isinstance(self.scale_fn, MLPScale):
            x_det = metadata["xyzcal.px.0"].to(device).float()
            y_det = metadata["xyzcal.px.1"].to(device).float()
            d = metadata["d"].to(device).float()
            return self.scale_fn(frame, x_det, y_det, lp, d)
        elif isinstance(self.scale_fn, SpatialChebyshevScale):
            x_det = metadata["xyzcal.px.0"].to(device).float()
            y_det = metadata["xyzcal.px.1"].to(device).float()
            return self.scale_fn(frame, x_det, y_det) / lp
        else:
            return self.scale_fn(frame) / lp

    def _wilson_tau(self, d: Tensor) -> Tensor:
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))
        return self.loss._get_tau({"d": d}, s_sq, d.device)

    def _conjugate_em(
        self,
        counts: Tensor,
        profile_mean: Tensor,
        bg_mean: Tensor,
        scale: Tensor,
        tau: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Solve the mean-field (CAVI) fixed point. Returns (alpha, beta, pi).

        The coordinate-ascent VI update for the per-pixel signal/background
        split uses the responsibility

            pi_p = s*Itil*prof_p / (s*Itil*prof_p + bg),
            Itil = exp(E_q[log I]) = exp(psi(alpha) - log beta),

        i.e. the GEOMETRIC-mean intensity exp(E[log I]), not the arithmetic
        mean alpha/beta. This is the exact CAVI update for q(I) (Bishop PRML
        10.9; Blei et al. 2017); using alpha/beta instead is an EM/MAP-style
        approximation that over-weights the signal. Given pi, the Gamma factor
        is alpha = alpha_W + sum_p pi_p c_p, beta = tau + sum_p s prof_p (beta
        is independent of I). We solve the fixed point in two phases:

        1. Converge Itil* WITHOUT tracking gradients (an inner inference loop),
           iterating to `em_tol` or `n_em_iters`.
        2. Differentiate the *converged* fixed point exactly via the implicit
           function theorem. With f(Itil, theta) the one-step map, the solution
           derivative is

               dItil*/dtheta = (d_theta f) / (1 - K),   K = df/dItil |_{Itil*},

           the (1 - K) factor accounting for Itil appearing on both sides of
           the fixed point. We realise it with one gradient-carrying step whose
           Jacobian is corrected by 1/(1 - K): `Itil_implicit` has value ~ Itil*
           but gradient dItil*/dtheta, so evaluating alpha there gives the exact
           total d(alpha)/d(theta). A naive one-step gradient (K = 0) is biased
           by 1 - K, and K is NOT small here (K ~ 0.6); full BPTT through the
           solve is correct only as iters -> inf and costs the whole tape, so we
           use the closed-form correction instead.

        K = trigamma(alpha) * sum_p c_p pi_p (1 - pi_p): df/dalpha = Itil*trigamma(alpha)
        and dalpha/dItil = sum_p c_p pi_p(1-pi_p)/Itil. At a stable fixed point
        0 <= K < 1.
        """
        s_prof = scale.unsqueeze(-1) * profile_mean  # (B, P), carries gradient
        cm = counts * mask  # (B, P)
        bg = bg_mean.unsqueeze(-1)

        # beta is constant in I and carries gradient through scale / profile.
        beta = tau + (s_prof * mask).sum(dim=-1)
        log_beta = beta.clamp(min=1e-12).log()

        def em_map(I_tilde: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            """One CAVI step from geometric-mean intensity I_tilde."""
            signal_rate = I_tilde.unsqueeze(-1) * s_prof
            pi = signal_rate / (signal_rate + bg).clamp(min=1e-12)
            alpha = self.alpha_W + (pi * cm).sum(dim=-1)
            I_tilde_new = torch.exp(torch.digamma(alpha.clamp(min=1e-6)) - log_beta)
            return alpha, pi, I_tilde_new

        # Phase 1: converge the geometric-mean fixed point (no gradient).
        with torch.no_grad():
            I_tilde = 1.0 / tau.clamp(min=1e-12)  # Wilson-mean cold start (B,)
            for _ in range(self.n_em_iters):
                _, _, I_new = em_map(I_tilde)
                rel = (I_new - I_tilde).abs() / I_tilde.clamp(min=1e-12)
                I_tilde = I_new
                if torch.all(rel < self.em_tol):
                    break

            # Fixed-point Jacobian K = dI_tilde_new/dI_tilde at Itil* (detached).
            alpha, pi, _ = em_map(I_tilde)
            trigamma = torch.special.polygamma(1, alpha.clamp(min=1e-6))
            K = trigamma * (cm * pi * (1.0 - pi)).sum(dim=-1)
            K = K.clamp(max=1.0 - 1e-3)

        # Phase 2: implicit-function gradient via a 1/(1-K)-corrected step.
        _, _, f = em_map(I_tilde)  # value ~ Itil*, carries d_theta f
        I_implicit = I_tilde + (f - I_tilde) / (1.0 - K)  # grad = dItil*/dtheta
        alpha, pi, _ = em_map(I_implicit)
        return alpha, beta, pi

    # ------------------------------------------------------------------
    # Calibrated intensity posterior (inference/export only).
    #
    # The mean-field Gamma from `_conjugate_em` is the cheap, differentiable
    # training device, but its variance is 2-5x too narrow (worst at high
    # background) because the data augmentation fixes the per-pixel signal/bg
    # split and discards the I-z allocation uncertainty (see
    # docs/conjugate_integrator.md S10). For trustworthy sigma(I) we instead
    # compute the *exact* collapsed posterior p(I | counts, profile, bg, scale)
    # by 1-D quadrature (Fix A), and propagate nuisance uncertainty over
    # q(profile), q(bg) via the law of total variance (Fix B).
    # ------------------------------------------------------------------

    def _quad_moments(
        self,
        counts: Tensor,
        mask: Tensor,
        e: Tensor,
        bg: Tensor,
        tau: Tensor,
        grid: Tensor,
        chunk: int,
    ) -> tuple[Tensor, Tensor]:
        """Exact E[I], Var[I] of the collapsed posterior by 1-D quadrature.

        Args:
            e: signal exposure per pixel, scale * profile, shape (B, P).
            bg: per-shoebox background rate, shape (B,).
            tau: Wilson prior rate, shape (B,).
            grid: per-shoebox intensity grid, shape (B, G).
            chunk: grid-points processed at once (bounds the (B, chunk, P) tensor).

        The collapsed log-posterior (augmentation-free, exact) is
            log p(I|c) = (alpha_W - 1) log I - (tau + sum_p e_p mask_p) I
                         + sum_p mask_p c_p log(e_p I + bg) + const.
        """
        G = grid.shape[1]
        cm = counts * mask  # (B, P)
        lin_coef = tau + (e * mask).sum(dim=-1)  # (B,) coefficient on I
        bg_u = bg.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        log_unnorm = torch.empty_like(grid)
        for lo in range(0, G, chunk):
            gI = grid[:, lo : lo + chunk]  # (B, g)
            rate = e[:, None, :] * gI[:, :, None] + bg_u  # (B, g, P)
            dterm = (cm[:, None, :] * torch.log(rate.clamp(min=1e-30))).sum(dim=-1)
            log_unnorm[:, lo : lo + chunk] = (
                (self.alpha_W - 1.0) * torch.log(gI.clamp(min=1e-30))
                - lin_coef[:, None] * gI
                + dterm
            )
        dw = torch.diff(grid, dim=1, prepend=grid[:, :1]).clamp(min=1e-30).log()
        w = torch.softmax(log_unnorm + dw, dim=1)  # (B, G) normalized mass
        m1 = (w * grid).sum(dim=-1)
        m2 = (w * grid.pow(2)).sum(dim=-1)
        return m1, (m2 - m1.pow(2)).clamp(min=0.0)

    @torch.no_grad()
    def exact_intensity_posterior(
        self,
        counts: Tensor,
        shoebox: Tensor,
        mask: Tensor,
        metadata: dict,
        *,
        n_nuisance: int = 16,
        n_grid: int = 1024,
        grid_chunk: int = 128,
    ) -> dict[str, Tensor]:
        """Calibrated per-observation intensity posterior (Fix A + Fix B).

        Use at inference/export for trustworthy sigma(I); training keeps the
        cheap mean-field Gamma. Returns per-observation tensors (all shape (B,)):
        `mean`, `var`, `std`, and `alpha`/`beta` of a Gamma moment-matched to
        (mean, var). With `n_nuisance <= 1` only Fix A is applied (quadrature at
        the nuisance posterior means); otherwise nuisance uncertainty is
        propagated by Monte Carlo over q(profile), q(bg).
        """
        counts = counts.clamp(min=0)
        B = shoebox.shape[0]
        device = shoebox.device
        shoebox_reshaped = (shoebox * mask).reshape(B, 1, *self.shoebox_shape)

        position = _get_normalized_position(metadata, device)
        x_profile = self.encoders["profile"](shoebox_reshaped, position=position)
        x_k_bg = self.encoders["k_bg"](shoebox_reshaped)
        x_r_bg = self.encoders["r_bg"](shoebox_reshaped)
        qbg = self.surrogates["qbg"](x_k_bg, x_r_bg)
        prf_labels = metadata.get(
            "profile_group_label", metadata.get("group_label")
        )
        prf_labels = prf_labels.long() if prf_labels is not None else None
        qp = self.surrogates["qp"](
            x_profile,
            mc_samples=max(n_nuisance, 1),
            group_labels=prf_labels,
            metadata=metadata,
        )

        scale = self._get_scale(metadata, device)  # (B,)
        tau = self._wilson_tau(metadata["d"].to(device).float())  # (B,)
        counts = counts.to(device)
        mask = mask.to(device)

        # Grid range from a mean-field pass at the posterior means.
        alpha_mf, beta_mf, _ = self._conjugate_em(
            counts, qp.mean_profile, qbg.mean, scale, tau, mask
        )
        mf_mean = alpha_mf / beta_mf
        mf_std = alpha_mf.sqrt() / beta_mf
        std_eff = 3.0 * mf_std  # exact posterior is up to ~2.5x wider
        lo = (mf_mean - 8.0 * std_eff).clamp(min=1e-8)
        hi = torch.maximum(mf_mean + 12.0 * std_eff, lo + 1e-3)
        steps = torch.linspace(0.0, 1.0, n_grid, device=device)
        grid = lo[:, None] + steps[None, :] * (hi - lo)[:, None]  # (B, G)

        # Nuisance samples (Fix B); n_nuisance <= 1 -> Fix A at the means.
        if n_nuisance <= 1:
            prof_samps = qp.mean_profile.unsqueeze(0)  # (1, B, P)
            bg_samps = qbg.mean.unsqueeze(0)  # (1, B)
        else:
            prof_samps = _sample_profile(qp, n_nuisance).permute(1, 0, 2)  # (S,B,P)
            bg_samps = qbg.rsample([n_nuisance])  # (S, B)

        means, varis = [], []
        for m in range(prof_samps.shape[0]):
            e = scale.unsqueeze(-1) * prof_samps[m]  # (B, P)
            mu, v = self._quad_moments(
                counts, mask, e, bg_samps[m], tau, grid, grid_chunk
            )
            means.append(mu)
            varis.append(v)
        means = torch.stack(means)  # (S, B)
        varis = torch.stack(varis)  # (S, B)

        # Law of total variance over the nuisance posterior.
        mean = means.mean(dim=0)
        var = (varis.mean(dim=0) + means.var(dim=0, unbiased=False)).clamp(min=1e-12)
        return {
            "mean": mean,
            "var": var,
            "std": var.sqrt(),
            "alpha": mean.pow(2) / var,
            "beta": mean / var,
        }

    # ------------------------------------------------------------------

    def _forward_impl(
        self,
        counts: Tensor,
        shoebox: Tensor,
        mask: Tensor,
        metadata: dict,
    ) -> dict[str, Any]:
        counts = counts.clamp(min=0)
        B = shoebox.shape[0]
        device = shoebox.device
        shoebox_masked = shoebox * mask
        shoebox_reshaped = shoebox_masked.reshape(B, 1, *self.shoebox_shape)

        # Encoders: profile + bg only
        position = _get_normalized_position(metadata, device)
        x_profile = self.encoders["profile"](
            shoebox_reshaped, position=position
        )
        x_k_bg = self.encoders["k_bg"](shoebox_reshaped)
        x_r_bg = self.encoders["r_bg"](shoebox_reshaped)

        qbg = self.surrogates["qbg"](x_k_bg, x_r_bg)
        prf_labels = metadata.get(
            "profile_group_label", metadata.get("group_label")
        )
        prf_labels = prf_labels.long() if prf_labels is not None else None
        qp = self.surrogates["qp"](
            x_profile,
            mc_samples=self.mc_samples,
            group_labels=prf_labels,
            metadata=metadata,
        )

        profile_mean = qp.mean_profile  # (B, P)
        bg_mean = qbg.mean  # (B,)

        scale = self._get_scale(metadata, device)  # (B,)

        d_per_obs = metadata["d"].to(device).float()
        tau = self._wilson_tau(d_per_obs)

        alpha, beta, pi = self._conjugate_em(
            counts, profile_mean, bg_mean, scale, tau, mask
        )

        qi = Gamma(alpha.clamp(min=1e-6), beta.clamp(min=1e-12))

        # Reconstruction rate
        if self.sample_I:
            zI = qi.rsample([self.mc_samples])  # (S, B)
            zI = zI.clamp(min=1e-10)
        else:
            zI = (alpha / beta).unsqueeze(0).expand(self.mc_samples, B)

        zp = _sample_profile(qp, self.mc_samples)
        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        zI_scaled = (
            (scale.unsqueeze(0) * zI).unsqueeze(-1).permute(1, 0, 2)
        )  # (B, S, 1)
        rate = zI_scaled * zp + zbg

        if "is_coset" in metadata:
            coset = metadata["is_coset"].bool().view(-1, 1, 1)
            rate = torch.where(coset, zbg, rate)

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
        if "asu_id" in metadata:
            out["asu_id"] = metadata["asu_id"].long().to(device)
        if "group_label" in metadata:
            _add_group_outputs(out, metadata, self.loss)

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
            "alpha": alpha,
            "beta": beta,
            "tau": tau,
            "pi_mean": pi.mean().detach(),
            "scale_mean": scale.mean().detach(),
            "scale_std": scale.std().detach(),
            "scale_min": scale.min().detach(),
            "scale_max": scale.max().detach(),
            "bg_mean": bg_mean.mean().detach(),
            "profile_max_mean": profile_mean.max(dim=-1)
            .values.mean()
            .detach(),
        }

    def _step(self, batch, step: Literal["train", "val"]):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]

        group_labels = metadata["group_label"].long()

        # -ELBO =  Poisson NLL + KL_prf + KL_i + KL_bg.
        # KL_i is the closed-form Gamma-Gamma KL between our conjugate
        # posterior and the Wilson prior
        loss_dict = self.loss(
            rate=forward_out["rates"],
            counts=forward_out["counts"],
            qp=outputs["qp"],
            qi=outputs["qi"],
            qbg=outputs["qbg"],
            mask=forward_out["mask"],
            group_labels=group_labels,
            metadata=metadata,
        )

        total_loss = loss_dict["loss"]

        _log_loss(
            self,
            kl=loss_dict["kl_mean"],
            nll=loss_dict["neg_ll_mean"],
            total_loss=total_loss,
            step=step,
            kl_components={
                "kl_prf": loss_dict["kl_prf_mean"],
                "kl_i": loss_dict["kl_i_mean"],
                "kl_bg": loss_dict["kl_bg_mean"],
            },
        )

        penalty, penalty_components = self._profile_basis_penalty()
        for name, value in penalty_components.items():
            self.log(f"{step} {name}", value, on_step=False, on_epoch=True)
        total_loss = total_loss + penalty

        with torch.no_grad():
            alpha = outputs["alpha"]
            beta = outputs["beta"]
            I_mean = alpha / beta
            I_var = alpha / beta.pow(2)
            self.log(
                f"{step} qi_mean", I_mean.mean(), on_step=False, on_epoch=True
            )
            self.log(
                f"{step} qi_var", I_var.mean(), on_step=False, on_epoch=True
            )
            self.log(
                f"{step} alpha_mean",
                alpha.mean(),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{step} beta_mean", beta.mean(), on_step=False, on_epoch=True
            )
            self.log(
                f"{step} pi_mean",
                outputs["pi_mean"],
                on_step=False,
                on_epoch=True,
            )
            for k in (
                "scale_mean",
                "scale_std",
                "scale_min",
                "scale_max",
                "bg_mean",
                "profile_max_mean",
            ):
                self.log(
                    f"{step} {k}", outputs[k], on_step=False, on_epoch=True
                )

        return {
            "loss": total_loss,
            "forward_out": forward_out,
            "loss_components": {
                "loss": total_loss.detach(),
                "nll": loss_dict["neg_ll_mean"].detach(),
                "kl": loss_dict["kl_mean"].detach(),
                "kl_prf": loss_dict["kl_prf_mean"].detach(),
                "kl_i": loss_dict["kl_i_mean"].detach(),
                "kl_bg": loss_dict["kl_bg_mean"].detach(),
            },
        }

    # Map predict_keys -> exact_intensity_posterior() output fields. Requesting
    # any of these in predict_keys triggers the calibrated quadrature export.
    _EXACT_KEY_MAP = {
        "qi_exact_mean": "mean",
        "qi_exact_var": "var",
        "qi_exact_std": "std",
        "qi_exact_alpha": "alpha",
        "qi_exact_beta": "beta",
    }

    def predict_step(self, batch, _batch_idx):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        result = {
            k: v
            for k, v in outputs["forward_out"].items()
            if k in self.predict_keys
        }
        wanted = [k for k in self.predict_keys if k in self._EXACT_KEY_MAP]
        if wanted:
            post = self.exact_intensity_posterior(
                counts,
                shoebox,
                mask,
                metadata,
                n_nuisance=self.exact_posterior_n_nuisance,
                n_grid=self.exact_posterior_n_grid,
            )
            for k in wanted:
                result[k] = post[self._EXACT_KEY_MAP[k]]
        return result
