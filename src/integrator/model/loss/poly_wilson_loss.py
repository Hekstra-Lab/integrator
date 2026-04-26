"""Polychromatic Wilson loss with per-wavelength-bin G_k.

Same structure as WilsonLoss but the global scale G is replaced by a
per-wavelength-bin vector G_k. Each refl is bucketed by metadata["wavelength"]
into one of n_lambda_bins bins, and tau_h is computed as

    tau_h = (1 / G_{k(h)}) * exp(2 B s_h^2)

where s_h^2 = 1 / (4 d_h^2) (Bragg-collapsed; wavelength cancels). The
single global B and the per-bin G_k all live in the same Normal-Normal
hyperprior structure, so the loss reduces to WilsonLoss when n_lambda_bins=1.
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import (
    Distribution,
    Gamma,
    Normal,
    Poisson,
    kl_divergence,
)

from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)
from integrator.model.loss.kl_helpers import (
    _kl,
    compute_bg_kl,
    compute_profile_kl,
)
from integrator.model.loss.per_bin_loss import _load_buffer
from integrator.model.loss.wilson_loss import WilsonLoss


class PolyWilsonLoss(WilsonLoss):
    """ELBO loss for polychromatic Laue: per-wavelength-bin G_k, global B.

    Args (in addition to WilsonLoss):
        wavelength_bin_edges: list/path of n_bins+1 floats. Must be sorted.
            Defines half-open bins [edge_i, edge_{i+1}); the leftmost edge is
            inclusive and the rightmost is exclusive (out-of-range λ are
            clamped to the nearest valid bin and a warning is printed once).
        init_log_K_per_bin: optional list of n_bins floats for q(log G_k) loc
            initialization. If None, all bins start at init_log_K (parent arg).

    Notes:
        - The parent's tau-based init path expects scalar (G, B), so we set
          init_from_tau=False unconditionally; data-driven per-bin init is
          deferred — for now, hyperprior + SGD do the work.
        - The hyperprior on G is the same Normal(hp_log_K_loc, hp_log_K_scale)
          as in the monochromatic case, broadcast across bins. So G_k bins
          are drawn iid from the same hyperprior — no hierarchical pooling
          across bins (could be added later if needed).
    """

    def __init__(
        self,
        *,
        wavelength_bin_edges: list[float] | str,
        init_log_K_per_bin: Sequence[float] | None = None,
        # Disable parent's tau-based init — it's scalar-only.
        init_from_tau: bool = False,
        tau_per_group=None,
        s_squared_per_group=None,
        **kwargs,
    ):
        import inspect

        parent_params = set(
            inspect.signature(WilsonLoss.__init__).parameters
        ) - {"self"}
        parent_kwargs = {k: v for k, v in kwargs.items() if k in parent_params}
        super().__init__(
            init_from_tau=init_from_tau,
            tau_per_group=tau_per_group,
            s_squared_per_group=s_squared_per_group,
            **parent_kwargs,
        )

        edges = _load_buffer(wavelength_bin_edges).to(torch.float32)
        if edges.ndim != 1 or edges.numel() < 2:
            raise ValueError(
                "wavelength_bin_edges must be 1-D with >= 2 entries, got "
                f"shape={tuple(edges.shape)}"
            )
        if not torch.all(edges[1:] > edges[:-1]):
            raise ValueError("wavelength_bin_edges must be strictly increasing")

        n_bins = int(edges.numel() - 1)
        self.wavelength_bin_edges: Tensor
        self.register_buffer("wavelength_bin_edges", edges)
        self.n_lambda_bins = n_bins

        # Replace parent's scalar K params with per-bin vectors.
        if init_log_K_per_bin is not None:
            init_loc = torch.tensor(init_log_K_per_bin, dtype=torch.float32)
            if init_loc.shape != (n_bins,):
                raise ValueError(
                    f"init_log_K_per_bin must have shape ({n_bins},), got "
                    f"{tuple(init_loc.shape)}"
                )
        else:
            init_loc = torch.full((n_bins,), float(self.q_log_K_loc.detach()))
        del self.q_log_K_loc
        del self.q_log_K_log_scale
        self.q_log_K_loc = nn.Parameter(init_loc)
        self.q_log_K_log_scale = nn.Parameter(torch.full((n_bins,), -2.0))

        self._oob_warned = False  # one-shot warn flag for out-of-range λ

    def q_log_K(self) -> Normal:
        return Normal(self.q_log_K_loc, F.softplus(self.q_log_K_log_scale))

    def p_log_K(self) -> Normal:
        loc = self.hp_log_K_loc.expand(self.n_lambda_bins)
        scale = self.hp_log_K_scale.expand(self.n_lambda_bins)
        return Normal(loc, scale)

    def kl_hyperparams(self) -> Tensor:
        """Sum across bins for G; scalar for B."""
        kl_K = kl_divergence(self.q_log_K(), self.p_log_K()).sum()
        kl_B = kl_divergence(self.q_log_B(), self.p_log_B())
        return kl_K + kl_B

    def _wavelength_to_bin(self, wavelength: Tensor) -> Tensor:
        """Bucketize per-refl wavelengths into [0, n_lambda_bins).

        Bin convention: `[edges[i], edges[i+1])` for i < n_bins - 1, and
        `[edges[-2], edges[-1]]` for the final bin (rightmost edge inclusive).
        Values strictly outside `[edges[0], edges[-1]]` are clamped to the
        nearest bin and a one-shot warning is printed.
        """
        edges = self.wavelength_bin_edges
        # right=True gives left-closed right-open semantics:
        #   bucketize(x) returns i s.t. edges[i-1] <= x < edges[i]
        # so bin = idx - 1, and bin == n_lambda_bins means x >= edges[-1].
        idx = torch.bucketize(wavelength, edges, right=True) - 1
        # OOB is strictly outside [edges[0], edges[-1]] — having λ == edges[-1]
        # is fine (we want it in the last bin, not flagged).
        if not self._oob_warned:
            oob = (wavelength < edges[0]) | (wavelength > edges[-1])
            n_oob = int(oob.sum())
            if n_oob > 0:
                print(
                    f"[PolyWilsonLoss] {n_oob} reflection(s) have wavelength "
                    f"outside [{float(edges[0]):.4f}, {float(edges[-1]):.4f}] Å "
                    "and are being clamped to the nearest bin. "
                    "Widen wavelength_bin_edges to silence this warning."
                )
                self._oob_warned = True
        return idx.clamp(0, self.n_lambda_bins - 1)

    def posterior_means(self) -> dict[str, float]:
        """Per-bin G summary plus scalar B (and alpha if learnable)."""
        s_K = F.softplus(self.q_log_K_log_scale)
        s_B = F.softplus(self.q_log_B_log_scale)
        K_means = (self.q_log_K_loc + 0.5 * s_K**2).exp().detach()
        out = {
            "K_mean": K_means.mean().item(),
            "K_min": K_means.min().item(),
            "K_max": K_means.max().item(),
            "B_mean": (self.q_log_B_loc + 0.5 * s_B**2).exp().item(),
            "B_std": self._lognormal_std(self.q_log_B_loc, s_B),
        }
        if self.learn_concentration:
            alphas = F.softplus(self.log_alpha_per_group).detach()
            out["alpha_mean"] = alphas.mean().item()
            out["alpha_min"] = alphas.min().item()
            out["alpha_max"] = alphas.max().item()
        return out

    def forward(
        self,
        rate: Tensor,
        counts: Tensor,
        qp: Distribution | ProfileSurrogateOutput,
        qi: Distribution,
        qbg: Distribution,
        mask: Tensor,
        group_labels: Tensor,
        **kwargs,
    ) -> dict[str, Tensor]:
        device = rate.device
        batch_size = rate.shape[0]
        counts = counts.to(device)
        mask = mask.to(device)
        groups = group_labels.long()

        kl = torch.zeros(batch_size, device=device)
        kl_prf = torch.zeros(batch_size, device=device)
        kl_i = torch.zeros(batch_size, device=device)
        kl_bg = torch.zeros(batch_size, device=device)

        # Profile KL — same as parent
        kl_prf = compute_profile_kl(
            qp,
            groups,
            self.profile_sigma_prior,
            None,
            None,
            self.pprf_weight,
            device,
            metadata=kwargs.get("metadata"),
        )
        kl = kl + kl_prf

        # Wilson intensity KL — per-refl tau using per-bin G_{k(λ)} and global B.
        metadata = kwargs.get("metadata")
        if metadata is None or "d" not in metadata or "wavelength" not in metadata:
            raise ValueError(
                "PolyWilsonLoss requires metadata['d'] and metadata['wavelength']."
            )
        d = metadata["d"].to(device)
        s_sq = 1.0 / (4.0 * d.pow(2))  # (B,)
        wavelength = metadata["wavelength"].to(device)
        lam_bin = self._wavelength_to_bin(wavelength)  # (B,) long

        if self.learn_concentration:
            alpha_i = F.softplus(self.log_alpha_per_group[groups])  # (B,)
        else:
            alpha_i = None

        for _ in range(self.n_wilson_samples):
            log_K_bins = self.q_log_K().rsample()  # (n_lambda_bins,)
            log_B = self.q_log_B().rsample()       # scalar
            K_per_refl = torch.exp(log_K_bins)[lam_bin]  # (B,)
            B = torch.exp(log_B)
            tau = (1.0 / K_per_refl) * torch.exp(2.0 * B * s_sq)
            if alpha_i is not None:
                p_i = Gamma(concentration=alpha_i, rate=alpha_i * tau)
            else:
                p_i = Gamma(
                    concentration=torch.ones_like(tau),
                    rate=tau,
                )
            kl_i = kl_i + _kl(qi, p_i, self.mc_samples, eps=self.eps)
        kl_i = kl_i / self.n_wilson_samples
        kl_i = kl_i * self.pi_weight
        kl = kl + kl_i

        # Background KL — same as parent
        kl_bg = compute_bg_kl(
            qbg,
            groups,
            self.bg_rate_per_group,
            self.bg_concentration_per_group,
            self.bg_concentration,
            self.pbg_weight,
            self.mc_samples,
            self.eps,
        )
        kl = kl + kl_bg

        # Hyperprior KL: KL on B + sum-over-bins KL on G_k.
        kl_hyper = self.kl_hyperparams() / self.dataset_size

        # Poisson NLL
        ll = Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))
        ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
        neg_ll = (-ll_mean).sum(1)

        loss = (neg_ll + kl).mean() + kl_hyper

        return {
            "loss": loss,
            "neg_ll_mean": neg_ll.mean(),
            "kl_mean": kl.mean(),
            "kl_prf_mean": kl_prf.mean(),
            "kl_i_mean": kl_i.mean(),
            "kl_bg_mean": kl_bg.mean(),
            "kl_hyper": kl_hyper,
        }
