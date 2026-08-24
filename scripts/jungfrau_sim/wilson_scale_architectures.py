"""Which architecture recovers per-image Wilson G (and B) best, trained the way the model is?

Companion to `wilson_per_image_prior.py`, which established the closed form. That script
solved G_i by coordinate descent (the exact MLE). Here we instead TRAIN each candidate
parameterization by gradient descent on the same objective, because that is what the
integrator actually does -- G and B are optimized through the ELBO by Adam, not by an EM
solve. The question this answers is architectural: given the realistic ~100 reflections
per image, which way of PARAMETERIZING G_i lets gradient descent recover the truth?

The objective is exactly the G/B-dependent part of the Wilson KL. For a per-reflection
posterior-mean intensity E_j on image i at resolution s_j^2, and prior mean
mu_ij = G_i exp(-2 B s_j^2), the tau-part of KL(q(I) || Exp(1/mu)) is

    L = mean_j [ log mu_ij + E_j / mu_ij ].

G and B enter the whole ELBO only here (verified in monochromatic_wilson_loss.py), so
holding the intensities E fixed and descending L in the architecture's parameters is a
faithful stand-in for the coordinate step the real model takes.

Architectures compared:
    global      one shared log G (what the loss does today)
    free        a free per-image log G (nn.Embedding) -- can it reach the MLE by SGD?
    amortized   log G = MLP(image summary stats) -- the restricted function class
    hier        free per-image log G + a learned population prior (partial pooling)
    solved      closed-form MLE from wilson_per_image_prior (the ceiling, not trained)

Run:  uv run python scripts/jungfrau_sim/wilson_scale_architectures.py [--per-image-B]
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from wilson_per_image_prior import (
    _corr,
    fit_solved,
    fit_solved_per_image_B,
    intensities,
    simulate,
)

torch.set_default_dtype(torch.float64)


def image_features(sim: dict, e: torch.Tensor) -> torch.Tensor:
    """The honest per-image summary stats an amortizing network would see, standardized.

    log mean intensity, mean s^2, log reflection count -- deliberately NOT the per-image G,
    which is what the network must infer from them.
    """
    idx, n = sim["image_idx"], sim["n_images"]
    cnt = torch.bincount(idx, minlength=n).clamp_min(1).double()
    mean_e = torch.bincount(idx, weights=e, minlength=n) / cnt
    mean_s = torch.bincount(idx, weights=sim["s_sq"], minlength=n) / cnt
    f = torch.stack([torch.log(mean_e.clamp_min(1e-6)), mean_s, torch.log(cnt)], dim=1)
    return (f - f.mean(0)) / f.std(0).clamp_min(1e-8)


class ScaleHead(nn.Module):
    """Maps image index (and features) to per-image (log G, B). B optionally per-image.

    `kind`: global | free | amortized | hier. `per_image_B` switches B from one shared
    softplus scalar to a per-image head of the same architecture as G.
    """

    def __init__(self, kind: str, n_images: int, feats: torch.Tensor, per_image_B: bool):
        super().__init__()
        self.kind = kind
        self.per_image_B = per_image_B
        self.register_buffer("feats", feats)
        d = feats.shape[1]

        if kind == "global":
            self.log_g = nn.Parameter(torch.zeros(1))
        elif kind in ("free", "hier"):
            self.log_g = nn.Parameter(torch.zeros(n_images))
            if kind == "hier":  # learned population prior over log G
                self.prior_mu = nn.Parameter(torch.zeros(1))
                self.prior_logsig = nn.Parameter(torch.zeros(1))
        elif kind == "amortized":
            self.mlp = nn.Sequential(
                nn.Linear(d, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1)
            )
        else:
            raise ValueError(kind)

        n_b = n_images if per_image_B else 1
        self.raw_b = nn.Parameter(torch.zeros(n_b))
        if per_image_B and kind == "amortized":
            self.mlp_b = nn.Sequential(
                nn.Linear(d, 32), nn.SiLU(), nn.Linear(32, 1)
            )

    def log_G(self) -> torch.Tensor:
        if self.kind == "global":
            return self.log_g.expand(self.feats.shape[0])
        if self.kind == "amortized":
            return self.mlp(self.feats).squeeze(1)
        return self.log_g  # free / hier

    def B(self) -> torch.Tensor:
        """Per-image B, shape (n_images,). A shared B is broadcast to every image."""
        n = self.feats.shape[0]
        if self.per_image_B and self.kind == "amortized":
            return torch.nn.functional.softplus(self.mlp_b(self.feats).squeeze(1))
        b = torch.nn.functional.softplus(self.raw_b)
        return b.expand(n) if b.numel() == 1 else b

    def prior_penalty(self) -> torch.Tensor:
        """Partial-pooling penalty: -log N(log G_i; mu, sigma), summed. Zero unless hier."""
        if self.kind != "hier":
            return torch.zeros(())
        sig = torch.exp(self.prior_logsig).clamp_min(1e-3)
        z = (self.log_g - self.prior_mu) / sig
        return (0.5 * z**2 + self.prior_logsig).sum()


def train(
    kind: str,
    sim: dict,
    e: torch.Tensor,
    per_image_B: bool,
    steps: int = 4000,
    lr: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adam on the Wilson-KL objective, exactly the term through which G,B reach the ELBO."""
    feats = image_features(sim, e)
    head = ScaleHead(kind, sim["n_images"], feats, per_image_B)
    idx, s_sq = sim["image_idx"], sim["s_sq"]
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        log_mu = head.log_G()[idx] - 2.0 * head.B()[idx] * s_sq
        # L = mean[ log mu + E/mu ];  E/mu = E * exp(-log mu)
        loss = (log_mu + e * torch.exp(-log_mu)).mean() + head.prior_penalty() / e.numel()
        loss.backward()
        opt.step()

    with torch.no_grad():
        g = torch.exp(head.log_G())
        b = head.B()
        if b.numel() == 1:
            b = b.expand(sim["n_images"])
    return g.detach(), b.detach()


def score(g_hat, b_hat, sim) -> dict:
    lg_hat = torch.log(g_hat.clamp_min(1e-12))
    return {
        "corr_logG": _corr(lg_hat, torch.log(sim["g_true"])),
        "rmse_logG": float((lg_hat - torch.log(sim["g_true"])).pow(2).mean().sqrt()),
        "rmse_B": float((b_hat - sim["b_true"]).pow(2).mean().sqrt()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-images", type=int, default=300)
    ap.add_argument("--n-refl", type=int, default=100)
    ap.add_argument("--sigma-logg", type=float, default=0.6)
    ap.add_argument("--b-global", type=float, default=20.0)
    ap.add_argument("--sigma-b", type=float, default=5.0)
    ap.add_argument("--per-image-B", action="store_true")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sigma_b = args.sigma_b if args.per_image_B else 0.0
    sim = simulate(
        args.n_images, args.n_refl, args.sigma_logg, args.b_global, sigma_b,
        bg=1.0, seed=args.seed,
    )
    e = intensities(sim, oracle=False)

    print("Wilson per-image scale: which ARCHITECTURE recovers G (trained by SGD on the KL)")
    print(f"  {args.n_images} images x {args.n_refl} refl, sigma_logG={args.sigma_logg}, "
          f"B={args.b_global}" + (f" +/- {args.sigma_b} (per-image)" if args.per_image_B
                                   else " (global)"))
    print(f"  intensities from counts (Poisson noise), {args.steps} Adam steps\n")

    print(f"    {'architecture':<12} {'corr(logG)':>11} {'rmse(logG)':>11} {'rmse(B)':>9}"
          f"   {'params':>8}")
    print("    " + "-" * 58)

    # Closed-form ceiling: the MLE for whichever case we are in (global vs per-image B).
    if args.per_image_B:
        gs, bs = fit_solved_per_image_B(sim, e)
        ref = "solved+Bi"
    else:
        gs, bs = fit_solved(sim, e)
        ref = "solved (MLE)"
    m = score(gs, bs, sim)
    print(f"    {ref:<12} {m['corr_logG']:>11.3f} {m['rmse_logG']:>11.3f}"
          f" {m['rmse_B']:>9.2f}   {'--':>8}")

    for kind in ("global", "free", "amortized", "hier"):
        g_hat, b_hat = train(kind, sim, e, args.per_image_B, steps=args.steps)
        m = score(g_hat, b_hat, sim)
        head = ScaleHead(kind, sim["n_images"], image_features(sim, e), args.per_image_B)
        npar = sum(p.numel() for p in head.parameters())
        print(f"    {kind:<12} {m['corr_logG']:>11.3f} {m['rmse_logG']:>11.3f}"
              f" {m['rmse_B']:>9.2f}   {npar:>8}")

    print("\n  Read: does a trained head reach `solved` (the MLE)? `free`/`hier` have one")
    print("  parameter per image; `amortized` has a fixed-size MLP that must generalize")
    print("  across images from summary stats alone. `global` is today's single-G loss.")


if __name__ == "__main__":
    main()
