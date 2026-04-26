"""
Simulate shoeboxes using Hermite-Gaussian profile basis.

Profiles are generated natively from the basis:
    h ~ N(0, I_d)
    prf = softmax(W_hermite @ h + b)

No PCA, no projection, no reconstruction error. The same W and b
are used by the network at inference time. h_true is exact.

Saved artifacts:
  counts.pt             - (n, 441) shoebox counts
  counts_intensity.pt   - (n, H, W) intensity component
  counts_background.pt  - (n, H, W) background component
  profiles.pt           - (n, 441) ground truth profiles
  reference.pt          - metadata dict (intensity, background, h_latent, etc.)
  masks.pt              - (n, 441) all ones
  profile_basis.pt      - {"W", "b", "d", "orders"} for generation and inference
  stats.pt              - [mean, var] of raw shoeboxes
  stats_anscombe.pt     - [mean, var] of Anscombe-transformed shoeboxes
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

torch.manual_seed(42)

# ──────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────
H = 21
W = 21
K = H * W  # 441
n = 200_000

save_dir = Path(
    "/Users/luis/master/notebooks/integrator_notes/code/simulating_shoeboxes/"
)
save_dir.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────
#  Hermite-Gaussian basis construction
# ──────────────────────────────────────────────────────────


def hermite_polynomial(n_order: int, x: torch.Tensor) -> torch.Tensor:
    """Probabilist's Hermite polynomial H_n(x) by recurrence."""
    if n_order == 0:
        return torch.ones_like(x)
    elif n_order == 1:
        return x
    h_prev2 = torch.ones_like(x)
    h_prev1 = x
    for k in range(2, n_order + 1):
        h_curr = x * h_prev1 - (k - 1) * h_prev2
        h_prev2 = h_prev1
        h_prev1 = h_curr
    return h_curr


def build_hermite_basis(
    H: int = 21,
    W: int = 21,
    max_order: int = 4,
    sigma_ref: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """
    2D Hermite-Gaussian basis on an H x W grid.

    Returns:
        W: (K, d) basis matrix (excluding (0,0) mode)
        b: (K,) bias = log of reference Gaussian profile
        orders: list of (nx, ny) tuples
    """
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float64),
        torch.arange(W, dtype=torch.float64),
        indexing="ij",
    )
    x_norm = (xx - cx) / sigma_ref
    y_norm = (yy - cy) / sigma_ref
    gaussian = torch.exp(-0.5 * (x_norm**2 + y_norm**2))

    # Reference profile (centered Gaussian) → bias
    ref = gaussian / gaussian.sum()
    b = torch.log(ref.reshape(-1).clamp(min=1e-10)).float()

    # Build basis functions, skip (0,0) — absorbed into b
    basis_list = []
    orders = []
    for nx in range(max_order + 1):
        for ny in range(max_order + 1 - nx):
            if nx == 0 and ny == 0:
                continue
            psi = (
                hermite_polynomial(nx, x_norm)
                * hermite_polynomial(ny, y_norm)
                * gaussian
            )
            psi = psi / psi.norm()
            basis_list.append(psi.reshape(-1))
            orders.append((nx, ny))

    W_basis = torch.stack(basis_list, dim=1).float()  # (K, d)
    return W_basis, b, orders


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-order",
        type=int,
        default=4,
        help="max Hermite order",
    )
    parser.add_argument(
        "--sigma-ref", type=float, default=3.0, help="reference Gaussian width"
    )
    args = parser.parse_args()

    # ── Step 1: Build Hermite basis ──
    print(
        f"Step 1: Building Hermite-Gaussian basis (max_order={args.max_order}, sigma_ref={args.sigma_ref})..."
    )
    W_basis, b, orders = build_hermite_basis(
        H, W, args.max_order, args.sigma_ref
    )
    d = W_basis.shape[1]

    print(f"  Basis functions: d={d}")
    print(f"  Orders: {orders}")

    # Save basis (used identically for generation AND inference)
    basis_data = {
        "W": W_basis,
        "b": b,
        "d": d,
        "orders": orders,
        "max_order": args.max_order,
        "sigma_ref": args.sigma_ref,
        "sigma_prior": 3.0,
        "basis_type": "hermite",
    }
    torch.save(basis_data, save_dir / "profile_basis.pt")
    print("  Saved profile_basis.pt")

    # ── Step 2: Generate profiles natively from basis ──
    print(f"\nStep 2: Sampling {n} profiles from h ~ N(0, I_{d})...")

    h_true = torch.randn(n, d) * 2.0  # scale up perturbations
    logits = h_true @ W_basis.T + b  # (n, K)
    profiles = F.softmax(logits, dim=-1)  # (n, K)

    entropies = -(profiles * profiles.clamp(min=1e-10).log()).sum(-1)
    print(
        f"  Profile entropy: mean={entropies.mean():.3f}, std={entropies.std():.3f}"
    )
    print(
        f"  Profile peak value: mean={profiles.max(-1).values.mean():.4f}, "
        f"range=[{profiles.max(-1).values.min():.4f}, {profiles.max(-1).values.max():.4f}]"
    )
    print(f"  h_true shape: {h_true.shape}")
    print(f"  h_true stats: mean={h_true.mean():.3f}, std={h_true.std():.3f}")
    print("  (no projection — h_true is exact by construction)")

    # ── Step 3: Generate shoeboxes ──
    print("\nStep 3: Generating shoeboxes...")

    pi = torch.distributions.Exponential(0.001).rsample([n])
    pbg = torch.distributions.Exponential(1.0).rsample([n])

    profiles_2d = profiles.reshape(n, H, W)
    i_samples = torch.poisson(pi.view(n, 1, 1) * profiles_2d)
    bg_samples = torch.poisson(pbg.view(n, 1, 1).expand(n, H, W).clone())
    shoeboxes = i_samples + bg_samples

    print(f"  Mean pixel: {shoeboxes.mean():.2f}, max: {shoeboxes.max():.0f}")
    print(f"  I range: [{pi.min():.2f}, {pi.max():.1f}], mean={pi.mean():.1f}")
    print(
        f"  bg range: [{pbg.min():.3f}, {pbg.max():.2f}], mean={pbg.mean():.2f}"
    )

    # ── Step 4: Save everything ──
    print("\nStep 4: Saving...")

    shoeboxes_flat = shoeboxes.flatten(1)
    n_test = int(n * 0.05)
    is_test = torch.zeros(n, dtype=torch.bool)
    is_test[torch.randperm(n)[:n_test]] = True

    metadata = {
        "shoebox_median": shoeboxes_flat.median(-1).values,
        "shoebox_var": shoeboxes_flat.var(-1),
        "shoebox_mean": shoeboxes_flat.mean(-1),
        "shoebox_min": shoeboxes_flat.min(-1).values,
        "shoebox_max": shoeboxes_flat.max(-1).values,
        "intensity": pi,
        "background": pbg,
        "h_latent": h_true,  # (n, d) — exact ground truth, no projection
        "refl_ids": torch.arange(1, n + 1),
        "is_test": is_test,
    }

    torch.save(shoeboxes_flat, save_dir / "counts.pt")
    torch.save(i_samples, save_dir / "counts_intensity.pt")
    torch.save(bg_samples, save_dir / "counts_background.pt")
    torch.save(profiles.flatten(1), save_dir / "profiles.pt")
    torch.save(metadata, save_dir / "reference.pt")
    torch.save(torch.ones_like(shoeboxes_flat), save_dir / "masks.pt")

    torch.save(
        torch.tensor(
            [shoeboxes_flat.float().mean(), shoeboxes_flat.float().var()]
        ),
        save_dir / "stats.pt",
    )
    ans = 2.0 * (shoeboxes_flat.float() + 0.375).sqrt()
    torch.save(
        torch.tensor([ans.mean(), ans.var()]), save_dir / "stats_anscombe.pt"
    )

    # Backwards compat with old Dirichlet code
    ref_profile = F.softmax(b, dim=-1)  # the reference Gaussian profile
    torch.save(ref_profile * 400, save_dir / "concentration.pt")

    print(f"  All saved to {save_dir}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"  Basis: Hermite-Gaussian, max_order={args.max_order}, sigma_ref={args.sigma_ref}"
    )
    print(f"  d={d} basis functions")
    print(f"  Prior: h ~ N(0, I_{d})")
    print("  Profile: prf = softmax(W @ h + b)")
    print(f"  Profile entropy: {entropies.mean():.3f} ± {entropies.std():.3f}")
    print("")
    print(f"  Network predicts: mu_h ({d}), logvar_h ({d}) → {2 * d} params")
    print(f"  KL: standard diagonal Gaussian, ~{d / 2:.0f} nats when q ≈ p")
    print("  (was ~250 nats with Dirichlet)")
    print("")
    print("  h_true is EXACT — no projection, no reconstruction error.")
    print("  SBC: check ranks of h_latent in reference.pt")

    # ── Plots ──
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(8):
            axes[0, i].imshow(profiles[i].view(H, W).numpy(), cmap="viridis")
            axes[0, i].axis("off")
            axes[0, i].set_title(f"profile {i}", fontsize=7)

            axes[1, i].imshow(shoeboxes[i].numpy(), cmap="viridis")
            axes[1, i].axis("off")
            axes[1, i].set_title(f"I={pi[i]:.0f} bg={pbg[i]:.1f}", fontsize=7)

        axes[0, 0].set_ylabel("profile", fontsize=9)
        axes[1, 0].set_ylabel("shoebox", fontsize=9)
        fig.suptitle(f"Hermite basis (max_order={args.max_order}, d={d})")
        fig.tight_layout()
        plt.savefig(save_dir / "diagnostic_hermite_native.png", dpi=150)
        plt.close(fig)

        # Show some basis functions
        fig, axes = plt.subplots(2, min(7, d), figsize=(14, 4))
        for i in range(min(7, d)):
            axes[0, i].imshow(W_basis[:, i].view(H, W).numpy(), cmap="RdBu_r")
            axes[0, i].axis("off")
            nx, ny = orders[i]
            axes[0, i].set_title(f"({nx},{ny})", fontsize=8)

            h_single = torch.zeros(d)
            h_single[i] = 2.0
            prf_single = F.softmax(h_single @ W_basis.T + b, dim=-1)
            axes[1, i].imshow(prf_single.view(H, W).numpy(), cmap="viridis")
            axes[1, i].axis("off")
            axes[1, i].set_title(f"h[{i}]=2", fontsize=8)

        axes[0, 0].set_ylabel("basis fn", fontsize=9)
        axes[1, 0].set_ylabel("effect", fontsize=9)
        fig.suptitle("Hermite basis functions and their effect on profile")
        fig.tight_layout()
        plt.savefig(save_dir / "hermite_basis_functions.png", dpi=150)
        plt.close(fig)

        print("\n  Plots saved")

    except ImportError:
        pass
