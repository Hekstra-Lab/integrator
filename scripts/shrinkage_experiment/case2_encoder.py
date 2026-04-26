"""Case 2 (encoder): Model B-style CNN encoder with known profile/background.

A lightweight encoder maps standardized shoeboxes → Gamma(α_i, β_i) for
intensity. Profile and background are known and fixed. This tests whether
the amortized inference pipeline reproduces the same bias pattern as
direct per-reflection optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma

H, W = 21, 21


class SimpleEncoder(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, 16)
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=0)
        self.norm2 = nn.GroupNorm(4, 32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.norm3 = nn.GroupNorm(8, 64)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.squeeze(-1).squeeze(-1)
        return F.relu(self.fc(x))


class GammaHead(nn.Module):
    def __init__(self, in_features: int = 64, eps: float = 1e-6):
        super().__init__()
        self.linear_k = nn.Linear(in_features, 1)
        self.linear_r = nn.Linear(in_features, 1)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> Gamma:
        k = F.softplus(self.linear_k(x)) + self.eps
        r = F.softplus(self.linear_r(x)) + self.eps
        return Gamma(k.squeeze(-1), r.squeeze(-1))


class IntensityEncoder(nn.Module):
    """Encoder that maps shoeboxes → Gamma surrogate for intensity only."""

    def __init__(self, out_dim: int = 64, eps: float = 1e-6):
        super().__init__()
        self.encoder = SimpleEncoder(out_dim)
        self.head = GammaHead(out_dim, eps)

    def forward(self, shoebox: torch.Tensor) -> Gamma:
        b = shoebox.shape[0]
        x = shoebox.reshape(b, 1, H, W)
        h = self.encoder(x)
        return self.head(h)


def run_case2_encoder(
    profiles: torch.Tensor,
    I_true: torch.Tensor,
    B_true: torch.Tensor,
    alpha0: float,
    beta0: float,
    counts: torch.Tensor,
    mc_samples: int = 100,
    n_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    eps: float = 1e-6,
) -> dict:
    """Train encoder and extract posterior means.

    Args:
        profiles: (N, 441) normalized profiles
        I_true: (N,) true intensities
        B_true: (N,) per-reflection background rates
        alpha0, beta0: prior parameters
        counts: (N, 441) observed counts (same as case2_direct)
        mc_samples: MC samples for ELBO
        n_epochs: training epochs
        batch_size: mini-batch size
        lr: learning rate
        eps: numerical stability

    Returns:
        dict with keys: mu, bias, alpha, beta
    """
    N = counts.shape[0]

    # Anscombe-transform and standardize for encoder input
    ans = 2.0 * torch.sqrt(counts + 3.0 / 8.0)
    mean_ans = ans.mean()
    std_ans = ans.std()
    standardized = (ans - mean_ans) / std_ans  # (N, 441)

    model = IntensityEncoder(out_dim=64, eps=eps)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01
    )
    prior = Gamma(torch.tensor(alpha0), torch.tensor(beta0))

    # Training loop
    n_batches = (N + batch_size - 1) // batch_size
    loss_history = []

    for epoch in range(n_epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0

        for b_idx in range(n_batches):
            start = b_idx * batch_size
            end = min(start + batch_size, N)
            idx = perm[start:end]

            batch_std = standardized[idx]
            batch_counts = counts[idx]
            batch_profiles = profiles[idx]
            batch_B = B_true[idx]

            qi = model(batch_std)  # Gamma distribution

            I_samples = qi.rsample([mc_samples])  # (mc, batch)

            # rate = I * profile + B
            rate_pred = (
                I_samples.unsqueeze(-1) * batch_profiles.unsqueeze(0)
                + batch_B[None, :, None]
            )

            # Poisson log-likelihood
            ll = (
                batch_counts[None, :, :] * torch.log(rate_pred + eps)
                - rate_pred
                - torch.lgamma(batch_counts[None, :, :] + 1)
            ).sum(dim=-1)  # (mc, batch)

            kl = torch.distributions.kl.kl_divergence(qi, prior)
            elbo = ll.mean(dim=0) - kl
            loss = -elbo.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item() * (end - start)

        avg_loss = epoch_loss / N
        loss_history.append(avg_loss)
        scheduler.step()

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.1f}")

    # Convergence diagnostic
    if len(loss_history) > 50:
        recent = loss_history[-1]
        earlier = loss_history[-50]
        pct_change = abs(recent - earlier) / abs(earlier) * 100
        if pct_change > 1.0:
            print(
                f"  WARNING: loss still changing ({pct_change:.1f}% over last 50 epochs)"
            )
        else:
            print(
                f"  Converged (loss changed {pct_change:.2f}% over last 50 epochs)"
            )
    print(f"  Final loss: {loss_history[-1]:.2f}")

    # Extract final predictions for all reflections
    model.eval()
    with torch.no_grad():
        # Process in batches to avoid memory issues
        all_alpha = []
        all_beta = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_std = standardized[start:end]
            qi = model(batch_std)
            all_alpha.append(qi.concentration)
            all_beta.append(qi.rate)

        alpha_final = torch.cat(all_alpha)
        beta_final = torch.cat(all_beta)

    # Compute per-reflection ELBO with more MC samples
    with torch.no_grad():
        all_elbo = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            qi = Gamma(alpha_final[start:end], beta_final[start:end])
            I_s = qi.rsample([mc_samples * 10])  # (1000, batch)
            rate_s = (
                I_s.unsqueeze(-1) * profiles[start:end].unsqueeze(0)
                + B_true[start:end][None, :, None]
            )
            ll_s = (
                counts[start:end][None, :, :] * torch.log(rate_s + eps)
                - rate_s
                - torch.lgamma(counts[start:end][None, :, :] + 1)
            ).sum(dim=-1)
            kl = torch.distributions.kl.kl_divergence(qi, prior)
            elbo = ll_s.mean(dim=0) - kl
            all_elbo.append(elbo)

        elbo_final = torch.cat(all_elbo)

    mu = alpha_final / beta_final
    bias = mu - I_true

    return {
        "mu": mu,
        "bias": bias,
        "alpha": alpha_final,
        "beta": beta_final,
        "elbo": elbo_final,
        "loss_history": loss_history,
    }
