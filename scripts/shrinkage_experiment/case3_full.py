"""Case 3 (full model): Encoder with unknown intensity, background, and profile.

All three quantities are treated as latent/learned. The encoder predicts
Gamma surrogates for intensity and background, plus a profile point estimate
(softmax). This tests whether bias is amplified when the model must
simultaneously learn all components.
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


class FullEncoder(nn.Module):
    """Encoder predicting Gamma(I), Gamma(B), and profile (softmax)."""

    def __init__(
        self, out_dim: int = 64, n_pixels: int = 441, eps: float = 1e-6
    ):
        super().__init__()
        self.encoder = SimpleEncoder(out_dim)
        # Intensity head
        self.linear_k_I = nn.Linear(out_dim, 1)
        self.linear_r_I = nn.Linear(out_dim, 1)
        # Background head
        self.linear_k_B = nn.Linear(out_dim, 1)
        self.linear_r_B = nn.Linear(out_dim, 1)
        # Profile head
        self.profile_head = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_pixels),
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> tuple[Gamma, Gamma, torch.Tensor]:
        b = x.shape[0]
        x = x.reshape(b, 1, H, W)
        h = self.encoder(x)

        k_I = F.softplus(self.linear_k_I(h)) + self.eps
        r_I = F.softplus(self.linear_r_I(h)) + self.eps
        q_I = Gamma(k_I.squeeze(-1), r_I.squeeze(-1))

        k_B = F.softplus(self.linear_k_B(h)) + self.eps
        r_B = F.softplus(self.linear_r_B(h)) + self.eps
        q_B = Gamma(k_B.squeeze(-1), r_B.squeeze(-1))

        profile = F.softmax(self.profile_head(h), dim=-1)

        return q_I, q_B, profile


def run_case3_full(
    profiles: torch.Tensor,
    I_true: torch.Tensor,
    B_true: torch.Tensor,
    alpha0: float,
    beta0: float,
    counts: torch.Tensor,
    alpha0_B: float = 2.0,
    beta0_B: float = 0.5,
    mc_samples: int = 100,
    n_epochs: int = 500,
    batch_size: int = 128,
    lr: float = 1e-3,
    eps: float = 1e-6,
) -> dict:
    """Train full encoder and extract posterior means.

    Args:
        profiles: (N, 441) true profiles (for bias computation only)
        I_true: (N,) true intensities
        B_true: (N,) per-reflection background rates
        alpha0, beta0: intensity prior Gamma parameters
        counts: (N, 441) observed counts (same as case2)
        alpha0_B, beta0_B: background prior Gamma parameters
        mc_samples: MC samples for ELBO
        n_epochs: training epochs
        batch_size: mini-batch size
        lr: learning rate
        eps: numerical stability

    Returns:
        dict with keys: mu, bias, alpha, beta, mu_B, bias_B, alpha_B, beta_B,
                         profile, elbo, loss_history
    """
    N = counts.shape[0]

    # Anscombe-transform and standardize for encoder input
    ans = 2.0 * torch.sqrt(counts + 3.0 / 8.0)
    mean_ans = ans.mean()
    std_ans = ans.std()
    standardized = (ans - mean_ans) / std_ans  # (N, 441)

    model = FullEncoder(out_dim=64, eps=eps)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01
    )
    prior_I = Gamma(torch.tensor(alpha0), torch.tensor(beta0))
    prior_B = Gamma(torch.tensor(alpha0_B), torch.tensor(beta0_B))

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

            q_I, q_B, profile = model(batch_std)

            I_samples = q_I.rsample([mc_samples])  # (mc, batch)
            B_samples = q_B.rsample([mc_samples])  # (mc, batch)

            # rate = I * profile + B
            rate = I_samples.unsqueeze(-1) * profile.unsqueeze(
                0
            ) + B_samples.unsqueeze(-1)

            # Poisson log-likelihood
            ll = (
                batch_counts[None, :, :] * torch.log(rate + eps)
                - rate
                - torch.lgamma(batch_counts[None, :, :] + 1)
            ).sum(dim=-1)  # (mc, batch)

            kl_I = torch.distributions.kl.kl_divergence(q_I, prior_I)
            kl_B = torch.distributions.kl.kl_divergence(q_B, prior_B)
            elbo = ll.mean(dim=0) - kl_I - kl_B
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
                f"  WARNING: loss still changing "
                f"({pct_change:.1f}% over last 50 epochs)"
            )
        else:
            print(
                f"  Converged (loss changed {pct_change:.2f}% over last 50 epochs)"
            )
    print(f"  Final loss: {loss_history[-1]:.2f}")

    # Extract final predictions
    model.eval()
    with torch.no_grad():
        all_alpha_I, all_beta_I = [], []
        all_alpha_B, all_beta_B = [], []
        all_profile = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_std = standardized[start:end]
            q_I, q_B, profile = model(batch_std)
            all_alpha_I.append(q_I.concentration)
            all_beta_I.append(q_I.rate)
            all_alpha_B.append(q_B.concentration)
            all_beta_B.append(q_B.rate)
            all_profile.append(profile)

        alpha_I = torch.cat(all_alpha_I)
        beta_I = torch.cat(all_beta_I)
        alpha_B = torch.cat(all_alpha_B)
        beta_B = torch.cat(all_beta_B)
        profile_pred = torch.cat(all_profile)

    # Compute per-reflection ELBO with more MC samples
    with torch.no_grad():
        all_elbo = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            qi = Gamma(alpha_I[start:end], beta_I[start:end])
            qb = Gamma(alpha_B[start:end], beta_B[start:end])
            batch_counts = counts[start:end]
            batch_profile = profile_pred[start:end]

            I_s = qi.rsample([mc_samples * 10])
            B_s = qb.rsample([mc_samples * 10])

            rate_s = I_s.unsqueeze(-1) * batch_profile.unsqueeze(
                0
            ) + B_s.unsqueeze(-1)
            ll_s = (
                batch_counts[None, :, :] * torch.log(rate_s + eps)
                - rate_s
                - torch.lgamma(batch_counts[None, :, :] + 1)
            ).sum(dim=-1)
            kl_I = torch.distributions.kl.kl_divergence(qi, prior_I)
            kl_B = torch.distributions.kl.kl_divergence(qb, prior_B)
            elbo = ll_s.mean(dim=0) - kl_I - kl_B
            all_elbo.append(elbo)

        elbo_final = torch.cat(all_elbo)

    mu_I = alpha_I / beta_I
    mu_B = alpha_B / beta_B
    bias_I = mu_I - I_true
    bias_B = mu_B - B_true

    return {
        "mu": mu_I,
        "bias": bias_I,
        "alpha": alpha_I,
        "beta": beta_I,
        "mu_B": mu_B,
        "bias_B": bias_B,
        "alpha_B": alpha_B,
        "beta_B": beta_B,
        "profile": profile_pred,
        "elbo": elbo_final,
        "loss_history": loss_history,
    }
