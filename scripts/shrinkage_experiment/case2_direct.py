"""Case 2 (direct): Per-reflection ELBO optimization with B > 0.

With nonzero background, conjugacy breaks. We optimize Gamma(α_i, β_i)
per reflection via gradient descent on the ELBO.
"""

import torch
import torch.nn.functional as F
from torch.distributions import Gamma


def run_case2_direct(
    profiles: torch.Tensor,
    I_true: torch.Tensor,
    B_true: torch.Tensor,
    alpha0: float,
    beta0: float,
    mc_samples: int = 100,
    n_steps: int = 3000,
    lr: float = 0.01,
    seed: int = 42,
    eps: float = 1e-6,
) -> dict:
    """Optimize per-reflection Gamma surrogates via ELBO.

    Args:
        profiles: (N, 441) normalized profiles
        I_true: (N,) true intensities
        B_true: (N,) per-reflection background rates
        alpha0, beta0: prior Gamma parameters
        mc_samples: MC samples for ELBO estimation
        n_steps: optimization steps
        lr: learning rate
        seed: random seed
        eps: numerical stability constant

    Returns:
        dict with keys: counts, S, mu, bias, alpha, beta, elbo, loss_history
    """
    torch.manual_seed(seed)
    N = profiles.shape[0]

    # Simulate B>0 counts
    rates = I_true.unsqueeze(1) * profiles + B_true.unsqueeze(1)
    counts = torch.poisson(rates)
    S = counts.sum(dim=1)

    # Smart initialization: start near conjugate posterior (pretending B=0)
    alpha_init = (alpha0 + S).clamp(min=1.0)
    beta_init = torch.full_like(alpha_init, beta0 + 1.0)

    def inv_softplus(x):
        return torch.where(x > 20, x, torch.log(torch.expm1(x.clamp(min=eps))))

    raw_alpha = (
        inv_softplus(alpha_init - eps).clone().detach().requires_grad_(True)
    )
    raw_beta = (
        inv_softplus(beta_init - eps).clone().detach().requires_grad_(True)
    )

    optimizer = torch.optim.Adam([raw_alpha, raw_beta], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr * 0.01
    )

    prior = Gamma(torch.tensor(alpha0), torch.tensor(beta0))
    loss_history = []

    for step in range(n_steps):
        alpha = F.softplus(raw_alpha) + eps
        beta = F.softplus(raw_beta) + eps

        q = Gamma(alpha, beta)
        I_samples = q.rsample([mc_samples])  # (S, N)

        # rate = I * profile + B  → (mc, N, 441)
        rate_pred = (
            I_samples.unsqueeze(-1) * profiles.unsqueeze(0)
            + B_true[None, :, None]
        )

        # Poisson log-likelihood: sum over pixels, mean over MC
        ll = (
            counts[None, :, :] * torch.log(rate_pred + eps)
            - rate_pred
            - torch.lgamma(counts[None, :, :] + 1)
        ).sum(dim=-1)  # (mc, N)

        kl = torch.distributions.kl.kl_divergence(q, prior)  # (N,)
        elbo = ll.mean(dim=0) - kl  # (N,)
        loss = -elbo.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw_alpha, raw_beta], max_norm=10.0)
        optimizer.step()
        scheduler.step()

        if step % 500 == 0:
            loss_history.append(loss.item())
            with torch.no_grad():
                a = F.softplus(raw_alpha) + eps
                b = F.softplus(raw_beta) + eps
                print(
                    f"  Step {step:4d}: loss={loss.item():.1f}, "
                    f"alpha=[{a.min():.1f},{a.max():.1f}], "
                    f"q_mean=[{(a / b).min():.1f},{(a / b).max():.1f}]"
                )

    # Final parameters with more MC samples for accurate ELBO
    with torch.no_grad():
        alpha_final = F.softplus(raw_alpha) + eps
        beta_final = F.softplus(raw_beta) + eps
        q_final = Gamma(alpha_final, beta_final)

        I_s = q_final.rsample([mc_samples * 10])  # (10S, N)
        rate_s = (
            I_s.unsqueeze(-1) * profiles.unsqueeze(0) + B_true[None, :, None]
        )
        ll_s = (
            counts[None, :, :] * torch.log(rate_s + eps)
            - rate_s
            - torch.lgamma(counts[None, :, :] + 1)
        ).sum(dim=-1)
        kl_final = torch.distributions.kl.kl_divergence(q_final, prior)
        elbo_final = ll_s.mean(dim=0) - kl_final

    mu = alpha_final / beta_final
    bias = mu - I_true

    return {
        "counts": counts,
        "S": S,
        "mu": mu,
        "bias": bias,
        "alpha": alpha_final,
        "beta": beta_final,
        "elbo": elbo_final,
        "loss_history": loss_history,
    }
