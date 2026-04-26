"""
Compare: buggy normalization (non-anscombe) vs correct normalization (anscombe).
Runs both with same random seed to see if the NaN is caused by the normalization mismatch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Gamma, Poisson

DATA_DIR = (
    "/Users/luis/master/notebooks/integrator_notes/code/simulating_shoeboxes"
)
BATCH_SIZE = 128
MC_SAMPLES = 20
H, W = 21, 21
N_PIXELS = H * W
LR = 0.001
GRAD_CLIP = 1.0
N_STEPS = 1000

# ── Load data ────────────────────────────────────────────────────────────
counts_all = torch.load(f"{DATA_DIR}/counts.pt", weights_only=False).squeeze(
    -1
)
masks_all = torch.load(f"{DATA_DIR}/masks.pt", weights_only=False).squeeze(-1)
stats = torch.load(f"{DATA_DIR}/stats_anscombe.pt", weights_only=False)
conc_raw = torch.load(f"{DATA_DIR}/concentration.pt", weights_only=False)

conc_prior = conc_raw.clone()
conc_prior[conc_prior > 2] *= 40
conc_prior /= conc_prior.sum()
alpha_prior = conc_prior.reshape(-1)

# Two standardizations
std_buggy = ((counts_all * masks_all) - stats[0]) / stats[1].sqrt()
ans = 2 * torch.sqrt(counts_all + 3.0 / 8.0)
std_correct = (ans - stats[1]) / stats[1].sqrt()

print(
    f"Buggy standardization:  range=[{std_buggy.min():.1f}, {std_buggy.max():.1f}], std={std_buggy.std():.2f}"
)
print(
    f"Correct standardization: range=[{std_correct.min():.1f}, {std_correct.max():.1f}], std={std_correct.std():.2f}"
)


class SimpleEncoder(nn.Module):
    def __init__(self, out_dim=64):
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

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.squeeze(-1).squeeze(-1)
        return F.relu(self.fc(x))


class GammaHead(nn.Module):
    def __init__(self, in_features=64, eps=1e-6):
        super().__init__()
        self.linear_k = nn.Linear(in_features, 1)
        self.linear_r = nn.Linear(in_features, 1)
        self.eps = eps

    def forward(self, x):
        k = F.softplus(self.linear_k(x)) + self.eps
        r = F.softplus(self.linear_r(x)) + self.eps
        return Gamma(k.flatten(), r.flatten())


class DirichletHead(nn.Module):
    def __init__(self, in_features=64, n_pixels=441, eps=1e-6):
        super().__init__()
        self.alpha_layer = nn.Linear(in_features, n_pixels)
        self.eps = eps

    def forward(self, x):
        return Dirichlet(F.softplus(self.alpha_layer(x)) + self.eps)


class MiniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_profile = SimpleEncoder(64)
        self.enc_intensity = SimpleEncoder(64)
        self.qp_head = DirichletHead()
        self.qi_head = GammaHead()
        self.qbg_head = GammaHead()

    def forward(self, shoebox):
        b = shoebox.shape[0]
        x = shoebox.reshape(b, 1, H, W)
        x_profile = self.enc_profile(x)
        x_intensity = self.enc_intensity(x)
        return (
            self.qp_head(x_profile),
            self.qi_head(x_intensity),
            self.qbg_head(x_intensity),
        )


def compute_loss(
    qp, qi, qbg, counts, mask, alpha_prior, mc_samples=20, eps=1e-6
):
    zI = qi.rsample([mc_samples]).unsqueeze(-1).permute(1, 0, 2)
    zp = qp.rsample([mc_samples]).permute(1, 0, 2)
    zbg = qbg.rsample([mc_samples]).unsqueeze(-1).permute(1, 0, 2)
    rate = zI * zp + zbg

    ll = Poisson(rate + eps).log_prob(counts.unsqueeze(1))
    ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
    neg_ll = (-ll_mean).sum(1)

    p_dir = Dirichlet(alpha_prior)
    kl_prf = torch.distributions.kl.kl_divergence(qp, p_dir)
    p_i = Gamma(torch.tensor(1.0), torch.tensor(0.001))
    kl_i = torch.distributions.kl.kl_divergence(qi, p_i)
    p_bg = Gamma(torch.tensor(1.0), torch.tensor(1.0))
    kl_bg = torch.distributions.kl.kl_divergence(qbg, p_bg)

    kl = kl_prf * 0.005 + kl_i * 0.5 + kl_bg * 0.5
    loss = (neg_ll + kl).mean()

    return (
        loss,
        neg_ll.mean().item(),
        kl_prf.mean().item(),
        rate.max().item(),
        rate.min().item(),
    )


def run_experiment(name, standardized_data, seed=42):
    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT: {name}")
    print(f"{'=' * 70}")

    torch.manual_seed(seed)
    model = MiniModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    n_samples = len(counts_all)
    nan_step = None

    for step in range(N_STEPS):
        # Cycle through data
        start = (step * BATCH_SIZE) % n_samples
        end = start + BATCH_SIZE
        if end > n_samples:
            start = 0
            end = BATCH_SIZE

        batch_std = standardized_data[start:end]
        batch_counts = counts_all[start:end]
        batch_masks = masks_all[start:end]

        try:
            qp, qi, qbg = model(batch_std)
        except Exception as e:
            print(f"  Step {step}: FORWARD EXCEPTION: {e}")
            nan_step = step
            break

        k_min = qi.concentration.min().item()
        k_max = qi.concentration.max().item()
        r_min = qi.rate.min().item()
        alpha_min = qp.concentration.min().item()
        alpha_max = qp.concentration.max().item()

        if qi.concentration.isnan().any():
            print(f"  Step {step}: qi.concentration has NaN!")
            nan_step = step
            break

        try:
            loss, nll, kl_prf, rate_max, rate_min = compute_loss(
                qp, qi, qbg, batch_counts, batch_masks, alpha_prior, MC_SAMPLES
            )
        except Exception as e:
            print(f"  Step {step}: LOSS EXCEPTION: {e}")
            nan_step = step
            break

        if loss.isnan() or loss.isinf():
            print(
                f"  Step {step}: loss={loss.item()}, nll={nll}, kl_prf={kl_prf}"
            )
            print(f"    k=[{k_min:.6g}, {k_max:.6g}], r_min={r_min:.6g}")
            print(f"    alpha=[{alpha_min:.6g}, {alpha_max:.6g}]")
            print(f"    rate=[{rate_min:.3g}, {rate_max:.3g}]")
            nan_step = step
            break

        if step % 100 == 0:
            print(
                f"  Step {step:4d}: loss={loss.item():>10.1f}  nll={nll:>9.1f}  "
                f"kl_prf={kl_prf:>8.0f}  k=[{k_min:.2f},{k_max:.2f}]  "
                f"r_min={r_min:.5f}  α=[{alpha_min:.1e},{alpha_max:.1f}]  "
                f"rate_max={rate_max:.1f}"
            )

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRAD_CLIP
        )

        if torch.isnan(grad_norm):
            print(f"  Step {step}: gradient norm is NaN!")
            print(f"    loss={loss.item()}, nll={nll}")
            for pname, p in model.named_parameters():
                if p.grad is not None and p.grad.isnan().any():
                    print(f"    NaN grad in: {pname}")
            nan_step = step
            break

        optimizer.step()

    if nan_step is None:
        print(f"  Completed {N_STEPS} steps without NaN.")
        # Print final state
        for pname, p in model.named_parameters():
            if "linear_k" in pname or "linear_r" in pname:
                print(f"    {pname}: max_abs={p.abs().max().item():.4f}")
    else:
        print(f"  *** NaN at step {nan_step} ***")

    return nan_step


# ── Run both experiments ─────────────────────────────────────────────────
result_buggy = run_experiment(
    "BUGGY (raw counts with anscombe stats)", std_buggy
)
result_correct = run_experiment(
    "CORRECT (anscombe-transformed counts)", std_correct
)

print(f"\n{'=' * 70}")
print("COMPARISON")
print(f"{'=' * 70}")
print(
    f"  Buggy normalization:   {'NaN at step ' + str(result_buggy) if result_buggy else 'No NaN in ' + str(N_STEPS) + ' steps'}"
)
print(
    f"  Correct normalization: {'NaN at step ' + str(result_correct) if result_correct else 'No NaN in ' + str(N_STEPS) + ' steps'}"
)
