"""
Minimal reproduction: trace NaN through actual training steps.
Runs locally on CPU with the simulated data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Exponential, Gamma, Poisson

torch.manual_seed(42)

DATA_DIR = (
    "/Users/luis/master/notebooks/integrator_notes/code/simulating_shoeboxes"
)
BATCH_SIZE = 64
MC_SAMPLES = 10
H, W = 21, 21
N_PIXELS = H * W
LR = 0.001
GRAD_CLIP = 1.0
N_STEPS = 300

# ── Load data ────────────────────────────────────────────────────────────
counts_all = torch.load(f"{DATA_DIR}/counts.pt", weights_only=False).squeeze(
    -1
)
masks_all = torch.load(f"{DATA_DIR}/masks.pt", weights_only=False).squeeze(-1)
stats = torch.load(f"{DATA_DIR}/stats_anscombe.pt", weights_only=False)
conc_raw = torch.load(f"{DATA_DIR}/concentration.pt", weights_only=False)

print(f"stats type: {type(stats)}, values: {stats}")
print(f"stats[0] (mean): {stats[0].item():.4f}")
print(f"stats[1] (var):  {stats[1].item():.4f}")
print(f"sqrt(var):       {stats[1].sqrt().item():.4f}")
print(
    f"counts: shape={counts_all.shape}, min={counts_all.min():.1f}, max={counts_all.max():.1f}"
)

# Standardize (SimulatedShoeboxLoader with anscombe=False bug)
standardized = ((counts_all * masks_all) - stats[0]) / stats[1].sqrt()
print("\nStandardized (non-anscombe path):")
print(
    f"  min={standardized.min():.2f}, max={standardized.max():.2f}, "
    f"mean={standardized.mean():.2f}, std={standardized.std():.2f}"
)
print(
    f"  has_nan={standardized.isnan().any()}, has_inf={standardized.isinf().any()}"
)

# Also check correct anscombe path
ans = 2 * torch.sqrt(counts_all + 3.0 / 8.0)
standardized_correct = (ans - stats[1]) / stats[1].sqrt()
print("\nStandardized (anscombe path - what SHOULD be used):")
print(
    f"  min={standardized_correct.min():.2f}, max={standardized_correct.max():.2f}, "
    f"mean={standardized_correct.mean():.2f}, std={standardized_correct.std():.2f}"
)

# Dirichlet prior
conc_prior = conc_raw.clone()
conc_prior[conc_prior > 2] *= 40
conc_prior /= conc_prior.sum()
alpha_prior = conc_prior.reshape(-1)
print(
    f"\nDirichlet prior: sum={alpha_prior.sum():.4f}, "
    f"min={alpha_prior.min():.6f}, max={alpha_prior.max():.6f}"
)

# ── Simple model mirroring the real architecture ─────────────────────────


class SimpleEncoder(nn.Module):
    """Mimics IntensityEncoder: conv→norm→relu→pool→conv→norm→relu→conv→norm→relu→pool→fc→relu"""

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
        x = F.relu(self.fc(x))
        return x


class GammaHead(nn.Module):
    """GammaDistributionRepamA"""

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
        qp = self.qp_head(x_profile)
        qi = self.qi_head(x_intensity)
        qbg = self.qbg_head(x_intensity)
        return qp, qi, qbg, x_profile, x_intensity


def compute_loss(
    qp, qi, qbg, counts, mask, alpha_prior, mc_samples=100, eps=1e-6
):
    """Mirrors the Loss module"""
    # MC samples
    zI = qi.rsample([mc_samples]).unsqueeze(-1).permute(1, 0, 2)
    zp = qp.rsample([mc_samples]).permute(1, 0, 2)
    zbg = qbg.rsample([mc_samples]).unsqueeze(-1).permute(1, 0, 2)

    # Rate
    rate = zI * zp + zbg

    # Poisson log-likelihood
    ll = Poisson(rate + eps).log_prob(counts.unsqueeze(1))
    ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
    neg_ll = (-ll_mean).sum(1)

    # KL terms
    p_dir = Dirichlet(alpha_prior)
    kl_prf = torch.distributions.kl.kl_divergence(qp, p_dir)

    p_i = Gamma(torch.tensor(1.0), torch.tensor(0.001))
    kl_i = torch.distributions.kl.kl_divergence(qi, p_i)

    p_bg = Exponential(torch.tensor(1.0))
    p_bg_gamma = Gamma(torch.tensor(1.0), torch.tensor(1.0))
    kl_bg = torch.distributions.kl.kl_divergence(qbg, p_bg_gamma)

    kl = kl_prf * 0.005 + kl_i * 0.5 + kl_bg * 0.5

    loss = (neg_ll + kl).mean()

    return {
        "loss": loss,
        "neg_ll": neg_ll.mean(),
        "kl_prf": kl_prf.mean(),
        "kl_i": kl_i.mean(),
        "kl_bg": kl_bg.mean(),
        "rate_max": rate.max(),
        "rate_min": rate.min(),
        "zI_max": zI.max(),
        "zI_min": zI.min(),
        "zp_max": zp.max(),
        "zp_min": zp.min(),
        "zbg_max": zbg.max(),
    }


# ── Training loop with NaN detection ────────────────────────────────────
print("\n" + "=" * 70)
print("TRAINING WITH NaN DETECTION")
print("=" * 70)

model = MiniModel()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)

n_samples = len(counts_all)
indices = torch.randperm(n_samples)

nan_found = False
for step in range(N_STEPS):
    # Get batch
    batch_idx = indices[
        (step * BATCH_SIZE) % n_samples : (step * BATCH_SIZE) % n_samples
        + BATCH_SIZE
    ]
    if len(batch_idx) < BATCH_SIZE:
        indices = torch.randperm(n_samples)
        batch_idx = indices[:BATCH_SIZE]

    batch_counts = counts_all[batch_idx]
    batch_std = standardized[batch_idx]
    batch_masks = masks_all[batch_idx]

    # Forward
    qp, qi, qbg, x_profile, x_intensity = model(batch_std)

    # Check encoder outputs
    enc_nan = x_intensity.isnan().any().item()
    enc_inf = x_intensity.isinf().any().item()
    enc_max = x_intensity.abs().max().item()
    prof_nan = x_profile.isnan().any().item()
    prof_max = x_profile.abs().max().item()

    # Check distribution params
    qi_k = qi.concentration
    qi_r = qi.rate
    qp_alpha = qp.concentration
    qbg_k = qbg.concentration
    qbg_r = qbg.rate

    k_nan = qi_k.isnan().any().item()
    k_max = qi_k.max().item()
    k_min = qi_k.min().item()
    r_max = qi_r.max().item()
    r_min = qi_r.min().item()
    alpha_max = qp_alpha.max().item()
    alpha_min = qp_alpha.min().item()
    alpha_sum_mean = qp_alpha.sum(-1).mean().item()

    # Compute loss
    try:
        loss_dict = compute_loss(
            qp,
            qi,
            qbg,
            batch_counts,
            batch_masks,
            alpha_prior,
            mc_samples=MC_SAMPLES,
        )
    except Exception as e:
        print(f"\nStep {step}: EXCEPTION in loss computation: {e}")
        print(
            f"  qi.concentration: nan={qi_k.isnan().any()}, min={k_min:.6g}, max={k_max:.6g}"
        )
        print(f"  qi.rate: min={r_min:.6g}, max={r_max:.6g}")
        print(f"  qp.alpha: min={alpha_min:.6g}, max={alpha_max:.6g}")
        print(f"  encoder_intensity: nan={enc_nan}, max={enc_max:.4f}")
        nan_found = True
        break

    loss = loss_dict["loss"]
    loss_nan = loss.isnan().item()
    loss_inf = loss.isinf().item()

    # Print every 10 steps or when something is wrong
    is_bad = loss_nan or loss_inf or enc_nan or enc_inf or k_nan
    if step % 25 == 0 or is_bad:
        print(
            f"Step {step:4d}: loss={loss.item() if not loss_nan else 'NaN':>12.2f}  "
            f"nll={loss_dict['neg_ll'].item():>10.2f}  "
            f"kl_prf={loss_dict['kl_prf'].item():>8.1f}  "
            f"kl_i={loss_dict['kl_i'].item():>8.2f}  "
            f"kl_bg={loss_dict['kl_bg'].item():>8.2f}  "
            f"k=[{k_min:.4f},{k_max:.4f}]  "
            f"r=[{r_min:.4f},{r_max:.4f}]  "
            f"α=[{alpha_min:.4f},{alpha_max:.4f}]  "
            f"α_sum={alpha_sum_mean:.1f}  "
            f"enc_max={enc_max:.2f}  "
            f"rate=[{loss_dict['rate_min'].item():.3g},{loss_dict['rate_max'].item():.3g}]"
        )

    if is_bad:
        print(f"\n*** NaN/Inf detected at step {step}! ***")
        print(f"  loss nan={loss_nan}, inf={loss_inf}")
        print(f"  encoder nan={enc_nan}, inf={enc_inf}, max={enc_max}")
        print(f"  qi.k: nan={k_nan}, min={k_min}, max={k_max}")
        print(f"  qi.r: min={r_min}, max={r_max}")
        print(
            f"  qbg.k: nan={qbg_k.isnan().any().item()}, min={qbg_k.min().item()}, max={qbg_k.max().item()}"
        )
        print(f"  neg_ll: {loss_dict['neg_ll'].item()}")
        print(f"  kl_prf: {loss_dict['kl_prf'].item()}")
        print(f"  kl_i: {loss_dict['kl_i'].item()}")
        print(f"  kl_bg: {loss_dict['kl_bg'].item()}")
        print(
            f"  rate: max={loss_dict['rate_max'].item()}, min={loss_dict['rate_min'].item()}"
        )
        print(
            f"  zI: max={loss_dict['zI_max'].item()}, min={loss_dict['zI_min'].item()}"
        )
        print(
            f"  zp: max={loss_dict['zp_max'].item()}, min={loss_dict['zp_min'].item()}"
        )
        print(f"  zbg: max={loss_dict['zbg_max'].item()}")

        # Check which weight has largest magnitude
        print("\n  Weight magnitudes:")
        for name, p in model.named_parameters():
            print(
                f"    {name}: max_abs={p.abs().max().item():.4g}, "
                f"nan={p.isnan().any().item()}"
            )
        nan_found = True
        break

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Check gradients before clipping
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

    if step % 25 == 0:
        grad_nan = any(
            p.grad.isnan().any().item()
            for p in model.parameters()
            if p.grad is not None
        )
        if grad_nan:
            print(
                f"  *** GRADIENT NaN at step {step} (before clip, norm={total_norm:.4g}) ***"
            )
            for name, p in model.named_parameters():
                if p.grad is not None and p.grad.isnan().any():
                    print(f"    {name}: grad has NaN")
            nan_found = True
            break

    optimizer.step()

if not nan_found:
    print(f"\nCompleted {N_STEPS} steps without NaN.")
    print(
        "The NaN may take longer to develop, or the issue is specific to GPU float32 numerics."
    )
    print("\nFinal model state:")
    for name, p in model.named_parameters():
        print(f"  {name}: max_abs={p.abs().max().item():.4g}")
