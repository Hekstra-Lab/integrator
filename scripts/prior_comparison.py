"""
Prior comparison: which priors allow proper training with all weights = 1.0?

Tests:
  1) Gamma(1, 0.001)  — current (extremely vague, mean=1000, CV=100%)
  2) Gamma(2, 0.002)  — same mean=1000 but mode=500, CV=71%
  3) Gamma(5, 0.005)  — same mean=1000, mode=800, CV=45%
  4) Gamma(10, 0.01)  — same mean=1000, mode=900, CV=32%
  5) Exponential(0.01) = Gamma(1, 0.01) — mean=100
  6) Hierarchical (learned) — starts at Gamma(1, 0.001), learns params

Also tests background priors:
  - Gamma(1, 1) vs Gamma(2, 0.5) for background

Tracks: loss, NLL, KL components, k growth, gradient norms, convergence speed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Gamma, Poisson

torch.manual_seed(42)

DATA_DIR = (
    "/Users/luis/master/notebooks/integrator_notes/code/simulating_shoeboxes"
)
BATCH_SIZE = 128
MC_SAMPLES = 10
H, W = 21, 21
N_PIXELS = H * W
LR = 0.001
GRAD_CLIP = 1.0
N_STEPS = 3000  # longer to see if things get stuck

# ── Load data ────────────────────────────────────────────────────────────
counts_all = torch.load(f"{DATA_DIR}/counts.pt", weights_only=False).squeeze(
    -1
)
masks_all = torch.load(f"{DATA_DIR}/masks.pt", weights_only=False).squeeze(-1)
stats = torch.load(f"{DATA_DIR}/stats_anscombe.pt", weights_only=False)
conc_raw = torch.load(f"{DATA_DIR}/concentration.pt", weights_only=False)

# Anscombe standardization
ans = 2 * torch.sqrt(counts_all + 3.0 / 8.0)
standardized = (ans - stats[0]) / stats[1].sqrt()

# Dirichlet prior (scaled to sum = N_PIXELS)
conc_prior = conc_raw.clone()
conc_prior[conc_prior > 2] *= 40
n_components = conc_prior.numel()
conc_prior = conc_prior / conc_prior.sum() * n_components
alpha_prior = conc_prior.reshape(-1)

print(f"Data: {counts_all.shape[0]} samples")
print(
    f"Standardized: range [{standardized.min():.2f}, {standardized.max():.2f}]"
)
print(f"Dirichlet prior: sum={alpha_prior.sum():.1f}")


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


class LearnableGammaPrior(nn.Module):
    """Learnable Gamma prior with LogNormal hyperprior."""

    def __init__(self, init_conc=1.0, init_rate=0.001, hyperprior_scale=2.0):
        super().__init__()
        import math

        self.log_conc = nn.Parameter(torch.tensor(math.log(init_conc)))
        self.log_rate = nn.Parameter(torch.tensor(math.log(init_rate)))
        self.hyperprior_scale = hyperprior_scale

    @property
    def concentration(self):
        return self.log_conc.exp()

    @property
    def rate(self):
        return self.log_rate.exp()

    def prior(self):
        return Gamma(self.concentration, self.rate)

    def hyperprior_lp(self):
        from torch.distributions import LogNormal

        hp = LogNormal(0.0, self.hyperprior_scale)
        return hp.log_prob(self.concentration) + hp.log_prob(self.rate)


def compute_loss(
    qp,
    qi,
    qbg,
    counts,
    mask,
    alpha_prior,
    p_i,
    p_bg,
    mc_samples=10,
    eps=1e-6,
    learnable_i=None,
    learnable_bg=None,
):
    """All KL weights = 1.0."""
    zI = qi.rsample([mc_samples]).unsqueeze(-1).permute(1, 0, 2)
    zp = qp.rsample([mc_samples]).permute(1, 0, 2)
    zbg = qbg.rsample([mc_samples]).unsqueeze(-1).permute(1, 0, 2)
    rate = zI * zp + zbg

    ll = Poisson(rate + eps).log_prob(counts.unsqueeze(1))
    ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
    neg_ll = (-ll_mean).sum(1)

    # Profile KL
    p_dir = Dirichlet(alpha_prior)
    kl_prf = torch.distributions.kl.kl_divergence(qp, p_dir)

    # Intensity KL
    if learnable_i is not None:
        kl_i = torch.distributions.kl.kl_divergence(qi, learnable_i.prior())
    else:
        kl_i = torch.distributions.kl.kl_divergence(qi, p_i)

    # Background KL
    if learnable_bg is not None:
        kl_bg = torch.distributions.kl.kl_divergence(qbg, learnable_bg.prior())
    else:
        kl_bg = torch.distributions.kl.kl_divergence(qbg, p_bg)

    # All weights = 1.0
    kl = kl_prf + kl_i + kl_bg
    loss = (neg_ll + kl).mean()

    # Add hyperprior for learnable priors
    hyperprior_lp = torch.zeros(1)
    if learnable_i is not None:
        hyperprior_lp = hyperprior_lp + learnable_i.hyperprior_lp()
    if learnable_bg is not None:
        hyperprior_lp = hyperprior_lp + learnable_bg.hyperprior_lp()
    loss = loss - hyperprior_lp.squeeze()

    return {
        "loss": loss,
        "neg_ll": neg_ll.mean().item(),
        "kl_prf": kl_prf.mean().item(),
        "kl_i": kl_i.mean().item(),
        "kl_bg": kl_bg.mean().item(),
        "kl_total": kl.mean().item(),
        "rate_mean": rate.mean().item(),
        "rate_max": rate.max().item(),
    }


def run_experiment(
    name,
    p_i_conc,
    p_i_rate,
    p_bg_conc,
    p_bg_rate,
    hierarchical_i=False,
    hierarchical_bg=False,
    seed=42,
):
    print(f"\n{'=' * 80}")
    print(f"  {name}")
    print(
        f"  I prior: Gamma({p_i_conc}, {p_i_rate}) mean={p_i_conc / p_i_rate:.0f}"
    )
    print(
        f"  BG prior: Gamma({p_bg_conc}, {p_bg_rate}) mean={p_bg_conc / p_bg_rate:.0f}"
    )
    if hierarchical_i:
        print("  ** Intensity prior is LEARNABLE (starts at above values)")
    print("  All KL weights = 1.0")
    print(f"{'=' * 80}")

    torch.manual_seed(seed)
    model = MiniModel()

    # Set up priors
    p_i = Gamma(torch.tensor(float(p_i_conc)), torch.tensor(float(p_i_rate)))
    p_bg = Gamma(
        torch.tensor(float(p_bg_conc)), torch.tensor(float(p_bg_rate))
    )

    learnable_i = None
    learnable_bg = None
    params = list(model.parameters())

    if hierarchical_i:
        learnable_i = LearnableGammaPrior(
            init_conc=p_i_conc, init_rate=p_i_rate, hyperprior_scale=2.0
        )
        params += list(learnable_i.parameters())
    if hierarchical_bg:
        learnable_bg = LearnableGammaPrior(
            init_conc=p_bg_conc, init_rate=p_bg_rate, hyperprior_scale=2.0
        )
        params += list(learnable_bg.parameters())

    optimizer = torch.optim.Adam(params, lr=LR)
    n_samples = len(counts_all)

    history = []
    nan_step = None

    for step in range(N_STEPS):
        start = (step * BATCH_SIZE) % n_samples
        end = start + BATCH_SIZE
        if end > n_samples:
            start = 0
            end = BATCH_SIZE

        batch_std = standardized[start:end]
        batch_counts = counts_all[start:end]
        batch_masks = masks_all[start:end]

        qp, qi, qbg = model(batch_std)

        loss_dict = compute_loss(
            qp,
            qi,
            qbg,
            batch_counts,
            batch_masks,
            alpha_prior,
            p_i=p_i,
            p_bg=p_bg,
            mc_samples=MC_SAMPLES,
            learnable_i=learnable_i,
            learnable_bg=learnable_bg,
        )

        loss = loss_dict["loss"]
        if loss.isnan() or loss.isinf():
            print(
                f"  Step {step}: loss is {'NaN' if loss.isnan() else 'Inf'}!"
            )
            nan_step = step
            break

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)

        if torch.isnan(grad_norm):
            print(f"  Step {step}: gradient NaN!")
            nan_step = step
            break

        optimizer.step()

        if step % 100 == 0:
            k_mean = qi.concentration.mean().item()
            k_max = qi.concentration.max().item()
            r_mean = qi.rate.mean().item()
            qi_mean_I = (qi.concentration / qi.rate).mean().item()

            record = {
                "step": step,
                "loss": loss.item(),
                "neg_ll": loss_dict["neg_ll"],
                "kl_prf": loss_dict["kl_prf"],
                "kl_i": loss_dict["kl_i"],
                "kl_bg": loss_dict["kl_bg"],
                "kl_total": loss_dict["kl_total"],
                "grad_norm": grad_norm.item(),
                "qi_k_mean": k_mean,
                "qi_k_max": k_max,
                "qi_r_mean": r_mean,
                "qi_mean_I": qi_mean_I,
                "rate_mean": loss_dict["rate_mean"],
            }

            if learnable_i is not None:
                record["learned_i_conc"] = learnable_i.concentration.item()
                record["learned_i_rate"] = learnable_i.rate.item()
                record["learned_i_mean"] = (
                    learnable_i.concentration / learnable_i.rate
                ).item()

            history.append(record)

            if step % 300 == 0:
                kl_frac = loss_dict["kl_total"] / (
                    loss_dict["neg_ll"] + loss_dict["kl_total"] + 1e-8
                )
                extra = ""
                if learnable_i is not None:
                    lc = learnable_i.concentration.item()
                    lr_ = learnable_i.rate.item()
                    extra = f"  p_I=Gamma({lc:.3f},{lr_:.5f})"
                print(
                    f"  Step {step:4d}: loss={loss.item():>8.1f}  "
                    f"nll={loss_dict['neg_ll']:>8.1f}  "
                    f"kl_i={loss_dict['kl_i']:>7.1f}  "
                    f"kl_tot={loss_dict['kl_total']:>7.1f} ({kl_frac:.0%})  "
                    f"k_mean={k_mean:>7.1f}  k_max={k_max:>7.1f}  "
                    f"I={qi_mean_I:>7.0f}  "
                    f"grad={grad_norm.item():>6.1f}{extra}"
                )

    result = {
        "name": name,
        "nan_step": nan_step,
        "history": history,
    }
    if nan_step is not None:
        print(f"  *** FAILED: NaN at step {nan_step} ***")
    else:
        print(f"  Completed {N_STEPS} steps.")
        if history:
            h = history[-1]
            print(
                f"  Final: loss={h['loss']:.1f}, nll={h['neg_ll']:.1f}, "
                f"kl_i={h['kl_i']:.1f}, k_mean={h['qi_k_mean']:.1f}, "
                f"k_max={h['qi_k_max']:.1f}"
            )

    return result


# ── Run experiments ──────────────────────────────────────────────────────

results = []

# 1) Current prior — extremely vague
results.append(
    run_experiment(
        "1) Gamma(1, 0.001) — current prior (mean=1000, CV=100%)",
        p_i_conc=1.0,
        p_i_rate=0.001,
        p_bg_conc=1.0,
        p_bg_rate=1.0,
    )
)

# 2) Slightly informative
results.append(
    run_experiment(
        "2) Gamma(2, 0.002) — mean=1000, CV=71%",
        p_i_conc=2.0,
        p_i_rate=0.002,
        p_bg_conc=1.0,
        p_bg_rate=1.0,
    )
)

# 3) Moderately informative
results.append(
    run_experiment(
        "3) Gamma(5, 0.005) — mean=1000, CV=45%",
        p_i_conc=5.0,
        p_i_rate=0.005,
        p_bg_conc=1.0,
        p_bg_rate=1.0,
    )
)

# 4) Informative
results.append(
    run_experiment(
        "4) Gamma(10, 0.01) — mean=1000, CV=32%",
        p_i_conc=10.0,
        p_i_rate=0.01,
        p_bg_conc=1.0,
        p_bg_rate=1.0,
    )
)

# 5) Exponential with more reasonable rate
results.append(
    run_experiment(
        "5) Gamma(1, 0.01) = Exp(0.01) — mean=100",
        p_i_conc=1.0,
        p_i_rate=0.01,
        p_bg_conc=1.0,
        p_bg_rate=1.0,
    )
)

# 6) Hierarchical (learned) starting from current
results.append(
    run_experiment(
        "6) Hierarchical (starts Gamma(1, 0.001), learned)",
        p_i_conc=1.0,
        p_i_rate=0.001,
        p_bg_conc=1.0,
        p_bg_rate=1.0,
        hierarchical_i=True,
    )
)

# 7) Hierarchical starting from better init
results.append(
    run_experiment(
        "7) Hierarchical (starts Gamma(2, 0.002), learned)",
        p_i_conc=2.0,
        p_i_rate=0.002,
        p_bg_conc=1.0,
        p_bg_rate=1.0,
        hierarchical_i=True,
    )
)

# 8) Both priors hierarchical
results.append(
    run_experiment(
        "8) Both hierarchical (I: Gamma(2,0.002), BG: Gamma(1,1))",
        p_i_conc=2.0,
        p_i_rate=0.002,
        p_bg_conc=1.0,
        p_bg_rate=1.0,
        hierarchical_i=True,
        hierarchical_bg=True,
    )
)


# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 100}")
print("SUMMARY — all weights = 1.0, 3000 steps")
print(f"{'=' * 100}")
print(
    f"{'Experiment':<55} {'status':>8} {'loss':>7} {'nll':>7} {'kl_i':>7} {'k_mean':>7} {'k_max':>7}"
)
print("-" * 100)

for r in results:
    if r["nan_step"] is not None:
        print(f"{r['name']:<55} {'NaN@' + str(r['nan_step']):>8}")
    elif r["history"]:
        h = r["history"][-1]
        print(
            f"{r['name']:<55} {'  OK':>8} {h['loss']:>7.0f} {h['neg_ll']:>7.0f} "
            f"{h['kl_i']:>7.1f} {h['qi_k_mean']:>7.1f} {h['qi_k_max']:>7.1f}"
        )


# ── K growth analysis ────────────────────────────────────────────────────
print(f"\n{'=' * 100}")
print("K GROWTH TRAJECTORIES (qi concentration)")
print(f"{'=' * 100}")
print(f"{'step':>5}", end="")
for r in results:
    label = r["name"][:12]
    print(f"  {label:>12}", end="")
print()
print("-" * (5 + 14 * len(results)))

# Get all steps
all_steps = set()
for r in results:
    for h in r["history"]:
        all_steps.add(h["step"])

for step in sorted(all_steps):
    if step % 300 != 0:
        continue
    print(f"{step:>5}", end="")
    for r in results:
        found = False
        for h in r["history"]:
            if h["step"] == step:
                print(f"  {h['qi_k_max']:>12.1f}", end="")
                found = True
                break
        if not found:
            print(f"  {'—':>12}", end="")
    print()


# ── KL analysis: how does KL_i scale with k for each prior? ─────────────
print(f"\n{'=' * 80}")
print("ANALYTICAL KL ANALYSIS: KL(Gamma(k, r) || prior) as k grows")
print("  (fixing r = k/1000 so mean intensity = 1000)")
print(f"{'=' * 80}")

ks = [1, 5, 10, 50, 100, 500, 1000, 5000]
priors = [
    ("Gamma(1, 0.001)", Gamma(torch.tensor(1.0), torch.tensor(0.001))),
    ("Gamma(2, 0.002)", Gamma(torch.tensor(2.0), torch.tensor(0.002))),
    ("Gamma(5, 0.005)", Gamma(torch.tensor(5.0), torch.tensor(0.005))),
    ("Gamma(10, 0.01)", Gamma(torch.tensor(10.0), torch.tensor(0.01))),
    ("Exp(0.01)", Gamma(torch.tensor(1.0), torch.tensor(0.01))),
]

header = f"{'k':>6}"
for name, _ in priors:
    header += f"  {name:>16}"
print(header)
print("-" * (6 + 18 * len(priors)))

for k_val in ks:
    k = torch.tensor(float(k_val))
    r = k / 1000.0  # fix mean = 1000
    q = Gamma(k, r)
    line = f"{k_val:>6}"
    for _, p in priors:
        kl = torch.distributions.kl.kl_divergence(q, p).item()
        line += f"  {kl:>16.1f}"
    print(line)

print(
    "\n(At k=1000 with mean=1000, the posterior is extremely precise: CV=3%)"
)
print("(The prior needs to provide enough KL penalty to prevent this)")
