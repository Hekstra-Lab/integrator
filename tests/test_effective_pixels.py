import torch

from integrator.model.integrators.integrator_utils import _effective_pixels


def test_tophat_gives_its_own_width():
    """A top-hat over m pixels has n_eff = m exactly."""
    P, m = 64, 10
    profile = torch.zeros(3, P)
    profile[:, :m] = 1.0 / m
    mask = torch.ones(3, P)

    n_eff, n_valid = _effective_pixels(profile, mask)

    assert torch.allclose(n_eff, torch.full((3,), float(m)), atol=1e-4)
    assert torch.allclose(n_valid, torch.full((3,), float(P)))


def test_delta_and_uniform_are_the_two_extremes():
    """All mass on one pixel gives 1; uniform over P gives P."""
    P = 50
    delta = torch.zeros(1, P)
    delta[0, 7] = 1.0
    uniform = torch.full((1, P), 1.0 / P)
    mask = torch.ones(1, P)

    assert _effective_pixels(delta, mask)[0].item() == torch.tensor(1.0)
    assert abs(_effective_pixels(uniform, mask)[0].item() - P) < 1e-3


def test_broader_profile_has_larger_n_eff():
    """n_eff orders profiles by breadth, which is the property the theory uses."""
    P = 128
    x = torch.arange(P, dtype=torch.float32) - P / 2
    mask = torch.ones(1, P)

    widths = [2.0, 5.0, 15.0]
    n_effs = []
    for s in widths:
        p = torch.exp(-0.5 * (x / s) ** 2)
        p = (p / p.sum()).unsqueeze(0)
        n_effs.append(_effective_pixels(p, mask)[0].item())

    assert n_effs == sorted(n_effs)
    # a Gaussian profile has n_eff = 2*sqrt(pi)*sigma
    assert abs(n_effs[0] - 2 * torch.pi**0.5 * 2.0) < 0.5


def test_masked_pixels_are_excluded_and_profile_renormalized():
    """Mass on masked-out pixels must not count toward n_eff."""
    P = 40
    profile = torch.zeros(1, P)
    profile[0, :8] = 0.5 / 8      # inside the mask
    profile[0, 20:] = 0.5 / 20    # outside it
    mask = torch.zeros(1, P)
    mask[0, :10] = 1.0

    n_eff, n_valid = _effective_pixels(profile, mask)

    assert abs(n_eff.item() - 8.0) < 1e-3
    assert n_valid.item() == 10.0


def test_accepts_trailing_spatial_dims():
    """3D shoebox profiles arrive shaped (B, D, H, W); flattening must not change n_eff."""
    flat = torch.rand(4, 3 * 5 * 7)
    flat = flat / flat.sum(-1, keepdim=True)
    mask = torch.ones(4, 3 * 5 * 7)

    a, _ = _effective_pixels(flat, mask)
    b, _ = _effective_pixels(flat.reshape(4, 3, 5, 7), mask.reshape(4, 3, 5, 7))

    assert torch.allclose(a, b)


def test_sample_axis_located_regardless_of_layout():
    """zp arrives as (batch, samples, pixels) on the hierarchical path and
    (samples, batch, pixels) elsewhere; averaging must pick the sample axis."""
    B, S, K = 7, 4, 50
    qp_mean = torch.rand(B, K)
    mask = torch.ones(B, K)

    bsk = torch.rand(B, S, K)
    sbk = bsk.permute(1, 0, 2).contiguous()

    def reduce(zp):
        batch = qp_mean.shape[0]
        axis = 1 if zp.shape[0] == batch else 0
        return _effective_pixels(zp.mean(axis), mask)[0]

    assert reduce(bsk).shape == (B,)
    assert torch.allclose(reduce(bsk), reduce(sbk))


def test_mc_average_and_softmax_at_the_mean_differ_in_both_directions():
    """`E[softmax(h)]` and `softmax(E[h])` are different profiles with no fixed
    ordering: softmax is neither convex nor concave, so averaging over the latent
    can be broader or sharper than decoding the posterior mean. Both are reported
    because the likelihood uses the former and the exported profile the latter."""
    torch.manual_seed(0)
    B, S, d, K = 64, 256, 4, 200
    W = torch.randn(K, d) * 2.0
    loc, scale = torch.randn(B, d), torch.full((B, d), 0.8)

    h = loc + scale * torch.randn(S, B, d)
    mask = torch.ones(B, K)

    n_eff = _effective_pixels(torch.softmax(loc @ W.T, dim=-1), mask)[0]
    n_eff_mc = _effective_pixels(torch.softmax(h @ W.T, dim=-1).mean(0), mask)[0]

    assert (n_eff_mc > n_eff).any()
    assert (n_eff_mc < n_eff).any()


def test_the_two_breadths_agree_when_the_latent_is_certain():
    """With no posterior spread the mixture collapses onto the decoded mean."""
    torch.manual_seed(0)
    B, S, d, K = 8, 512, 4, 120
    W = torch.randn(K, d)
    loc = torch.randn(B, d)

    h = loc + 1e-4 * torch.randn(S, B, d)
    mask = torch.ones(B, K)

    n_eff = _effective_pixels(torch.softmax(loc @ W.T, dim=-1), mask)[0]
    n_eff_mc = _effective_pixels(torch.softmax(h @ W.T, dim=-1).mean(0), mask)[0]

    assert torch.allclose(n_eff, n_eff_mc, rtol=1e-3)
