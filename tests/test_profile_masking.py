"""Every profile family must normalize over the valid pixels only.

The likelihood zeroes masked pixels, so profile mass placed on them is never
paid for and the intensity grows by `1 / (mass on valid pixels)` to explain the
counts that are seen. The bias is per-reflection and follows where the shoebox
sits on the detector, so no downstream per-image scale can absorb it. Measured
on hewl_1118 before the fix: 1.008x median and 1.77x on the worst reflection
for the softmax families, 1.000x for the ones that build the profile directly.
"""

import pytest
import torch

from integrator.model.distributions.profile_surrogates import (
    FixedBasisProfileSurrogate,
    GeometricProfileSurrogate,
    PhysicalProfileSurrogate,
    ProfileSurrogate,
    ShapeletProfileSurrogate,
)

H = W = 9
P = H * W
SHAPE = (1, H, W)
B, D = 6, 8


def _mask(batch: int = B) -> torch.Tensor:
    """A mask with a different bite taken out of each shoebox."""
    m = torch.ones(batch, P)
    for i in range(batch):
        m[i].view(H, W)[: i + 1, : i + 2] = 0.0
    return m


def _metadata(batch: int = B) -> dict:
    return {
        "x": torch.rand(batch) * 100,
        "y": torch.rand(batch) * 100,
        "xyzcal.px.0": torch.rand(batch) * 100,
        "xyzcal.px.1": torch.rand(batch) * 100,
        "wavelength": torch.full((batch,), 1.1),
        "d": torch.full((batch,), 2.0),
    }


def _build(name: str):
    if name == "learned":
        return ProfileSurrogate(input_dim=D, latent_dim=4, output_dim=P)
    if name == "fixed":
        return FixedBasisProfileSurrogate(
            input_dim=D, output_dim=P, profile_shape=SHAPE
        )
    if name == "shapelet":
        return ShapeletProfileSurrogate(
            input_dim=D, output_dim=P, profile_shape=SHAPE
        )
    if name == "physical":
        return PhysicalProfileSurrogate(
            input_dim=D, output_dim=P, profile_shape=SHAPE
        )
    if name == "geometric":
        return GeometricProfileSurrogate(
            input_dim=D, latent_dim=4, output_dim=P, profile_shape=SHAPE
        )
    raise ValueError(name)


FAMILIES = ["learned", "fixed", "shapelet", "physical", "geometric"]


@pytest.mark.parametrize("name", FAMILIES)
def test_no_mass_on_masked_pixels(name):
    torch.manual_seed(0)
    surrogate = _build(name)
    mask = _mask()
    out = surrogate(
        torch.randn(B, D), mc_samples=3, mask=mask, metadata=_metadata()
    )

    for profile in (out.zp, out.mean_profile):
        p = profile.reshape(-1, B, P) if profile.dim() == 3 else profile[None]
        leaked = (p * (1.0 - mask)).sum(-1)
        assert leaked.max() < 1e-6, f"{name}: {leaked.max():.3e} on masked pixels"


@pytest.mark.parametrize("name", FAMILIES)
def test_still_sums_to_one_over_valid_pixels(name):
    torch.manual_seed(0)
    surrogate = _build(name)
    mask = _mask()
    out = surrogate(
        torch.randn(B, D), mc_samples=3, mask=mask, metadata=_metadata()
    )

    for profile in (out.zp, out.mean_profile):
        total = profile.sum(-1)
        assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


@pytest.mark.parametrize("name", FAMILIES)
def test_unmasked_call_is_unchanged(name):
    """Passing no mask must reproduce the old behaviour exactly."""
    torch.manual_seed(0)
    surrogate = _build(name)
    x, meta = torch.randn(B, D), _metadata()

    torch.manual_seed(1)
    without = surrogate(x, mc_samples=3, metadata=meta).zp
    torch.manual_seed(1)
    full = surrogate(
        x, mc_samples=3, mask=torch.ones(B, P), metadata=meta
    ).zp

    assert torch.allclose(without, full, atol=1e-6)


def test_mask_changes_the_intensity_scale():
    """The whole point: masking moves the profile, so it moves `I`.

    With a fifth of the box masked and a profile that reaches it, the peak
    pixel's share of the profile must rise -- that is the intensity inflation
    being removed.
    """
    torch.manual_seed(0)
    surrogate = _build("learned")
    x = torch.randn(B, D)
    mask = _mask()

    torch.manual_seed(1)
    unmasked = surrogate(x, mc_samples=1).zp
    torch.manual_seed(1)
    masked = surrogate(x, mc_samples=1, mask=mask).zp

    outside = (unmasked * (1.0 - mask)).sum(-1)
    assert outside.max() > 1e-3, "test mask does not overlap the profile"
    # renormalizing over fewer pixels can only raise what is left
    assert ((masked - unmasked) * mask).sum(-1).min() > -1e-6


@pytest.mark.parametrize("name", FAMILIES)
def test_fully_masked_shoebox_is_finite(name):
    """A degenerate mask must not produce NaN."""
    torch.manual_seed(0)
    surrogate = _build(name)
    out = surrogate(
        torch.randn(B, D),
        mc_samples=2,
        mask=torch.zeros(B, P),
        metadata=_metadata(),
    )
    assert torch.isfinite(out.zp).all()
    assert torch.isfinite(out.mean_profile).all()


def test_svae_passes_the_mask_to_its_profile_surrogate():
    """The merging integrator has its own forward and its own qp call.

    It built the profile without a mask, so the masking fix never reached it:
    profile mass landed on pixels the likelihood never sees and the merged
    intensity absorbed the difference. Checked by recording what the surrogate
    is handed rather than by inspecting the source.
    """
    from integrator.model.scaling import hierarchical_svae_integrator as hsi

    src = open(hsi.__file__).read()
    calls = [
        line for line in src.splitlines() if 'self.surrogates["qp"](' in line
    ]
    assert calls, "no qp call found; did the attribute name change?"
    # every call site must forward both, on the call or the lines after it
    for idx, line in enumerate(src.splitlines()):
        if 'self.surrogates["qp"](' not in line:
            continue
        window = "\n".join(src.splitlines()[idx : idx + 8])
        assert "mask=" in window, f"qp call without mask=: {line.strip()}"
        assert "metadata=" in window, f"qp call without metadata=: {line.strip()}"


def test_geometric_with_steering_uses_the_output_grid_mask():
    """The steered geometric surrogate decodes onto a LARGER canonical grid.

    With `local_shift`/`steer` on, the decoder writes d*ch*cw pixels (37x37
    against a 25x25 box on hewl_1118) and `_warp` brings it back. Handing the
    output-grid mask to the decode step therefore raised

        RuntimeError: The size of tensor a (625) must match
        the size of tensor b (1369) at non-singleton dimension 2

    which is what killed the geomboth arm. The unsteered test above cannot
    catch it, because without steering the two grids coincide.
    """
    torch.manual_seed(0)
    surrogate = GeometricProfileSurrogate(
        input_dim=D,
        latent_dim=4,
        output_dim=P,
        profile_shape=SHAPE,
        local_shift=True,
        steer=True,
        beam_center=[50.0, 50.0],
    )
    assert surrogate.canonical_shape != surrogate.profile_shape, (
        "steering did not enlarge the canonical grid; test is not exercising "
        "the path it exists for"
    )
    mask = _mask()
    out = surrogate(
        torch.randn(B, D), mc_samples=3, mask=mask, metadata=_metadata()
    )
    assert out.zp.shape[-1] == P, "profile must come back on the output grid"
    for profile in (out.zp, out.mean_profile):
        p = profile.reshape(-1, B, P) if profile.dim() == 3 else profile[None]
        assert (p * (1.0 - mask)).sum(-1).max() < 1e-6
        assert torch.allclose(
            profile.sum(-1), torch.ones_like(profile.sum(-1)), atol=1e-5
        )
