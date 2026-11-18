"""Tests for the ported DIALS physical scaling model.

Three levels of check: the Gaussian smoother against the numbers in
DIALS' own `tests/algorithms/scaling/test_gaussian_smoother.py`, the
spherical harmonic basis against closed-form real harmonics, and a
forward evaluation against a DIALS-written `inverse_scale_factor`.
"""

from __future__ import annotations

from math import exp
from pathlib import Path

import numpy as np
import pytest
import torch

from integrator.dials_port.harmonics import (
    ALIGNMENT_AXIS,
    SQRT2_DIALS,
    align_axis_along_z,
    calc_theta_phi,
    n_abs_params,
    scan_angles,
    sph_harm_basis,
)
from integrator.dials_port.io import Experiment, ScalingModelRef, load_npz
from integrator.dials_port.scale_model import (
    PhysicalScalingModel,
    ScaleFeatures,
    build_features,
)
from integrator.dials_port.smoother import (
    GaussianSmoother1D,
    build_smoother,
    nparam_to_val,
)
from integrator.dials_port.types import DTYPE, Reflections

_EXPT = Path("/Users/luis/dials_out/816_sbgrid_HEWL/pass1/scaled.expt")
_CACHE = Path(
    "/private/tmp/claude-501/-Users-luis-integrator/"
    "a69c47c1-a7b1-409e-b127-f918bd14acbc/scratchpad/hewl_scaled.npz"
)


class TestSmootherConstruction:
    """Constructor semantics, checked against the DIALS test values."""

    @pytest.mark.parametrize(
        ("num_intervals", "n_values", "n_average", "sigma"),
        [
            (1, 2, 2, 0.65),
            (2, 3, 3, 0.70),
            (3, 5, 3, 0.70),
            (4, 6, 3, 0.70),
            (5, 7, 3, 0.70),
        ],
    )
    def test_shape(
        self,
        num_intervals: int,
        n_values: int,
        n_average: int,
        sigma: float,
    ) -> None:
        smoother = GaussianSmoother1D(
            (0.0, float(num_intervals)), num_intervals
        )
        assert smoother.n_values == n_values
        assert smoother.n_average == n_average
        assert smoother.sigma == pytest.approx(sigma)
        assert smoother.spacing == pytest.approx(1.0)

    def test_positions(self) -> None:
        small = GaussianSmoother1D((0.0, 1.0), 1)
        assert small.positions.tolist() == [0.0, 1.0]

        large = GaussianSmoother1D((0.0, 4.0), 4)
        assert large.positions.tolist() == [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]

    @pytest.mark.parametrize(
        ("n_params", "expected"), [(2, 1), (3, 2), (4, 2), (50, 48), (38, 36)]
    )
    def test_nparam_to_val(self, n_params: int, expected: int) -> None:
        assert nparam_to_val(n_params) == expected

    def test_too_few_parameters(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            nparam_to_val(1)


class TestSmootherAgainstDialsTestValues:
    """Reproduce `test_2DGaussianSmoother`'s hand-derived numbers.

    That test drives a 2D smoother, which shares a single sigma across
    both axes, so the 1D smoothers are forced onto sigma 0.65 and their
    weights combined by outer product. Every other quantity -- the index
    range, the `exp(-ds^2)` kernel, the parameter positions -- is the
    1D code under test.
    """

    @staticmethod
    def _weights() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_smoother = GaussianSmoother1D((0.0, 1.0), 1)
        y_smoother = GaussianSmoother1D((0.0, 4.0), 4)
        x_smoother.set_smoothing(3, 0.65)
        y_smoother.set_smoothing(3, 0.65)
        x = x_smoother.weights(torch.tensor([0.5], dtype=DTYPE))
        y = y_smoother.weights(torch.tensor([1.0], dtype=DTYPE))
        return x.weight[0], y.weight[0], y.index[0]

    def test_sum_of_weights(self) -> None:
        x_weight, y_weight, _ = self._weights()
        expected = 2.0 * exp(-2.5 / 0.65**2) + 4.0 * exp(-0.5 / 0.65**2)
        total = float(torch.outer(x_weight, y_weight).sum())
        assert total == pytest.approx(expected)

    def test_interpolated_value(self) -> None:
        x_weight, y_weight, y_index = self._weights()
        # 6 x 2 grid of parameters, x fastest, as in the DIALS test.
        params = torch.tensor(
            [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0],
            dtype=DTYPE,
        ).reshape(6, 2)
        weight = torch.outer(y_weight, x_weight)
        numerator = 10.0 * exp(-0.5 / 0.65**2) + 2.0 * exp(-2.5 / 0.65**2)
        denominator = 2.0 * exp(-2.5 / 0.65**2) + 4.0 * exp(-0.5 / 0.65**2)
        value = float((weight * params[y_index, :2]).sum() / weight.sum())
        assert value == pytest.approx(numerator / denominator)

    def test_index_range_at_lower_edge(self) -> None:
        smoother = GaussianSmoother1D((0.0, 4.0), 4)
        weights = smoother.weights(torch.tensor([1.0], dtype=DTYPE))
        assert weights.index[0].tolist() == [0, 1, 2]
        assert (weights.weight[0] > 0).all()


class TestSmootherBehaviour:
    """Properties the vectorised implementation must preserve."""

    def test_matches_scalar_reference(self) -> None:
        smoother = GaussianSmoother1D((0.0, 12.0), 12)
        params = torch.linspace(0.5, 2.0, smoother.n_values, dtype=DTYPE)
        x = torch.linspace(0.0, 12.0, 61, dtype=DTYPE)

        positions = smoother.positions.numpy()
        expected = []
        for value in x.tolist():
            z = value / smoother.spacing
            start = int(np.sign(z - 1.5) * np.floor(abs(z - 1.5) + 0.5)) + 1
            stop = start + 3
            if start < 0:
                start, stop = 0, max(2, stop)
            if stop > smoother.n_values:
                start, stop = (
                    min(start, smoother.n_values - 2),
                    smoother.n_values,
                )
            total = weighted = 0.0
            for i in range(start, stop):
                w = exp(-(((z - positions[i]) / smoother.sigma) ** 2))
                total += w
                weighted += w * float(params[i])
            expected.append(weighted / total if total > 0 else 0.0)

        got = smoother.value(x, params)
        assert got.numpy() == pytest.approx(np.asarray(expected))

    def test_constant_parameters_reproduce_the_constant(self) -> None:
        smoother = GaussianSmoother1D((0.0, 20.0), 20)
        params = torch.full((smoother.n_values,), 1.7, dtype=DTYPE)
        x = torch.linspace(0.0, 20.0, 97, dtype=DTYPE)
        assert smoother.value(x, params).numpy() == pytest.approx(1.7)

    def test_is_differentiable(self) -> None:
        smoother = GaussianSmoother1D((0.0, 8.0), 8)
        params = torch.ones(smoother.n_values, dtype=DTYPE, requires_grad=True)
        x = torch.linspace(0.0, 8.0, 33, dtype=DTYPE)
        smoother.value(x, params).sum().backward()
        assert params.grad is not None
        assert torch.isfinite(params.grad).all()
        assert (params.grad != 0).any()

    def test_build_smoother_zeroes_the_coordinate(self) -> None:
        x = torch.linspace(3.0, 11.0, 50, dtype=DTYPE)
        smoother, zeroed = build_smoother(x, 10)
        assert float(zeroed.min()) == pytest.approx(0.0)
        assert smoother.n_values == 10

    def test_build_smoother_rejects_a_mismatch(self) -> None:
        x = torch.linspace(0.0, 10.0, 50, dtype=DTYPE)
        # 4 parameters imply 2 intervals, which only holds 3 values.
        with pytest.raises(ValueError, match="parameters"):
            build_smoother(x, 4)


def _cartesian_real_harmonics(
    v: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    """Closed-form real spherical harmonics for `l = 1, 2`."""
    r = np.linalg.norm(v, axis=1)
    x, y, z = v[:, 0] / r, v[:, 1] / r, v[:, 2] / r
    c1 = np.sqrt(3.0 / (4.0 * np.pi))
    c2 = 0.5 * np.sqrt(15.0 / np.pi)
    return {
        (1, -1): c1 * y,
        (1, 0): c1 * z,
        (1, 1): c1 * x,
        (2, -2): c2 * x * y,
        (2, -1): c2 * y * z,
        (2, 0): 0.25 * np.sqrt(5.0 / np.pi) * (3.0 * z * z - 1.0),
        (2, 1): c2 * x * z,
        (2, 2): 0.25 * np.sqrt(15.0 / np.pi) * (x * x - y * y),
    }


class TestSphericalHarmonics:
    """The absorption basis, its ordering and its normalisation."""

    @staticmethod
    def _directions(seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(size=(64, 3))

    def test_matches_closed_form(self) -> None:
        v = self._directions()
        theta, phi = calc_theta_phi(v)
        basis = sph_harm_basis(theta, phi, theta, phi, lmax=2)
        expected = _cartesian_real_harmonics(v)

        column = 0
        for degree in (1, 2):
            for order in range(-degree, degree + 1):
                assert basis[:, column] == pytest.approx(
                    expected[(degree, order)]
                ), f"mismatch at l={degree}, m={order}"
                column += 1
        assert column == basis.shape[1]

    def test_averages_the_two_directions(self) -> None:
        a, b = self._directions(1), self._directions(2)
        theta_a, phi_a = calc_theta_phi(a)
        theta_b, phi_b = calc_theta_phi(b)
        mean = 0.5 * (
            sph_harm_basis(theta_a, phi_a, theta_a, phi_a, lmax=3)
            + sph_harm_basis(theta_b, phi_b, theta_b, phi_b, lmax=3)
        )
        got = sph_harm_basis(theta_a, phi_a, theta_b, phi_b, lmax=3)
        assert got == pytest.approx(mean)

    @pytest.mark.parametrize(
        ("lmax", "expected"), [(1, 3), (2, 8), (4, 24), (6, 48)]
    )
    def test_parameter_count(self, lmax: int, expected: int) -> None:
        assert n_abs_params(lmax) == expected
        v = self._directions()
        theta, phi = calc_theta_phi(v)
        assert (
            sph_harm_basis(theta, phi, theta, phi, lmax).shape[1] == expected
        )

    def test_truncated_sqrt2_is_negligible(self) -> None:
        v = self._directions()
        theta, phi = calc_theta_phi(v)
        exact = sph_harm_basis(theta, phi, theta, phi, lmax=4)
        truncated = sph_harm_basis(
            theta, phi, theta, phi, lmax=4, sqrt2=SQRT2_DIALS
        )
        relative = np.abs(truncated - exact) / np.maximum(np.abs(exact), 1e-12)
        assert relative.max() < 1e-9

    def test_theta_phi_conventions(self) -> None:
        v = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        theta, phi = calc_theta_phi(v)
        assert theta == pytest.approx([0.0, np.pi / 2, np.pi / 2])
        assert phi == pytest.approx([0.0, 0.0, 1.5 * np.pi])

    def test_align_axis_puts_the_axis_on_z(self) -> None:
        v = np.array([list(ALIGNMENT_AXIS)], dtype=np.float64)
        assert align_axis_along_z(ALIGNMENT_AXIS, v)[0] == pytest.approx(
            [0.0, 0.0, 1.0], abs=1e-12
        )

    def test_align_axis_is_a_rotation(self) -> None:
        v = self._directions(3)
        rotated = align_axis_along_z(ALIGNMENT_AXIS, v)
        assert np.linalg.norm(rotated, axis=1) == pytest.approx(
            np.linalg.norm(v, axis=1)
        )


def _toy_experiment() -> Experiment:
    """A minimal scaled experiment with a hand-made physical model."""
    rng = np.random.default_rng(7)
    return Experiment(
        cell=(79.0, 79.0, 38.0, 90.0, 90.0, 90.0),
        space_group_hall=" P 4nw 2abw",
        wavelength=1.0,
        s0=np.array([0.0, 0.0, 1.0]),
        rotation_axis=np.array([1.0, 0.0, 0.0]),
        fixed_rotation=np.eye(3),
        setting_rotation=np.eye(3),
        image_range=(1, 100),
        osc_range=(0.0, 50.0),
        scaling_model=ScalingModelRef(
            model_id="physical",
            components={
                "scale": 1.0 + 0.05 * rng.normal(size=10),
                "decay": 0.5 * rng.normal(size=8),
                "absorption": 0.02 * rng.normal(size=24),
            },
            config={
                "lmax": 4,
                "s_norm_fac": 0.08,
                "d_norm_fac": 0.06,
                "abs_surface_weight": 5e5,
            },
        ),
    )


def _toy_reflections(n: int = 256) -> Reflections:
    """Reflections with the columns the scaling model reads."""
    rng = np.random.default_rng(11)
    s1 = rng.normal(size=(n, 3))
    s1 /= np.linalg.norm(s1, axis=1)[:, None]

    def t(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a)).to(DTYPE)

    return Reflections(
        hkl=torch.zeros((n, 3), dtype=torch.int64),
        intensity=t(rng.uniform(1.0, 100.0, n)),
        variance=t(rng.uniform(1.0, 10.0, n)),
        d=t(rng.uniform(2.0, 20.0, n)),
        phi=t(rng.uniform(0.0, 99.0, n)),
        s1=t(s1),
        partiality=torch.ones(n, dtype=DTYPE),
        lp=torch.ones(n, dtype=DTYPE),
        qe=torch.ones(n, dtype=DTYPE),
        dataset_id=torch.zeros(n, dtype=torch.int64),
        scale=torch.ones(n, dtype=DTYPE),
    )


class TestPhysicalScalingModel:
    """Assembly, factorisation, gradients and restraints."""

    @staticmethod
    def _built() -> tuple[PhysicalScalingModel, ScaleFeatures]:
        expt = _toy_experiment()
        return (
            PhysicalScalingModel.from_dials(expt),
            build_features(_toy_reflections(), expt),
        )

    def test_from_dials_reads_every_component(self) -> None:
        model, _ = self._built()
        assert model.scale.numel() == 10
        assert model.decay.numel() == 8
        assert model.absorption.numel() == 24
        assert model.abs_surface_weight == pytest.approx(5e5)

    def test_parameters_are_registered(self) -> None:
        model, _ = self._built()
        names = {name for name, _ in model.named_parameters()}
        assert names == {"scale", "decay", "absorption"}

    def test_forward_is_the_product_of_components(self) -> None:
        model, feats = self._built()
        with torch.no_grad():
            product = (
                model.scale_component(feats)
                * model.decay_component(feats)
                * model.absorption_component(feats)
            )
            assert model(feats).numpy() == pytest.approx(product.numpy())

    def test_forward_shape_and_finiteness(self) -> None:
        model, feats = self._built()
        scales = model(feats)
        assert scales.shape == (len(feats),)
        assert torch.isfinite(scales).all()
        assert (scales > 0).all()

    def test_decay_uses_the_inverse_square_resolution(self) -> None:
        model, feats = self._built()
        with torch.no_grad():
            b = feats.decay.apply(model.decay)
            expected = torch.exp(b / (2.0 * feats.d**2))
            assert model.decay_component(feats).numpy() == pytest.approx(
                expected.numpy()
            )

    def test_absorption_is_unity_without_coefficients(self) -> None:
        model, feats = self._built()
        with torch.no_grad():
            model.absorption.zero_()
            assert model.absorption_component(feats).numpy() == pytest.approx(
                1.0
            )

    def test_gradients_reach_every_parameter(self) -> None:
        model, feats = self._built()
        model(feats).sum().backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert (parameter.grad != 0).any(), name

    def test_restraint_is_the_weighted_l2(self) -> None:
        model, _ = self._built()
        with torch.no_grad():
            expected = 5e5 * float((model.absorption**2).sum()) + 1e-1 * float(
                (model.decay**2).sum()
            )
            assert float(model.restraint()) == pytest.approx(expected)

    def test_restraint_is_differentiable(self) -> None:
        model, _ = self._built()
        model.restraint().backward()
        assert (
            model.scale.grad is None
            or float(model.scale.grad.abs().sum()) == 0.0
        )
        assert (model.absorption.grad != 0).any()

    def test_scan_angles_are_affine_in_the_image_index(self) -> None:
        expt = _toy_experiment()
        angles = scan_angles(expt, np.array([0.0, 1.0, 50.0]))
        assert np.rad2deg(angles) == pytest.approx([0.0, 0.5, 25.0])


@pytest.mark.skipif(
    not (_EXPT.exists() and _CACHE.exists()),
    reason="needs the scaled HEWL dataset",
)
class TestAgainstDials:
    """Forward evaluation against DIALS' own `inverse_scale_factor`."""

    @staticmethod
    def _relative_difference(points_per_degree: int | None) -> np.ndarray:
        from integrator.dials_port.io import load_experiment

        expt = load_experiment(_EXPT)
        refl = load_npz(_CACHE)
        model = PhysicalScalingModel.from_dials(expt)
        feats = build_features(
            refl,
            expt,
            lmax=int(expt.scaling_model.config["lmax"]),
            points_per_degree=points_per_degree,
            sqrt2=SQRT2_DIALS,
        )
        with torch.no_grad():
            scales = model(feats)
        return (torch.abs(scales - refl.scale) / refl.scale).numpy()

    def test_reproduces_the_dials_column(self) -> None:
        relative = self._relative_difference(points_per_degree=2)
        assert relative.max() < 1e-6, f"max relative {relative.max():.3e}"

    def test_exact_harmonics_stay_close(self) -> None:
        # Exact evaluation is the model we want; DIALS' 0.5 degree lookup
        # grid is what it actually wrote, so the two differ at 1e-4.
        relative = self._relative_difference(points_per_degree=None)
        assert relative.max() < 1e-2
        assert np.median(relative) < 1e-3
