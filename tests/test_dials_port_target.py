"""Fast unit tests for the `dials_port` target, error model and outliers.

Everything here runs on small synthetic data and needs neither DIALS nor
the HEWL reference. The quantitative comparison against DIALS lives in
`scripts/verify_dials_port_target.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from integrator.dials_port.error_model import (
    BasicErrorModel,
    ErrorModelBinner,
    ErrorModelFilter,
    central_window,
    deviation_prefactor,
    normalised_deviations,
    order_statistics,
    prepare_error_model_data,
    refine_error_model,
    variance_penalty,
)
from integrator.dials_port.outliers import (
    group_median,
    leave_one_out_z,
    limit_outlier_weights,
    reject_outliers,
)
from integrator.dials_port.refine import refine
from integrator.dials_port.symmetry import build_asu_map
from integrator.dials_port.target import (
    GammaLikelihood,
    GaussianLeastSquares,
    PoissonLikelihood,
    ScalingTarget,
    group_multiplicity,
    merged_intensity,
    resolve_groups,
)
from integrator.dials_port.types import DTYPE, IDTYPE, Reflections


def make_reflections(
    n_groups: int = 20,
    per_group: int = 6,
    seed: int = 0,
    noise: float = 0.06,
    var_factor: float = 0.4,
) -> Reflections:
    """Synthetic observations with a known scale and merged intensity."""
    rng = np.random.default_rng(seed)
    n = n_groups * per_group
    group = np.repeat(np.arange(n_groups), per_group)
    ih = rng.uniform(60.0, 900.0, n_groups)[group]
    scale = rng.uniform(0.7, 1.5, n)
    mean = ih * scale
    intensity = mean * (1.0 + noise * rng.standard_normal(n))
    return Reflections(
        hkl=torch.from_numpy(
            np.stack([group, group % 5, group // 5], axis=1)
        ).to(IDTYPE),
        intensity=torch.from_numpy(intensity).to(DTYPE),
        variance=torch.from_numpy(np.maximum(mean, 1.0) * var_factor).to(
            DTYPE
        ),
        d=torch.full((n,), 2.0, dtype=DTYPE),
        phi=torch.from_numpy(rng.uniform(0.0, 50.0, n)).to(DTYPE),
        s1=torch.zeros(n, 3, dtype=DTYPE),
        partiality=torch.ones(n, dtype=DTYPE),
        lp=torch.ones(n, dtype=DTYPE),
        qe=torch.ones(n, dtype=DTYPE),
        dataset_id=torch.zeros(n, dtype=IDTYPE),
        scale=torch.from_numpy(scale).to(DTYPE),
    )


@pytest.fixture
def refl() -> Reflections:
    """Default synthetic observations."""
    return make_reflections()


@pytest.fixture
def amap(refl: Reflections):
    """Symmetry map in P1, so groups follow the constructed `hkl`."""
    return build_asu_map(refl.hkl, "P 1")


class TestMergedIntensity:
    """`merged_intensity`."""

    def test_matches_numpy_weighted_mean(self, refl: Reflections) -> None:
        group = refl.hkl[:, 0]
        n_groups = int(group.max()) + 1
        w = 1.0 / refl.variance
        ih, ih_refl = merged_intensity(
            refl.intensity, w, refl.scale, group, n_groups
        )
        gi = group.numpy()
        num = np.bincount(
            gi,
            weights=(w * refl.scale * refl.intensity).numpy(),
            minlength=n_groups,
        )
        den = np.bincount(
            gi,
            weights=(w * refl.scale**2).numpy(),
            minlength=n_groups,
        )
        assert np.allclose(ih.numpy(), num / den, rtol=0, atol=1e-12)
        assert torch.equal(ih_refl, ih[group])

    def test_exact_for_noiseless_data(self) -> None:
        clean = make_reflections(noise=0.0)
        group = clean.hkl[:, 0]
        ih, _ = merged_intensity(
            clean.intensity,
            1.0 / clean.variance,
            clean.scale,
            group,
            int(group.max()) + 1,
        )
        truth = clean.intensity / clean.scale
        assert torch.allclose(ih[group], truth, rtol=1e-12)

    def test_empty_group_is_zero(self) -> None:
        intensity = torch.tensor([1.0, 2.0], dtype=DTYPE)
        w = torch.ones(2, dtype=DTYPE)
        g = torch.ones(2, dtype=DTYPE)
        group = torch.tensor([0, 2], dtype=IDTYPE)
        ih, _ = merged_intensity(intensity, w, g, group, 3)
        assert float(ih[1]) == 0.0

    def test_group_multiplicity(self, refl: Reflections) -> None:
        group = refl.hkl[:, 0]
        n_h = group_multiplicity(group, int(group.max()) + 1)
        assert torch.all(n_h == 6.0)


class TestResolveGroups:
    """`resolve_groups` and its centric handling."""

    def test_laue_matches_asu_map(self, amap) -> None:
        gid, n = resolve_groups(amap, anomalous=False)
        assert torch.equal(gid, amap.group_id_laue)
        assert n == amap.n_groups_laue

    def test_pooling_is_idempotent(self, refl, amap) -> None:
        once, n_once = resolve_groups(amap, True, pool_centrics=True)
        pooled = build_asu_map(refl.hkl, "P 1")
        pooled.group_id, pooled.n_groups = once, n_once
        twice, n_twice = resolve_groups(pooled, True, pool_centrics=True)
        assert n_twice == n_once
        assert torch.equal(twice, once)

    def test_pooling_never_increases_group_count(self, amap) -> None:
        _, pooled = resolve_groups(amap, True, pool_centrics=True)
        _, split = resolve_groups(amap, True, pool_centrics=False)
        assert pooled <= split


class TestLikelihoods:
    """The three likelihoods and their weighting."""

    @pytest.mark.parametrize(
        "lik",
        [
            GaussianLeastSquares("observed"),
            GaussianLeastSquares("expected"),
            GaussianLeastSquares("unity"),
            GammaLikelihood(),
            PoissonLikelihood(),
        ],
    )
    def test_gradient_is_finite_and_nonzero(self, refl, amap, lik) -> None:
        raw = torch.log(refl.scale.clone() * 1.3).requires_grad_(True)
        ScalingTarget(lik)(refl, amap, torch.exp(raw)).backward()
        assert torch.isfinite(raw.grad).all()
        assert float(raw.grad.abs().max()) > 0.0

    def test_gaussian_matches_hand_formula(self, refl, amap) -> None:
        lik = GaussianLeastSquares("observed")
        target = ScalingTarget(lik)
        ih, _ = target.merge(refl, amap, refl.scale)
        expected = (
            (refl.intensity - refl.scale * ih) ** 2 / refl.variance
        ).sum()
        got = target(refl, amap, refl.scale)
        assert torch.allclose(got, expected, rtol=1e-12)

    def test_unknown_weight_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown weight_mode"):
            GaussianLeastSquares("magic")  # type: ignore[arg-type]

    def test_expected_weights_use_the_model_mean(self, refl, amap) -> None:
        lik = GaussianLeastSquares("expected", variance_power=2.0)
        ih, w = ScalingTarget(lik).merge(refl, amap, refl.scale)
        mu = refl.scale * ih
        expected = 1.0 / (
            refl.variance * (mu / refl.intensity.abs()) ** 2
        )
        assert torch.allclose(w, expected, rtol=1e-10)

    def test_variance_power_zero_is_observed(self, refl, amap) -> None:
        a = ScalingTarget(
            GaussianLeastSquares("expected", variance_power=0.0)
        ).merge(refl, amap, refl.scale)[1]
        b = ScalingTarget(GaussianLeastSquares("observed")).merge(
            refl, amap, refl.scale
        )[1]
        assert torch.allclose(a, b, rtol=1e-12)

    def test_gamma_minimised_at_the_observation(self) -> None:
        lik = GammaLikelihood()
        one = make_reflections(n_groups=1, per_group=1, noise=0.0)
        best = lik(one, torch.ones(1, dtype=DTYPE), one.intensity).sum()
        for factor in (0.5, 0.8, 1.2, 2.0):
            worse = lik(
                one, torch.ones(1, dtype=DTYPE), one.intensity * factor
            ).sum()
            assert float(worse) > float(best)

    def test_gamma_shape_is_i_squared_over_var(self, refl) -> None:
        k = GammaLikelihood().shape(refl)
        assert torch.allclose(
            k, refl.intensity**2 / refl.variance, rtol=1e-10
        )

    def test_poisson_minimised_at_the_observation(self) -> None:
        lik = PoissonLikelihood()
        one = make_reflections(n_groups=1, per_group=1, noise=0.0)
        best = lik(one, torch.ones(1, dtype=DTYPE), one.intensity).sum()
        for factor in (0.5, 1.5):
            worse = lik(
                one, torch.ones(1, dtype=DTYPE), one.intensity * factor
            ).sum()
            assert float(worse) > float(best)


class TestScalingTarget:
    """Gradients and the `Ih` profiling."""

    @pytest.mark.parametrize("detach", [False, True])
    @pytest.mark.parametrize(
        "lik",
        [
            GaussianLeastSquares("observed"),
            GammaLikelihood(),
            PoissonLikelihood(),
        ],
    )
    def test_finite_difference(self, refl, amap, lik, detach) -> None:
        target = ScalingTarget(lik, detach_ih=detach)
        group_id, n_groups = resolve_groups(amap, False)
        raw = torch.log(refl.scale.clone()).requires_grad_(True)
        target(refl, amap, torch.exp(raw)).backward()
        analytic = raw.grad.detach().clone()

        with torch.no_grad():
            ih0, w0 = target.merge(refl, amap, refl.scale)

        def objective(g: torch.Tensor) -> torch.Tensor:
            if detach:
                ih = ih0
            else:
                _, ih = merged_intensity(
                    refl.intensity, w0, g, group_id, n_groups
                )
            if isinstance(lik, GaussianLeastSquares):
                return (w0 * (refl.intensity - g * ih) ** 2).sum()
            return lik(refl, g, ih).sum()

        eps = 1e-6
        for i in (0, 13, 47, 91):
            pert = raw.detach().clone()
            pert[i] += eps
            up = float(objective(torch.exp(pert)))
            pert[i] -= 2 * eps
            dn = float(objective(torch.exp(pert)))
            numeric = (up - dn) / (2 * eps)
            # Central differences at eps=1e-6 in float64 are good to a
            # few parts in 1e5 on this data.
            assert abs(analytic[i] - numeric) <= 1e-4 * max(
                abs(numeric), 1.0
            )

    def test_envelope_theorem_for_gaussian(self, refl, amap) -> None:
        grads = []
        for detach in (False, True):
            raw = torch.log(refl.scale.clone()).requires_grad_(True)
            ScalingTarget(
                GaussianLeastSquares("observed"), detach_ih=detach
            )(refl, amap, torch.exp(raw)).backward()
            grads.append(raw.grad.detach().clone())
        rel = float((grads[0] - grads[1]).norm() / grads[1].norm())
        assert rel < 1e-10

    def test_dih_term_is_real_for_observed_gamma(self, refl, amap) -> None:
        grads = []
        for detach in (False, True):
            raw = torch.log(refl.scale.clone()).requires_grad_(True)
            ScalingTarget(
                GammaLikelihood("observed"), detach_ih=detach
            )(refl, amap, torch.exp(raw)).backward()
            grads.append(raw.grad.detach().clone())
        rel = float((grads[0] - grads[1]).norm() / grads[1].norm())
        assert rel > 1e-3

    def test_accepts_a_callable_scale(self, refl, amap) -> None:
        target = ScalingTarget(GaussianLeastSquares())
        direct = target(refl, amap, refl.scale)
        lazy = target(refl, amap, lambda: refl.scale)
        assert torch.allclose(direct, lazy, rtol=1e-14)

    def test_rmsd_is_root_mean_functional(self, refl, amap) -> None:
        target = ScalingTarget(GaussianLeastSquares())
        loss = target(refl, amap, refl.scale)
        rmsd = target.rmsd(refl, amap, refl.scale)
        assert torch.allclose(rmsd**2 * len(refl), loss, rtol=1e-12)

    def test_true_scale_beats_a_perturbed_one(self, refl, amap) -> None:
        target = ScalingTarget(GaussianLeastSquares())
        good = float(target(refl, amap, refl.scale))
        bad = float(target(refl, amap, refl.scale * 1.4))
        assert bad > good


class TestErrorModel:
    """The two-parameter basic error model."""

    def test_apply_matches_the_formula(self) -> None:
        model = BasicErrorModel(a=1.3, b=0.05)
        var = torch.tensor([4.0, 9.0], dtype=DTYPE)
        intensity = torch.tensor([100.0, 200.0], dtype=DTYPE)
        expected = 1.3**2 * (var + (0.05 * intensity) ** 2)
        assert torch.allclose(model.apply(var, intensity), expected)

    def test_identity_model_leaves_variance_alone(self) -> None:
        var = torch.tensor([4.0, 9.0], dtype=DTYPE)
        intensity = torch.tensor([100.0, 200.0], dtype=DTYPE)
        got = BasicErrorModel(a=1.0, b=0.0).apply(var, intensity)
        assert torch.allclose(got, var)

    def test_i_sigma_limit(self) -> None:
        assert BasicErrorModel(2.0, 0.1).i_sigma_limit == pytest.approx(5.0)
        assert BasicErrorModel(1.0, 0.0).i_sigma_limit == float("inf")

    def test_sigma_prime_divides_by_scale(self) -> None:
        model = BasicErrorModel(1.0, 0.0)
        var = torch.tensor([4.0], dtype=DTYPE)
        intensity = torch.tensor([10.0], dtype=DTYPE)
        scale = torch.tensor([2.0], dtype=DTYPE)
        assert float(model.sigma_prime(var, intensity, scale)) == 1.0

    def test_deviation_prefactor(self) -> None:
        intensity = torch.tensor([12.0], dtype=DTYPE)
        variance = torch.tensor([1.0], dtype=DTYPE)
        scale = torch.tensor([1.0], dtype=DTYPE)
        ih = torch.tensor([10.0], dtype=DTYPE)
        n_h = torch.tensor([4.0], dtype=DTYPE)
        got = normalised_deviations(
            intensity, variance, scale, ih, n_h, BasicErrorModel(1.0, 0.0)
        )
        assert float(got) == pytest.approx(2.0 * np.sqrt(3.0 / 4.0))

    def test_order_statistics_are_symmetric(self) -> None:
        x = order_statistics(101)
        assert x[50] == pytest.approx(0.0, abs=1e-12)
        assert np.allclose(x, -x[::-1], atol=1e-12)

    def test_central_window_keeps_the_middle(self) -> None:
        rng = np.random.default_rng(0)
        delta = rng.standard_normal(20000)
        x, y = central_window(delta, cutoff=1.5)
        assert x.size == y.size
        # Phi(1.5) - Phi(-1.5) = 0.8664
        assert 0.86 < x.size / delta.size < 0.874
        assert np.all(np.diff(y) >= 0)
        assert abs(x.min() + 1.5) < 0.02 and abs(x.max() - 1.5) < 0.02

    def test_central_window_slope_recovers_a_scaling(self) -> None:
        rng = np.random.default_rng(1)
        for true_a in (0.7, 1.0, 1.9):
            x, y = central_window(true_a * rng.standard_normal(200000))
            slope = np.polyfit(x, y, 1)[0]
            assert slope == pytest.approx(true_a, rel=0.02)

    def test_variance_penalty_minimised_at_one(self) -> None:
        assert variance_penalty(np.array([1.0]))[0] == pytest.approx(0.0)
        grid = np.array([0.4, 0.7, 1.3, 2.0])
        assert np.all(variance_penalty(grid) > 0.0)

    def test_binner_covers_the_data(self, refl, amap) -> None:
        data = prepare_error_model_data(
            refl,
            amap,
            refl.scale,
            ErrorModelFilter(min_ih=1.0, min_reflections=10),
        )
        binner = ErrorModelBinner(data, n_bins=4)
        assert binner.info.refl_per_bin.sum() <= len(data)
        assert np.all(np.diff(binner.info.boundaries) < 0)
        assert binner.info.initial_variances.size == binner.n_kept

    def test_filters_reject_too_few_reflections(self, refl, amap) -> None:
        with pytest.raises(ValueError, match="Insufficient reflections"):
            prepare_error_model_data(
                refl,
                amap,
                refl.scale,
                ErrorModelFilter(min_ih=1e12, min_reflections=10),
            )

    @staticmethod
    def _planted(a_true: float, per_group: int = 8, seed: int = 7):
        """Data whose true errors are `a_true` times those reported."""
        rng = np.random.default_rng(seed)
        n_groups = 1200
        n = n_groups * per_group
        group = np.repeat(np.arange(n_groups), per_group)
        ih = rng.uniform(300.0, 3000.0, n_groups)[group]
        intensity = ih + a_true * np.sqrt(ih) * rng.standard_normal(n)
        refl = Reflections(
            hkl=torch.from_numpy(
                np.stack([group, group % 13, group // 13], axis=1)
            ).to(IDTYPE),
            intensity=torch.from_numpy(intensity).to(DTYPE),
            variance=torch.from_numpy(ih).to(DTYPE),
            d=torch.full((n,), 2.0, dtype=DTYPE),
            phi=torch.zeros(n, dtype=DTYPE),
            s1=torch.zeros(n, 3, dtype=DTYPE),
            partiality=torch.ones(n, dtype=DTYPE),
            lp=torch.ones(n, dtype=DTYPE),
            qe=torch.ones(n, dtype=DTYPE),
            dataset_id=torch.zeros(n, dtype=IDTYPE),
            scale=torch.ones(n, dtype=DTYPE),
        )
        return refl, build_asu_map(refl.hkl, "P 1")

    @pytest.mark.parametrize("a_true", [0.8, 1.5, 2.0])
    def test_unbiased_prefactor_recovers_planted_a(self, a_true) -> None:
        refl, amap = self._planted(a_true)
        result = refine_error_model(
            refl,
            amap,
            refl.scale,
            filters=ErrorModelFilter(min_reflections=100),
            a_tolerance=1e-4,
            max_cycles=300,
            prefactor="unbiased",
        )
        assert result.model.a == pytest.approx(a_true, rel=0.03)
        assert result.model.b < 0.01
        assert not result.failed

    @pytest.mark.parametrize("per_group", [4, 8])
    def test_dials_prefactor_biases_a_by_n_minus_one_over_n(
        self, per_group
    ) -> None:
        """The shipped prefactor scales `a` by exactly `(n-1)/n`."""
        refl, amap = self._planted(1.5, per_group=per_group)
        fits = {}
        for mode in ("dials", "unbiased"):
            fits[mode] = refine_error_model(
                refl,
                amap,
                refl.scale,
                filters=ErrorModelFilter(min_reflections=100),
                a_tolerance=1e-4,
                max_cycles=300,
                prefactor=mode,
            ).model.a
        ratio = fits["dials"] / fits["unbiased"]
        assert ratio == pytest.approx(
            (per_group - 1) / per_group, rel=1e-3
        )

    def test_unknown_prefactor_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown prefactor"):
            deviation_prefactor(
                torch.tensor([4.0], dtype=DTYPE),
                "magic",  # type: ignore[arg-type]
            )

    def test_history_is_recorded(self, refl, amap) -> None:
        result = refine_error_model(
            refl,
            amap,
            refl.scale,
            filters=ErrorModelFilter(min_ih=1.0, min_reflections=10),
            max_cycles=5,
        )
        # A converged run takes one extra `a` step after the loop.
        assert len(result.a_history) - len(result.b_history) in (0, 1)
        assert result.n_refl > 0
        assert (
            result.binning.bin_variances.size
            == result.binning.mean_intensities.size
        )


class TestOutliers:
    """Leave-one-out normalised deviation rejection."""

    def test_group_median_matches_numpy(self) -> None:
        rng = np.random.default_rng(3)
        values = torch.from_numpy(rng.standard_normal(60)).to(DTYPE)
        group = torch.from_numpy(np.repeat(np.arange(6), 10)).to(IDTYPE)
        got = group_median(values, group, 6).numpy()
        want = [
            np.median(values.numpy()[group.numpy() == g]) for g in range(6)
        ]
        assert np.allclose(got, want)

    def test_weight_cap(self) -> None:
        weights = torch.tensor([1.0, 1.0, 1.0, 500.0], dtype=DTYPE)
        group = torch.zeros(4, dtype=IDTYPE)
        capped = limit_outlier_weights(weights, group, 1, factor=10.0)
        assert float(capped[3]) == pytest.approx(10.0)
        assert torch.allclose(capped[:3], weights[:3])

    def test_leave_one_out_excludes_the_reflection(self) -> None:
        intensity = torch.tensor([10.0, 10.0, 10.0, 100.0], dtype=DTYPE)
        w = torch.ones(4, dtype=DTYPE)
        g = torch.ones(4, dtype=DTYPE)
        group = torch.zeros(4, dtype=IDTYPE)
        z, n_h = leave_one_out_z(intensity, w, g, group, 1)
        assert torch.all(n_h == 4.0)
        # The outlier is measured against 10, not against the mean of 32.5.
        assert float(z[3]) > float(z[0])
        assert float(z[3]) == pytest.approx(
            90.0 / np.sqrt(1.0 + 1.0 / 3.0), rel=1e-9
        )

    def test_finds_planted_outliers(self) -> None:
        refl = make_reflections(n_groups=25, per_group=8, seed=11)
        planted = [3, 19, 42, 100, 175]
        intensity = refl.intensity.clone()
        intensity[planted] *= 12.0
        refl = Reflections(
            **{
                **{f: getattr(refl, f) for f in refl.__dataclass_fields__},
                "intensity": intensity,
            }
        )
        this_map = build_asu_map(refl.hkl, "P 1")
        mask = reject_outliers(refl, this_map, scale=refl.scale, zmax=6.0)
        assert set(planted) <= set(torch.nonzero(mask).flatten().tolist())

    def test_clean_data_flags_nothing(self) -> None:
        clean = make_reflections(noise=0.0)
        clean_map = build_asu_map(clean.hkl, "P 1")
        mask = reject_outliers(clean, clean_map, scale=clean.scale)
        assert int(mask.sum()) == 0

    def test_small_groups_are_never_tested(self) -> None:
        refl = make_reflections(n_groups=10, per_group=2, seed=2)
        amap = build_asu_map(refl.hkl, "P 1")
        mask = reject_outliers(refl, amap, scale=refl.scale, zmax=0.01)
        assert int(mask.sum()) == 0

    def test_one_sided_ignores_the_lower_tail(self) -> None:
        """A dim outlier is caught two-sided and missed one-sided."""
        refl = make_reflections(n_groups=25, per_group=8, seed=13)
        intensity = refl.intensity.clone()
        bright, dim = 10, 50
        intensity[bright] *= 10.0
        intensity[dim] *= 0.1
        refl = Reflections(
            **{
                **{f: getattr(refl, f) for f in refl.__dataclass_fields__},
                "intensity": intensity,
            }
        )
        amap = build_asu_map(refl.hkl, "P 1")
        both = reject_outliers(refl, amap, scale=refl.scale, zmax=6.0)
        upper = reject_outliers(
            refl,
            amap,
            scale=refl.scale,
            zmax=6.0,
            zmax_lower=float("inf"),
        )
        assert bool(both[bright]) and bool(both[dim])
        assert bool(upper[bright]) and not bool(upper[dim])

    def test_single_round_thresholds_nest(self) -> None:
        """Per round the criteria nest, even though the runs need not.

        Only the largest score per group is flagged each round, so
        removing a low outlier under the two-sided test changes what the
        next round sees. The flagged *sets* of a one-sided and a
        two-sided run are therefore not nested even though the
        per-round criteria are.
        """
        refl = make_reflections(n_groups=25, per_group=8, seed=17)
        amap = build_asu_map(refl.hkl, "P 1")
        group_id, n_groups = resolve_groups(amap, False)
        z, n_h = leave_one_out_z(
            refl.intensity,
            1.0 / refl.variance,
            refl.scale,
            group_id,
            n_groups,
        )
        testable = n_h > 2
        both = (z.abs() > 2.5) & testable
        upper = (z > 2.5) & testable
        mid = ((z > 2.5) | (z < -4.0)) & testable
        assert torch.all(upper <= mid)
        assert torch.all(mid <= both)

    def test_terminates_on_pathological_data(self) -> None:
        n = 60
        intensity = torch.ones(n, dtype=DTYPE)
        refl = Reflections(
            hkl=torch.zeros(n, 3, dtype=IDTYPE),
            intensity=intensity,
            variance=torch.full((n,), 1e-12, dtype=DTYPE),
            d=torch.ones(n, dtype=DTYPE),
            phi=torch.zeros(n, dtype=DTYPE),
            s1=torch.zeros(n, 3, dtype=DTYPE),
            partiality=torch.ones(n, dtype=DTYPE),
            lp=torch.ones(n, dtype=DTYPE),
            qe=torch.ones(n, dtype=DTYPE),
            dataset_id=torch.zeros(n, dtype=IDTYPE),
            scale=torch.ones(n, dtype=DTYPE),
        )
        amap = build_asu_map(refl.hkl, "P 1")
        mask = reject_outliers(refl, amap, zmax=1e-6, max_rounds=1000)
        assert int(mask.sum()) < n


class ImageScale(torch.nn.Module):
    """A per-image multiplicative scale, keyed on `phi`.

    The first image is pinned at `1`, breaking the `g -> c g` gauge
    freedom of the scaling target the same way DIALS'
    `fix_initial_parameter` does.
    """

    def __init__(self, n_images: int) -> None:
        super().__init__()
        self.log_g = torch.nn.Parameter(
            torch.zeros(n_images - 1, dtype=DTYPE)
        )

    def forward(self, refl: Reflections) -> torch.Tensor:
        """Return `(N,)` inverse scale factors."""
        log_g = torch.cat(
            [torch.zeros(1, dtype=DTYPE), self.log_g]
        )
        return torch.exp(log_g[refl.phi.long()])


class TestRefine:
    """The outer loop."""

    @staticmethod
    def _dataset(seed: int = 0, n_images: int = 12):
        rng = np.random.default_rng(seed)
        n_groups, per_group = 80, 10
        n = n_groups * per_group
        group = np.repeat(np.arange(n_groups), per_group)
        image = rng.integers(0, n_images, n)
        true_scale = np.exp(0.35 * np.sin(np.arange(n_images) / 2.5))[image]
        ih = rng.uniform(80.0, 1200.0, n_groups)[group]
        mean = ih * true_scale
        intensity = mean * (1.0 + 0.05 * rng.standard_normal(n))
        refl = Reflections(
            hkl=torch.from_numpy(
                np.stack([group, group % 7, group // 7], axis=1)
            ).to(IDTYPE),
            intensity=torch.from_numpy(intensity).to(DTYPE),
            variance=torch.from_numpy(np.maximum(mean, 1.0) * 0.3).to(DTYPE),
            d=torch.full((n,), 2.0, dtype=DTYPE),
            phi=torch.from_numpy(image.astype(np.float64)).to(DTYPE),
            s1=torch.zeros(n, 3, dtype=DTYPE),
            partiality=torch.ones(n, dtype=DTYPE),
            lp=torch.ones(n, dtype=DTYPE),
            qe=torch.ones(n, dtype=DTYPE),
            dataset_id=torch.zeros(n, dtype=IDTYPE),
            scale=torch.ones(n, dtype=DTYPE),
        )
        return refl, build_asu_map(refl.hkl, "P 1"), true_scale

    def test_recovers_a_per_image_scale(self) -> None:
        refl, amap, true_scale = self._dataset()
        result = refine(
            refl,
            amap,
            ImageScale(12),
            ScalingTarget(GaussianLeastSquares()),
            n_cycles=2,
            error_model=False,
            outlier_rejection=False,
        )
        got = result.scale.numpy()
        got = got / got.mean()
        want = true_scale / true_scale.mean()
        assert np.corrcoef(got, want)[0, 1] > 0.99
        assert np.abs(got - want).max() < 0.05

    def test_rmsd_decreases_within_a_cycle(self) -> None:
        refl, amap, _ = self._dataset(seed=4)
        result = refine(
            refl,
            amap,
            ImageScale(12),
            ScalingTarget(GaussianLeastSquares()),
            n_cycles=1,
            error_model=False,
            outlier_rejection=False,
        )
        steps = result.history[0].steps
        assert steps[-1].rmsd < steps[0].rmsd

    def test_history_is_populated(self) -> None:
        refl, amap, _ = self._dataset(seed=5)
        result = refine(
            refl,
            amap,
            ImageScale(12),
            ScalingTarget(GaussianLeastSquares()),
            n_cycles=2,
            error_model=False,
        )
        assert len(result.history) >= 1
        for entry in result.history:
            assert entry.n_refl > 0
            assert entry.steps
        assert np.isfinite(result.rmsd)

    def test_flags_planted_outliers(self) -> None:
        refl, amap, _ = self._dataset(seed=6)
        intensity = refl.intensity.clone()
        planted = [5, 300, 700]
        intensity[planted] *= 9.0
        refl = Reflections(
            **{
                **{f: getattr(refl, f) for f in refl.__dataclass_fields__},
                "intensity": intensity,
            }
        )
        result = refine(
            refl,
            amap,
            ImageScale(12),
            ScalingTarget(GaussianLeastSquares()),
            n_cycles=3,
            error_model=False,
            outlier_rejection=True,
        )
        assert bool(result.outliers[planted].all())

    def test_fixed_model_is_a_noop(self) -> None:
        refl, amap, _ = self._dataset(seed=8)
        model = ImageScale(12)
        for p in model.parameters():
            p.requires_grad_(False)
        result = refine(
            refl,
            amap,
            model,
            ScalingTarget(GaussianLeastSquares()),
            n_cycles=1,
            error_model=False,
            outlier_rejection=False,
        )
        assert torch.allclose(result.scale, torch.ones_like(result.scale))
        assert result.history[0].scale_converged
