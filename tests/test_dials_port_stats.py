"""Known-answer tests for `integrator.dials_port.stats`.

Every case here is small and analytic. Agreement with cctbx on a real
dataset is checked separately by `scripts/verify_dials_port_stats.py`,
which needs a DIALS runtime.
"""

from __future__ import annotations

import math

import pytest
import torch

from integrator.dials_port import stats
from integrator.dials_port.symmetry import build_asu_map
from integrator.dials_port.types import DTYPE, IDTYPE, MergedData, Reflections

CELL = (10.0, 10.0, 10.0, 90.0, 90.0, 90.0)


def make_reflections(
    hkl: torch.Tensor,
    intensity: torch.Tensor,
    sigma: torch.Tensor,
) -> Reflections:
    """Build a minimal `Reflections` for the columns stats touches.

    Args:
        hkl: `(N, 3)` Miller indices.
        intensity: `(N,)` intensities.
        sigma: `(N,)` standard deviations.

    Returns:
        Observations with unit scale, partiality, lp and qe.
    """
    n = hkl.shape[0]
    d = 1.0 / torch.sqrt(stats.d_star_sq(hkl, CELL))
    ones = torch.ones(n, dtype=DTYPE)
    return Reflections(
        hkl=hkl.to(IDTYPE),
        intensity=intensity.to(DTYPE),
        variance=(sigma.to(DTYPE) ** 2),
        d=d,
        phi=torch.zeros(n, dtype=DTYPE),
        s1=torch.zeros(n, 3, dtype=DTYPE),
        partiality=ones,
        lp=ones,
        qe=ones,
        dataset_id=torch.zeros(n, dtype=IDTYPE),
        scale=ones,
    )


def replicated(
    hkl: torch.Tensor,
    true_i: torch.Tensor,
    n_obs: int,
    noise: float,
    seed: int,
) -> Reflections:
    """Observe each reflection `n_obs` times with Gaussian noise.

    Args:
        hkl: `(M, 3)` unique Miller indices.
        true_i: `(M,)` true intensities.
        n_obs: observations per reflection.
        noise: standard deviation of the added noise.
        seed: RNG seed.

    Returns:
        `(M * n_obs)` observations.
    """
    generator = torch.Generator().manual_seed(seed)
    rep_hkl = hkl.repeat_interleave(n_obs, dim=0)
    rep_i = true_i.repeat_interleave(n_obs)
    if noise > 0:
        rep_i = rep_i + noise * torch.randn(
            rep_i.shape, generator=generator, dtype=DTYPE
        )
    sigma = torch.full_like(rep_i, max(noise, 1e-3))
    return make_reflections(rep_hkl, rep_i, sigma)


def unique_hkl(n: int) -> torch.Tensor:
    """Return `n` distinct Miller indices in P1, none symmetry related."""
    out = []
    h = k = ll = 0
    while len(out) < n:
        ll += 1
        if ll > 6:
            ll = 1
            k += 1
        if k > 6:
            k = 0
            h += 1
        out.append((h + 1, k + 1, ll))
    return torch.tensor(out[:n], dtype=IDTYPE)


class TestResolutionBins:
    def test_equal_volume_edges(self) -> None:
        """Edges are equally spaced in `d*^3`, i.e. in shell volume."""
        d = torch.linspace(10.0, 2.0, 500, dtype=DTYPE)
        bins = stats.resolution_bins(d, n_bins=8)
        r_cubed = bins.limits.sqrt() ** 3
        steps = r_cubed[1:] - r_cubed[:-1]
        assert torch.allclose(steps, steps[0], rtol=1e-12)

    def test_every_reflection_is_binned(self) -> None:
        """The tolerance-expanded outer edges leave nothing outside."""
        d = torch.linspace(10.0, 2.0, 500, dtype=DTYPE)
        bins = stats.resolution_bins(d, n_bins=20)
        assert int(bins.bin_id.min()) == 0
        assert int(bins.bin_id.max()) == 19
        assert bins.n_bins == 20

    def test_monotonic_in_resolution(self) -> None:
        """Higher `d*^2` never lands in a lower-index shell."""
        d = torch.linspace(10.0, 2.0, 500, dtype=DTYPE)
        bins = stats.resolution_bins(d, n_bins=10)
        order = torch.argsort(1.0 / d**2)
        assert bool((bins.bin_id[order].diff() >= 0).all())

    def test_labels_and_ranges(self) -> None:
        """Labels read low resolution first and match `d_range`."""
        d = torch.linspace(10.0, 2.0, 100, dtype=DTYPE)
        bins = stats.resolution_bins(d, n_bins=4)
        assert len(bins.labels) == 4
        first, last = bins.d_range(0), bins.d_range(3)
        assert first[0] > first[1] > last[0] > last[1]

    def test_rejects_bad_input(self) -> None:
        d = torch.linspace(10.0, 2.0, 10, dtype=DTYPE)
        with pytest.raises(ValueError):
            stats.resolution_bins(d, n_bins=0)
        with pytest.raises(ValueError):
            stats.resolution_bins(torch.empty(0, dtype=DTYPE))


class TestMergeReduction:
    def test_inverse_variance_weighted_mean(self) -> None:
        """Two observations collapse to the inverse-variance mean."""
        hkl = torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=IDTYPE)
        intensity = torch.tensor([10.0, 20.0], dtype=DTYPE)
        sigma = torch.tensor([1.0, 2.0], dtype=DTYPE)
        red = stats._merge_equivalents(
            intensity, sigma, torch.zeros(2, dtype=IDTYPE), 1
        )
        expected = (10.0 / 1.0 + 20.0 / 4.0) / (1.0 / 1.0 + 1.0 / 4.0)
        assert red.mean.item() == pytest.approx(expected)
        assert red.sigma.item() == pytest.approx(math.sqrt(1.0 / 1.25))
        assert int(red.mult.item()) == 2
        assert hkl.shape == (2, 3)

    def test_r_factors_of_a_known_pair(self) -> None:
        """R-merge, R-meas and R-pim on one two-fold group."""
        intensity = torch.tensor([90.0, 110.0], dtype=DTYPE)
        sigma = torch.ones(2, dtype=DTYPE)
        red = stats._merge_equivalents(
            intensity, sigma, torch.zeros(2, dtype=IDTYPE), 1
        )
        r_merge, r_meas, r_pim = stats._r_factors(red)
        # mean 100, sum|dev| = 20, sum I = 200
        assert r_merge == pytest.approx(0.1)
        assert r_meas == pytest.approx(0.1 * math.sqrt(2.0))
        assert r_pim == pytest.approx(0.1)

    def test_singletons_do_not_enter_r_factors(self) -> None:
        """A multiplicity-one group contributes to neither side."""
        intensity = torch.tensor([90.0, 110.0, 1000.0], dtype=DTYPE)
        sigma = torch.ones(3, dtype=DTYPE)
        group = torch.tensor([0, 0, 1], dtype=IDTYPE)
        red = stats._merge_equivalents(intensity, sigma, group, 2)
        assert stats._r_factors(red)[0] == pytest.approx(0.1)


class TestCCHalf:
    def test_perfect_data_gives_one(self) -> None:
        """Noiseless replicates correlate perfectly between halves."""
        hkl = unique_hkl(200)
        true_i = torch.linspace(10.0, 1000.0, 200, dtype=DTYPE)
        refl = replicated(hkl, true_i, n_obs=8, noise=0.0, seed=1)
        amap = build_asu_map(refl.hkl, "P 1")
        cc = stats.cc_half_random(refl, amap, seed=0)
        assert cc.value == pytest.approx(1.0, abs=1e-12)
        assert cc.n_refl == 200

    def test_pure_noise_gives_zero(self) -> None:
        """With no true signal the halves are uncorrelated."""
        hkl = unique_hkl(400)
        true_i = torch.zeros(400, dtype=DTYPE)
        refl = replicated(hkl, true_i, n_obs=8, noise=100.0, seed=2)
        amap = build_asu_map(refl.hkl, "P 1")
        cc = stats.cc_half_random(refl, amap, seed=0, n_repeats=8)
        assert abs(cc.value) < 0.1

    def test_repeats_shrink_the_spread(self) -> None:
        """Averaging splits reduces the seed-to-seed scatter."""
        hkl = unique_hkl(300)
        true_i = torch.linspace(10.0, 500.0, 300, dtype=DTYPE)
        refl = replicated(hkl, true_i, n_obs=6, noise=80.0, seed=3)
        amap = build_asu_map(refl.hkl, "P 1")
        one = torch.tensor(
            [stats.cc_half_random(refl, amap, seed=s).value for s in range(12)]
        )
        many = torch.tensor(
            [
                stats.cc_half_random(refl, amap, seed=s, n_repeats=8).value
                for s in range(12)
            ]
        )
        assert float(many.std()) < float(one.std())

    def test_single_repeat_has_no_std(self) -> None:
        hkl = unique_hkl(50)
        refl = replicated(
            hkl, torch.linspace(1.0, 50.0, 50, dtype=DTYPE), 4, 1.0, seed=4
        )
        amap = build_asu_map(refl.hkl, "P 1")
        assert math.isnan(stats.cc_half_random(refl, amap).std)

    def test_rejects_zero_repeats(self) -> None:
        hkl = unique_hkl(10)
        refl = replicated(hkl, torch.ones(10, dtype=DTYPE), 4, 1.0, seed=5)
        amap = build_asu_map(refl.hkl, "P 1")
        with pytest.raises(ValueError):
            stats.cc_half_random(refl, amap, n_repeats=0)


class TestCCHalfSigmaTau:
    def test_perfect_data_gives_one(self) -> None:
        """Zero internal variance drives the estimator to one.

        Not exactly one: the merged sigma is floored at the propagated
        `1 / sqrt(n)` of the unit input sigmas, which leaves a residual
        `var_e` of order `1 / n`.
        """
        hkl = unique_hkl(100)
        true_i = torch.linspace(10.0, 1000.0, 100, dtype=DTYPE)
        refl = replicated(hkl, true_i, n_obs=5, noise=0.0, seed=6)
        amap = build_asu_map(refl.hkl, "P 1")
        cc, n_refl = stats.cc_half_sigma_tau(stats.sigma_tau_merge(refl, amap))
        assert cc == pytest.approx(1.0, abs=1e-5)
        assert n_refl == 100

    def test_matches_its_own_definition(self) -> None:
        """Recompute `(var_y - var_e/2) / (var_y + var_e/2)` by hand."""
        hkl = unique_hkl(120)
        true_i = torch.linspace(10.0, 800.0, 120, dtype=DTYPE)
        refl = replicated(hkl, true_i, n_obs=7, noise=60.0, seed=7)
        amap = build_asu_map(refl.hkl, "P 1")
        merged = stats.sigma_tau_merge(refl, amap)
        sel = merged.multiplicity > 1
        i = merged.intensity[sel]
        var_y = float(((i - i.mean()) ** 2).sum() / (i.numel() - 1))
        var_e = 2.0 * float((merged.sigma[sel] ** 2).mean())
        expected = (var_y - 0.5 * var_e) / (var_y + 0.5 * var_e)
        assert stats.cc_half_sigma_tau(merged)[0] == pytest.approx(expected)

    def test_is_deterministic(self) -> None:
        """Unlike the random split, repeated calls agree exactly."""
        hkl = unique_hkl(80)
        refl = replicated(
            hkl, torch.linspace(5.0, 400.0, 80, dtype=DTYPE), 6, 30.0, seed=8
        )
        amap = build_asu_map(refl.hkl, "P 1")
        first = stats.cc_half_sigma_tau(stats.sigma_tau_merge(refl, amap))
        second = stats.cc_half_sigma_tau(stats.sigma_tau_merge(refl, amap))
        assert first == second

    def test_too_few_reflections(self) -> None:
        merged = MergedData(
            hkl=torch.zeros(1, 3, dtype=IDTYPE),
            intensity=torch.ones(1, dtype=DTYPE),
            sigma=torch.ones(1, dtype=DTYPE),
            multiplicity=torch.tensor([4], dtype=IDTYPE),
            d=torch.ones(1, dtype=DTYPE),
            centric=torch.zeros(1, dtype=torch.bool),
            plus=torch.ones(1, dtype=torch.bool),
        )
        assert stats.cc_half_sigma_tau(merged) == (0.0, 1)


class TestCCStar:
    @pytest.mark.parametrize(
        ("cc_half", "expected"),
        [
            (1.0, 1.0),
            (0.0, 0.0),
            (0.5, math.sqrt(2.0 * 0.5 / 1.5)),
            (0.9, math.sqrt(1.8 / 1.9)),
        ],
    )
    def test_known_values(self, cc_half: float, expected: float) -> None:
        assert stats.cc_star(cc_half) == pytest.approx(expected)

    def test_negative_is_signed(self) -> None:
        assert stats.cc_star(-0.5) == pytest.approx(-math.sqrt(1.0 / 0.5))

    def test_poles_and_nan(self) -> None:
        assert stats.cc_star(-1.0) == float("-inf")
        assert math.isnan(stats.cc_star(float("nan")))


class TestRSplit:
    def test_perfect_data_gives_zero(self) -> None:
        """Identical halves make the numerator vanish."""
        hkl = unique_hkl(150)
        true_i = torch.linspace(10.0, 900.0, 150, dtype=DTYPE)
        refl = replicated(hkl, true_i, n_obs=6, noise=0.0, seed=9)
        amap = build_asu_map(refl.hkl, "P 1")
        assert stats.r_split(refl, amap).value == pytest.approx(0.0, abs=1e-12)

    def test_known_two_group_value(self) -> None:
        """A hand-computable split: constant offset between halves."""
        # Four observations per reflection, two at I-a and two at I+a.
        # Any split into halves of two either separates them (halves
        # I-a and I+a) or mixes them (both halves I).
        hkl = unique_hkl(2).repeat_interleave(4, dim=0)
        intensity = torch.tensor([90.0, 90.0, 110.0, 110.0] * 2, dtype=DTYPE)
        refl = make_reflections(hkl, intensity, torch.ones(8, dtype=DTYPE))
        amap = build_asu_map(refl.hkl, "P 1")
        value = stats.r_split(refl, amap, seed=0, n_repeats=200).value
        # The mixed split contributes 0, the separated split
        # 2^-1/2 * (2 * 20) / (0.5 * 400) = 0.1414.
        assert 0.0 < value < 0.15


class TestCCAnom:
    def test_perfect_anomalous_signal(self) -> None:
        """A noiseless Bijvoet difference correlates perfectly."""
        base = unique_hkl(150)
        hkl = torch.cat([base, -base], dim=0)
        generator = torch.Generator().manual_seed(11)
        mean_i = torch.linspace(100.0, 900.0, 150, dtype=DTYPE)
        delta = 30.0 * torch.randn(150, generator=generator, dtype=DTYPE)
        true_i = torch.cat([mean_i + delta, mean_i - delta])
        refl = replicated(hkl, true_i, n_obs=6, noise=0.0, seed=12)
        amap = build_asu_map(refl.hkl, "P 1")
        cc = stats.cc_anom(refl, amap, seed=0)
        assert cc.value == pytest.approx(1.0, abs=1e-9)
        assert cc.n_refl == 150

    def test_no_anomalous_signal(self) -> None:
        """Equal Friedel mates leave only noise to correlate."""
        base = unique_hkl(200)
        hkl = torch.cat([base, -base], dim=0)
        mean_i = torch.linspace(100.0, 900.0, 200, dtype=DTYPE)
        refl = replicated(
            hkl, torch.cat([mean_i, mean_i]), n_obs=8, noise=50.0, seed=13
        )
        amap = build_asu_map(refl.hkl, "P 1")
        cc = stats.cc_anom(refl, amap, seed=0, n_repeats=8)
        assert abs(cc.value) < 0.2

    def test_centrics_are_excluded(self) -> None:
        """A centric-only dataset yields no Bijvoet pairs."""
        # In P -1 every reflection is centric.
        base = unique_hkl(40)
        hkl = torch.cat([base, -base], dim=0)
        refl = replicated(
            hkl,
            torch.linspace(10.0, 400.0, 80, dtype=DTYPE),
            n_obs=4,
            noise=5.0,
            seed=14,
        )
        amap = build_asu_map(refl.hkl, "P -1")
        assert bool(amap.centric.all())
        assert stats.cc_anom(refl, amap).n_refl == 0


class TestWeightedCCHalf:
    def test_perfect_data_gives_one(self) -> None:
        hkl = unique_hkl(200)
        true_i = torch.linspace(10.0, 1000.0, 200, dtype=DTYPE)
        refl = replicated(hkl, true_i, n_obs=8, noise=0.0, seed=15)
        amap = build_asu_map(refl.hkl, "P 1")
        cc, neff = stats.weighted_cc_half(refl, amap, seed=0)
        assert cc.value == pytest.approx(1.0, abs=1e-9)
        assert 0 < neff <= 200

    def test_neff_equals_n_for_uniform_weights(self) -> None:
        """Kish's effective size is the raw count when weights match."""
        hkl = unique_hkl(120)
        refl = replicated(
            hkl,
            torch.linspace(10.0, 600.0, 120, dtype=DTYPE),
            n_obs=4,
            noise=20.0,
            seed=16,
        )
        amap = build_asu_map(refl.hkl, "P 1")
        _, neff = stats.weighted_cc_half(refl, amap, seed=0)
        assert neff == pytest.approx(120.0)


class TestCompleteness:
    def test_full_coverage(self) -> None:
        """Observing the whole complete set gives exactly one."""
        complete = stats.build_complete_set(CELL, "P 1", d_min=4.0)
        hkl = complete.hkl
        refl = replicated(
            hkl,
            torch.full((hkl.shape[0],), 100.0, dtype=DTYPE),
            n_obs=2,
            noise=1.0,
            seed=17,
        )
        amap = build_asu_map(refl.hkl, "P 1")
        merged = stats.sigma_tau_merge(refl, amap)
        value = stats.completeness(
            merged,
            CELL,
            "P 1",
            d_min=4.0,
            d_max=1e6,
            complete=complete,
        )
        assert value == pytest.approx(1.0)

    def test_half_coverage(self) -> None:
        """Dropping half the complete set halves the completeness."""
        complete = stats.build_complete_set(CELL, "P 1", d_min=4.0)
        keep = complete.hkl[::2]
        refl = replicated(
            keep,
            torch.full((keep.shape[0],), 100.0, dtype=DTYPE),
            n_obs=2,
            noise=1.0,
            seed=18,
        )
        amap = build_asu_map(refl.hkl, "P 1")
        merged = stats.sigma_tau_merge(refl, amap)
        value = stats.completeness(
            merged, CELL, "P 1", d_min=4.0, d_max=1e6, complete=complete
        )
        n_total = complete.hkl.shape[0]
        assert value == pytest.approx(keep.shape[0] / n_total)

    def test_symmetry_shrinks_the_complete_set(self) -> None:
        """A higher-symmetry group needs fewer unique reflections."""
        p1 = stats.build_complete_set(CELL, "P 1", d_min=4.0)
        p422 = stats.build_complete_set(CELL, "P 4 2 2", d_min=4.0)
        assert p422.hkl.shape[0] < p1.hkl.shape[0]

    def test_anomalous_set_is_larger(self) -> None:
        """Keeping Friedel mates apart adds the acentric mates back."""
        plain = stats.build_complete_set(CELL, "P 4 2 2", d_min=4.0)
        anom = stats.build_complete_set(
            CELL, "P 4 2 2", d_min=4.0, anomalous=True
        )
        n_acentric = int((~plain.centric).sum())
        assert anom.hkl.shape[0] == plain.hkl.shape[0] + n_acentric


class TestIOverSigma:
    def test_the_two_ratios_differ(self) -> None:
        """Mean of the ratio is not the ratio of the means."""
        merged = MergedData(
            hkl=torch.zeros(3, 3, dtype=IDTYPE),
            intensity=torch.tensor([10.0, 100.0, 1000.0], dtype=DTYPE),
            sigma=torch.tensor([1.0, 1.0, 100.0], dtype=DTYPE),
            multiplicity=torch.full((3,), 4, dtype=IDTYPE),
            d=torch.ones(3, dtype=DTYPE),
            centric=torch.zeros(3, dtype=torch.bool),
            plus=torch.ones(3, dtype=torch.bool),
        )
        ios = stats.i_over_sigma(merged)
        assert ios.i_mean == pytest.approx(370.0)
        assert ios.sigi_mean == pytest.approx(34.0)
        assert ios.i_over_sigma_mean == pytest.approx((10 + 100 + 10) / 3)
        assert ios.i_mean_over_sigi_mean == pytest.approx(370.0 / 34.0)
        assert ios.i_over_sigma_mean != ios.i_mean_over_sigi_mean

    def test_unmerged_ratio_is_smaller(self) -> None:
        """Merging `n` observations buys roughly `sqrt(n)` in I/sigma."""
        hkl = unique_hkl(100)
        refl = replicated(
            hkl,
            torch.full((100,), 100.0, dtype=DTYPE),
            n_obs=9,
            noise=10.0,
            seed=19,
        )
        amap = build_asu_map(refl.hkl, "P 1")
        merged = stats.sigma_tau_merge(refl, amap)
        ios = stats.i_over_sigma(merged, refl)
        assert ios.unmerged_i_over_sigma_mean == pytest.approx(10.0, rel=0.1)


class TestMergingStatistics:
    def test_end_to_end_table(self) -> None:
        """The top-level entry point runs and is self-consistent."""

        class FakeExperiment:
            cell = CELL
            space_group = "P 4 2 2"

        complete = stats.build_complete_set(CELL, "P 4 2 2", d_min=2.5)
        hkl = torch.cat([complete.hkl, -complete.hkl], dim=0)
        generator = torch.Generator().manual_seed(20)
        # Friedel mates share a true intensity, so the Laue merge these
        # statistics are built on sees a consistent set of equivalents.
        base_i = 1000.0 * torch.rand(
            complete.hkl.shape[0], generator=generator, dtype=DTYPE
        )
        true_i = torch.cat([base_i, base_i])
        refl = replicated(hkl, true_i, n_obs=6, noise=20.0, seed=21)
        amap = build_asu_map(refl.hkl, "P 4 2 2")

        result = stats.merging_statistics(
            refl, amap, FakeExperiment(), n_bins=4, n_repeats=2
        )
        assert len(result.shells) == 4
        assert result.overall.n_obs == len(refl)
        assert result.overall.n_uniq == sum(s.n_uniq for s in result.shells)
        assert result.overall.n_obs == sum(s.n_obs for s in result.shells)
        assert result.overall.completeness == pytest.approx(1.0)
        assert result.overall.cc_half_sigma_tau > 0.99
        assert result.overall.r_merge < 0.05
        table = result.to_table()
        assert len(table.splitlines()) == 6
        assert str(result) == table

    def test_shell_multiplicities_are_consistent(self) -> None:
        """Every observation lands in exactly one shell."""

        class FakeExperiment:
            cell = CELL
            space_group = "P 1"

        hkl = unique_hkl(300)
        refl = replicated(
            hkl,
            torch.linspace(10.0, 900.0, 300, dtype=DTYPE),
            n_obs=4,
            noise=15.0,
            seed=22,
        )
        amap = build_asu_map(refl.hkl, "P 1")
        result = stats.merging_statistics(
            refl, amap, FakeExperiment(), n_bins=3
        )
        assert sum(s.n_obs for s in result.shells) == len(refl)

    def test_rejects_all_zero_sigma(self) -> None:
        class FakeExperiment:
            cell = CELL
            space_group = "P 1"

        hkl = unique_hkl(10)
        refl = make_reflections(
            hkl.repeat_interleave(2, dim=0),
            torch.ones(20, dtype=DTYPE),
            torch.zeros(20, dtype=DTYPE),
        )
        amap = build_asu_map(refl.hkl, "P 1")
        with pytest.raises(ValueError):
            stats.merging_statistics(refl, amap, FakeExperiment())
