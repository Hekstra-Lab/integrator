"""Unit tests for `integrator.dials_port.merge` and `.french_wilson`.

Small synthetic inputs with hand-computed answers. Agreement with the
real cctbx implementation on a full dataset is checked separately by
`scripts/verify_dials_port_merge.py`, which needs a DIALS runtime.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from integrator.dials_port.french_wilson import (
    determine_binning_counting_sorted,
    french_wilson,
    interp,
    standardized_median,
)
from integrator.dials_port.merge import (
    grouping,
    merge,
    r_factors,
    scale_observations,
    separate_bijvoet,
)
from integrator.dials_port.types import (
    DTYPE,
    IDTYPE,
    AsuMap,
    MergedData,
    Reflections,
)

SQRT2 = math.sqrt(2.0)


def make_refl(
    intensity: list[float],
    sigma: list[float],
    scale: list[float] | None = None,
    d: list[float] | None = None,
) -> Reflections:
    """Build a minimal `Reflections` with the columns merging touches."""
    n = len(intensity)
    i = torch.tensor(intensity, dtype=DTYPE)
    s = torch.tensor(sigma, dtype=DTYPE)
    ones = torch.ones(n, dtype=DTYPE)
    return Reflections(
        hkl=torch.zeros((n, 3), dtype=IDTYPE),
        intensity=i,
        variance=s * s,
        d=torch.tensor(d, dtype=DTYPE) if d else ones.clone(),
        phi=torch.zeros(n, dtype=DTYPE),
        s1=torch.zeros((n, 3), dtype=DTYPE),
        partiality=ones.clone(),
        lp=ones.clone(),
        qe=ones.clone(),
        dataset_id=torch.zeros(n, dtype=IDTYPE),
        scale=torch.tensor(scale, dtype=DTYPE) if scale else ones.clone(),
    )


def make_amap(
    group_id: list[int],
    centric: list[bool] | None = None,
    plus: list[bool] | None = None,
) -> AsuMap:
    """Build an `AsuMap` with an explicit Laue grouping.

    The anomalous grouping is derived as `2 * laue + plus`, exactly as
    `symmetry.build_asu_map` does.
    """
    n = len(group_id)
    laue = torch.tensor(group_id, dtype=IDTYPE)
    n_laue = int(laue.max()) + 1
    p = torch.tensor(plus if plus else [True] * n, dtype=torch.bool)
    key = laue * 2 + p.to(IDTYPE)
    uniq, anom = torch.unique(key, return_inverse=True)
    hkl = torch.stack(
        [laue, torch.zeros(n, dtype=IDTYPE), torch.zeros(n, dtype=IDTYPE)],
        dim=1,
    )
    return AsuMap(
        asu_hkl=hkl,
        isym=p.to(IDTYPE),
        plus=p,
        centric=torch.tensor(
            centric if centric else [False] * n, dtype=torch.bool
        ),
        epsilon=torch.ones(n, dtype=DTYPE),
        sys_absent=torch.zeros(n, dtype=torch.bool),
        group_id=anom.to(IDTYPE),
        group_id_laue=laue,
        n_groups=int(uniq.numel()),
        n_groups_laue=n_laue,
    )


class TestScaling:
    def test_divides_intensity_and_squares_the_variance(self) -> None:
        refl = make_refl([10.0, 20.0], [1.0, 2.0], scale=[2.0, 4.0])
        intensity, variance = scale_observations(refl)
        assert torch.allclose(intensity, torch.tensor([5.0, 5.0]).double())
        assert torch.allclose(variance, torch.tensor([0.25, 0.25]).double())


class TestMerge:
    def test_weighted_mean_known_answer(self) -> None:
        # w = [1, 1/4]; mean = (10 + 5) / 1.25 = 12; sigma = sqrt(1/1.25).
        refl = make_refl([10.0, 20.0], [1.0, 2.0])
        merged = merge(refl, make_amap([0, 0]))
        assert len(merged) == 1
        assert merged.intensity.item() == pytest.approx(12.0, rel=1e-15)
        assert merged.sigma.item() == pytest.approx(
            math.sqrt(1.0 / 1.25), rel=1e-15
        )
        assert merged.multiplicity.tolist() == [2]

    def test_internal_variance_takes_the_larger_estimate(self) -> None:
        # V_w = 1.25 / (1.5625 - 1.0625) * 20 = 50; /n = 25 > 1/sum_w.
        refl = make_refl([10.0, 20.0], [1.0, 2.0])
        merged = merge(refl, make_amap([0, 0]), use_internal_variance=True)
        assert merged.sigma.item() == pytest.approx(5.0, rel=1e-15)

    def test_internal_variance_floored_by_propagated_error(self) -> None:
        # Identical observations have zero scatter, so the external
        # estimate must win.
        refl = make_refl([10.0, 10.0], [1.0, 1.0])
        merged = merge(refl, make_amap([0, 0]), use_internal_variance=True)
        assert merged.sigma.item() == pytest.approx(math.sqrt(0.5), rel=1e-15)

    def test_singleton_keeps_its_own_sigma(self) -> None:
        refl = make_refl([7.0], [3.0])
        merged = merge(refl, make_amap([0]))
        assert merged.intensity.item() == 7.0
        assert merged.sigma.item() == 3.0

    def test_singleton_sigma_is_verbatim_not_recomputed(self) -> None:
        # sqrt(1 / (1 / s**2)) is not bitwise s for most s.
        sigma = 0.1
        refl = make_refl([1.0], [sigma])
        merged = merge(refl, make_amap([0]))
        assert merged.sigma.item() == sigma

    def test_sigma_threshold_drops_but_still_counts(self) -> None:
        # max_sigma = 2, threshold = 2e-6, so the 1e-9 observation goes.
        refl = make_refl([10.0, 20.0, 1e6], [1.0, 2.0, 1e-9])
        merged = merge(refl, make_amap([0, 0, 0]))
        assert merged.intensity.item() == pytest.approx(12.0, rel=1e-15)
        assert merged.multiplicity.tolist() == [3]

    def test_sigma_threshold_can_leave_a_singleton(self) -> None:
        refl = make_refl([5.0, 999.0], [1.0, 1e-9])
        merged = merge(refl, make_amap([0, 0]))
        assert merged.intensity.item() == 5.0
        assert merged.sigma.item() == 1.0
        assert merged.multiplicity.tolist() == [2]

    def test_sigma_threshold_is_strict(self) -> None:
        # sigma exactly equal to the threshold is dropped: cctbx keeps
        # observations with `s > threshold`, not `s >= threshold`.
        refl = make_refl([10.0, 999.0], [1.0, 1e-6])
        merged = merge(refl, make_amap([0, 0]))
        assert merged.intensity.item() == 10.0

    def test_group_of_zero_sigmas_is_emptied(self) -> None:
        # `Reflections` stores variance, so sigma is non-negative by
        # construction and only the zero limb of cctbx's guard against
        # non-positive sigmas is reachable. A zero maximum makes the
        # threshold zero, and `s > 0` then rejects every member.
        refl = make_refl([10.0, 20.0], [0.0, 0.0])
        merged = merge(refl, make_amap([0, 0]))
        assert merged.intensity.item() == 0.0
        assert merged.sigma.item() == 0.0
        assert merged.multiplicity.tolist() == [2]

    def test_groups_are_independent(self) -> None:
        refl = make_refl([10.0, 20.0, 7.0], [1.0, 2.0, 3.0])
        merged = merge(refl, make_amap([0, 0, 1]))
        assert len(merged) == 2
        assert merged.intensity.tolist() == pytest.approx([12.0, 7.0])
        assert merged.multiplicity.tolist() == [2, 1]

    def test_scale_is_applied_before_merging(self) -> None:
        refl = make_refl([20.0, 20.0], [2.0, 2.0], scale=[2.0, 2.0])
        merged = merge(refl, make_amap([0, 0]))
        assert merged.intensity.item() == pytest.approx(10.0, rel=1e-15)
        assert merged.sigma.item() == pytest.approx(math.sqrt(0.5), rel=1e-15)


class TestGrouping:
    def test_laue_grouping_passes_through(self) -> None:
        amap = make_amap([0, 0, 1], plus=[True, False, True])
        gid, n = grouping(amap, anomalous=False)
        assert n == 2
        assert gid.tolist() == [0, 0, 1]

    def test_acentric_mates_split(self) -> None:
        amap = make_amap([0, 0], plus=[True, False])
        _, n = grouping(amap, anomalous=True)
        assert n == 2

    def test_centric_mates_pool(self) -> None:
        # cctbx's anomalous ASU maps h and -h together for centrics.
        amap = make_amap([0, 0], centric=[True, True], plus=[True, False])
        gid, n = grouping(amap, anomalous=True)
        assert n == 1
        assert gid.tolist() == [0, 0]

    def test_mixed_centric_and_acentric(self) -> None:
        amap = make_amap(
            [0, 0, 1, 1],
            centric=[True, True, False, False],
            plus=[True, False, True, False],
        )
        _, n = grouping(amap, anomalous=True)
        assert n == 3


class TestRFactors:
    def test_known_answer(self) -> None:
        # mean = 12; num = |10-12| + |20-12| = 10; den = 30.
        refl = make_refl([10.0, 20.0], [1.0, 2.0])
        stats = r_factors(refl, make_amap([0, 0]))
        assert stats["rmerge"] == pytest.approx(1.0 / 3.0, rel=1e-14)
        assert stats["rmeas"] == pytest.approx(SQRT2 / 3.0, rel=1e-14)
        assert stats["rpim"] == pytest.approx(1.0 / 3.0, rel=1e-14)
        assert stats["rint"] == pytest.approx(1.0 / 3.0, rel=1e-14)

    def test_singletons_are_excluded(self) -> None:
        # Adding a lone observation must not move any statistic.
        pair = make_refl([10.0, 20.0], [1.0, 2.0])
        with_extra = make_refl([10.0, 20.0, 500.0], [1.0, 2.0, 1.0])
        a = r_factors(pair, make_amap([0, 0]))
        b = r_factors(with_extra, make_amap([0, 0, 1]))
        for key in a:
            assert a[key] == pytest.approx(b[key], rel=1e-14)

    def test_rint_uses_absolute_denominator(self) -> None:
        # A negative observation makes the signed and absolute sums
        # differ, which is the only thing separating rmerge from rint.
        refl = make_refl([-10.0, 30.0], [1.0, 1.0])
        stats = r_factors(refl, make_amap([0, 0]))
        assert stats["rmerge"] == pytest.approx(40.0 / 20.0, rel=1e-14)
        assert stats["rint"] == pytest.approx(40.0 / 40.0, rel=1e-14)

    def test_uses_the_weighted_mean(self) -> None:
        # The unweighted mean would be 15, giving num = 10 either way;
        # unequal sigmas break the tie.
        refl = make_refl([10.0, 20.0], [1.0, 3.0])
        stats = r_factors(refl, make_amap([0, 0]))
        weighted = (10.0 + 20.0 / 9.0) / (1.0 + 1.0 / 9.0)
        expected = (abs(10.0 - weighted) + abs(20.0 - weighted)) / 30.0
        assert stats["rmerge"] == pytest.approx(expected, rel=1e-14)

    def test_zero_denominator_reports_zero(self) -> None:
        refl = make_refl([-5.0, 5.0], [1.0, 1.0])
        stats = r_factors(refl, make_amap([0, 0]))
        assert stats["rmerge"] == 0.0


class TestSeparateBijvoet:
    def make_merged(
        self,
        intensity: list[float],
        sigma: list[float],
        plus: list[bool],
        hkl: list[list[int]],
        centric: list[bool] | None = None,
    ) -> MergedData:
        n = len(intensity)
        return MergedData(
            hkl=torch.tensor(hkl, dtype=IDTYPE),
            intensity=torch.tensor(intensity, dtype=DTYPE),
            sigma=torch.tensor(sigma, dtype=DTYPE),
            multiplicity=torch.ones(n, dtype=IDTYPE),
            d=torch.ones(n, dtype=DTYPE),
            centric=torch.tensor(
                centric if centric else [False] * n, dtype=torch.bool
            ),
            plus=torch.tensor(plus, dtype=torch.bool),
        )

    def test_pairs_mates_and_forms_the_difference(self) -> None:
        merged = self.make_merged(
            [10.0, 6.0], [3.0, 4.0], [True, False], [[1, 0, 0], [1, 0, 0]]
        )
        out = separate_bijvoet(merged)
        assert out["I_plus"].tolist() == [10.0]
        assert out["I_minus"].tolist() == [6.0]
        assert out["dano"].item() == pytest.approx(4.0)
        assert out["sigdano"].item() == pytest.approx(5.0)
        assert bool(out["has_plus"][0]) and bool(out["has_minus"][0])

    def test_missing_mate_is_nan(self) -> None:
        merged = self.make_merged([10.0], [3.0], [True], [[1, 0, 0]])
        out = separate_bijvoet(merged)
        assert math.isnan(out["I_minus"].item())
        assert math.isnan(out["dano"].item())
        assert bool(out["has_plus"][0])
        assert not bool(out["has_minus"][0])

    def test_centric_fills_both_hemispheres_without_a_difference(
        self,
    ) -> None:
        merged = self.make_merged(
            [10.0], [3.0], [True], [[2, 0, 0]], centric=[True]
        )
        out = separate_bijvoet(merged)
        assert out["I_plus"].item() == 10.0
        assert out["I_minus"].item() == 10.0
        assert math.isnan(out["dano"].item())
        assert math.isnan(out["sigdano"].item())

    def test_distinct_reflections_stay_separate(self) -> None:
        merged = self.make_merged(
            [10.0, 6.0, 1.0],
            [1.0, 1.0, 1.0],
            [True, False, True],
            [[1, 0, 0], [1, 0, 0], [2, 0, 0]],
        )
        out = separate_bijvoet(merged)
        assert out["hkl"].shape == (2, 3)


class TestInterp:
    def test_matches_numpy_including_clamping(self) -> None:
        # Tolerance rather than equality: numpy's compiled kernel
        # contracts `slope * (x - x0) + y0` into an FMA on platforms
        # that have one, which moves the last bit.
        xp = np.array([0.0, 1.0, 2.5, 4.0])
        fp = np.array([1.0, -2.0, 7.0, 0.5])
        x = np.array([-3.0, 0.0, 0.7, 2.5, 3.9, 4.0, 9.0])
        got = interp(
            torch.from_numpy(x), torch.from_numpy(xp), torch.from_numpy(fp)
        )
        assert np.allclose(
            got.numpy(), np.interp(x, xp, fp), rtol=1e-15, atol=0
        )

    def test_hits_sample_points_exactly(self) -> None:
        xp = torch.tensor([0.0, 1.0, 2.5, 4.0], dtype=DTYPE)
        fp = torch.tensor([1.0, -2.0, 7.0, 0.5], dtype=DTYPE)
        assert interp(xp, xp, fp).tolist() == fp.tolist()

    def test_clamps_rather_than_extrapolating(self) -> None:
        xp = torch.tensor([0.0, 1.0], dtype=DTYPE)
        fp = torch.tensor([5.0, 6.0], dtype=DTYPE)
        out = interp(torch.tensor([-100.0, 100.0], dtype=DTYPE), xp, fp)
        assert out.tolist() == [5.0, 6.0]


class TestStandardizedMedian:
    def test_scales_the_median_by_log2(self) -> None:
        a = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
        assert standardized_median(a).item() == pytest.approx(
            2.0 / math.log(2.0)
        )

    def test_even_sample_averages_the_two_central_values(self) -> None:
        # torch.median alone would return 2.0 here, numpy returns 2.5.
        a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=DTYPE)
        assert standardized_median(a).item() == pytest.approx(
            2.5 / math.log(2.0)
        )

    def test_empty_sample_is_nan(self) -> None:
        assert math.isnan(
            standardized_median(torch.empty(0, dtype=DTYPE)).item()
        )


class TestBinning:
    def test_edges_span_the_data_exactly(self) -> None:
        x = torch.linspace(1.0, 5.0, 100, dtype=DTYPE)
        edges = determine_binning_counting_sorted(x, max_bins=5)
        assert edges[0].item() == x.min().item()
        assert edges[-1].item() == x.max().item()

    def test_min_bin_size_beats_max_bins(self) -> None:
        # 100 points, min_bin_size 20 => at most 5 bins whatever
        # `max_bins` asks for.
        x = torch.rand(100, dtype=DTYPE)
        edges = determine_binning_counting_sorted(x, max_bins=60)
        assert edges.numel() - 1 == 5

    def test_matches_numpy_reference(self) -> None:
        x = torch.rand(500, dtype=DTYPE)
        edges = determine_binning_counting_sorted(x, max_bins=60)
        n, per = 500, max(500 // 60, 20)
        want = np.interp(
            np.linspace(0, n, n // per + 1),
            np.arange(n),
            np.sort(x.numpy()),
        )
        assert np.allclose(edges.numpy(), want, rtol=1e-15, atol=0)

    def test_survives_fewer_reflections_than_one_bin(self) -> None:
        # DIALS would produce a zero-bin edge array and crash here.
        x = torch.linspace(1.0, 2.0, 5, dtype=DTYPE)
        assert determine_binning_counting_sorted(x).numel() == 2


def make_merged_fw(
    intensity: float,
    sigma: float,
    centric: bool,
    n: int = 60,
) -> MergedData:
    """Constant-intensity merged set, so `Sigma` is exactly `I / log 2`.

    A flat intensity makes the per-shell median independent of the
    binning, which pins `h` to a closed form the tests can check.
    """
    return MergedData(
        hkl=torch.zeros((n, 3), dtype=IDTYPE),
        intensity=torch.full((n,), intensity, dtype=DTYPE),
        sigma=torch.full((n,), sigma, dtype=DTYPE),
        multiplicity=torch.full((n,), 5, dtype=IDTYPE),
        d=torch.linspace(2.0, 8.0, n, dtype=DTYPE),
        centric=torch.full((n,), centric, dtype=torch.bool),
        plus=torch.ones(n, dtype=torch.bool),
    )


class TestFrenchWilson:
    def test_acentric_gaussian_branch(self) -> None:
        i_obs, sig = 100.0, 1.0
        out = french_wilson(make_merged_fw(i_obs, sig, centric=False))
        big_sigma = i_obs / math.log(2.0)
        h = i_obs / sig - abs(sig / big_sigma)
        assert h >= 3.0
        assert bool(out["valid"].all())
        j = h * sig
        assert out["J"][0].item() == pytest.approx(j, rel=1e-14)
        assert out["sigJ"][0].item() == pytest.approx(sig, rel=1e-14)
        assert out["F"][0].item() == pytest.approx(math.sqrt(j), rel=1e-14)
        assert out["sigF"][0].item() == pytest.approx(
            0.5 * sig / math.sqrt(j), rel=1e-14
        )

    def test_centric_asymptotic_branch(self) -> None:
        i_obs, sig = 100.0, 1.0
        out = french_wilson(make_merged_fw(i_obs, sig, centric=True))
        big_sigma = i_obs / math.log(2.0)
        h = i_obs / sig - abs(sig / (2.0 * big_sigma))
        assert h >= 4.0
        h2 = 1.0 / h**2
        h4, h6 = h2**2, h2**3
        post_f = math.sqrt(h) * (
            1.0
            - (3.0 / 8.0) * h2
            - (87.0 / 128.0) * h4
            - (2889.0 / 1024.0) * h6
        )
        post_sig_f = math.sqrt(
            h * ((1.0 / 4.0) * h2 + (15.0 / 32.0) * h4 + (273.0 / 128.0) * h6)
        )
        assert out["F"][0].item() == pytest.approx(
            post_f * math.sqrt(sig), rel=1e-14
        )
        assert out["sigF"][0].item() == pytest.approx(
            post_sig_f * math.sqrt(sig), rel=1e-14
        )
        assert out["J"][0].item() == pytest.approx(
            h * sig * (1.0 - 0.5 * h2 - 0.75 * h4 - 3.0 * h6), rel=1e-14
        )
        assert out["sigJ"][0].item() == pytest.approx(
            2.0 * sig * post_f * post_sig_f, rel=1e-14
        )

    def test_acentric_table_branch(self) -> None:
        from integrator.dials_port import fw_tables

        i_obs, sig = 1.0, 1.0
        out = french_wilson(make_merged_fw(i_obs, sig, centric=False))
        big_sigma = i_obs / math.log(2.0)
        h = i_obs / sig - abs(sig / big_sigma)
        assert -4.0 <= h < 3.0
        grid = np.arange(len(fw_tables.ac_zj)) / 10.0 - 4.0
        want_j = np.interp(h, grid, fw_tables.ac_zj) * sig
        want_f = np.interp(h, grid, fw_tables.ac_zf) * math.sqrt(sig)
        assert out["J"][0].item() == pytest.approx(want_j, rel=1e-14)
        assert out["F"][0].item() == pytest.approx(want_f, rel=1e-14)

    def test_centric_table_branch(self) -> None:
        from integrator.dials_port import fw_tables

        i_obs, sig = 1.0, 1.0
        out = french_wilson(make_merged_fw(i_obs, sig, centric=True))
        big_sigma = i_obs / math.log(2.0)
        h = i_obs / sig - abs(sig / (2.0 * big_sigma))
        assert -4.0 <= h < 4.0
        grid = np.arange(len(fw_tables.c_zj)) / 10.0 - 4.0
        want_j = np.interp(h, grid, fw_tables.c_zj) * sig
        want_f = np.interp(h, grid, fw_tables.c_zf) * math.sqrt(sig)
        assert out["J"][0].item() == pytest.approx(want_j, rel=1e-14)
        assert out["F"][0].item() == pytest.approx(want_f, rel=1e-14)

    def test_rejects_weak_reflections(self) -> None:
        # I / sigma = -4 is below the h_min + 0.3 = -3.7 cut.
        out = french_wilson(make_merged_fw(-4.0, 1.0, centric=False))
        assert not bool(out["valid"].any())

    def test_table_lengths_bound_the_branch_switch(self) -> None:
        from integrator.dials_port import fw_tables

        assert len(fw_tables.ac_zj) == 71
        assert len(fw_tables.c_zj) == 81
        for table in (
            fw_tables.ac_zj_sd,
            fw_tables.ac_zf,
            fw_tables.ac_zf_sd,
        ):
            assert len(table) == len(fw_tables.ac_zj)
        for table in (fw_tables.c_zj_sd, fw_tables.c_zf, fw_tables.c_zf_sd):
            assert len(table) == len(fw_tables.c_zj)


class TestFrenchWilsonLatentBugs:
    def make_one_zero_sigma(self) -> MergedData:
        merged = make_merged_fw(100.0, 1.0, centric=False)
        merged.sigma[0] = 0.0
        return merged

    def test_zero_sigma_with_positive_intensity_leaks_by_default(
        self,
    ) -> None:
        # DIALS' `&` lets this row through into the I / sigma division.
        out = french_wilson(self.make_one_zero_sigma())
        assert bool(out["valid"][0])
        assert math.isnan(out["J"][0].item())
        assert math.isnan(out["F"][0].item())

    def test_fix_bugs_rejects_it(self) -> None:
        out = french_wilson(self.make_one_zero_sigma(), fix_bugs=True)
        assert not bool(out["valid"][0])
        assert bool(out["valid"][1:].all())

    def test_rejection_criterion_is_inert_by_default(self) -> None:
        # `sigma_iobs_rejection_criterion` never reaches `h_min` in
        # DIALS, so raising it must change nothing.
        merged = make_merged_fw(-4.0, 1.0, centric=False)
        loose = french_wilson(merged, sigma_iobs_rejection_criterion=-10.0)
        assert not bool(loose["valid"].any())

    def test_fix_bugs_honours_the_rejection_criterion(self) -> None:
        merged = make_merged_fw(-4.0, 1.0, centric=False)
        out = french_wilson(
            merged, sigma_iobs_rejection_criterion=-10.0, fix_bugs=True
        )
        assert bool(out["valid"].all())
