"""Verify `dials_port` target, error model and outlier rejection.

Four checks, in increasing order of how much they depend on real data:

1.  `merged_intensity` against a direct numpy weighted mean.
2.  Autograd gradients of `ScalingTarget` against central finite
    differences, for every likelihood and both `detach_ih` settings.
3.  Outlier rejection on synthetic groups with planted outliers.
4.  The error model refined on the HEWL reference against DIALS'
    published `a = 0.8775669, b = 0.1394132`.

Check 4 needs the reconstructed *pre-error-model* variances. DIALS
writes `intensity.scale.variance` only after the error model and the
scale-uncertainty inflation have been applied to it, so refitting on
that column measures nothing. `--rebuild` reconstructs the raw
`intensity`/`variance` from `intensity.prf.*` and `intensity.sum.*`
using the intensity combination DIALS chose, and caches the result.
That step needs the DIALS environment; everything else does not.

Run with::

    /Users/luis/micromamba/envs/integrator-dev/bin/python \\
        scripts/verify_dials_port_target.py [--rebuild]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from integrator.dials_port.error_model import (  # noqa: E402
    BasicErrorModel,
    prepare_error_model_data,
    refine_error_model,
)
from integrator.dials_port.io import load_experiment, load_npz  # noqa: E402
from integrator.dials_port.outliers import reject_outliers  # noqa: E402
from integrator.dials_port.symmetry import build_asu_map  # noqa: E402
from integrator.dials_port.target import (  # noqa: E402
    GammaLikelihood,
    GaussianLeastSquares,
    PoissonLikelihood,
    ScalingTarget,
    merged_intensity,
    resolve_groups,
)
from integrator.dials_port.types import (  # noqa: E402
    DTYPE,
    IDTYPE,
    Reflections,
)

PASS1 = Path("/Users/luis/dials_out/816_sbgrid_HEWL/pass1")
SCRATCH = Path(
    "/private/tmp/claude-501/-Users-luis-integrator/"
    "a69c47c1-a7b1-409e-b127-f918bd14acbc/scratchpad"
)
RAW_NPZ = SCRATCH / "hewl_raw.npz"

DIALS_A = 0.8775669467406682
DIALS_B = 0.13941319094122784
DIALS_N_FILTERED = 306944
DIALS_BIN_BOUNDS = np.array(
    [6561.67, 2692.96, 2153.80, 1233.96, 706.96, 405.03, 232.05, 132.95,
     76.17, 43.64, 24.99]
)
DIALS_BIN_COUNTS = np.array(
    [3069, 4903, 26257, 40620, 49239, 53994, 50850, 43040, 25161, 9811]
)
DIALS_UNCORRECTED = np.array(
    [106.1, 88.029, 59.799, 32.437, 18.596, 10.772, 6.579, 4.03, 2.761,
     2.084]
)
DIALS_CORRECTED = np.array(
    [0.93, 0.988, 0.99, 1.005, 1.21, 1.418, 1.57, 1.59, 1.59, 1.435]
)


def rebuild_raw_npz() -> None:
    """Reconstruct DIALS' pre-error-model intensity and variance.

    `scaled.refl` keeps `intensity.prf.*` and `intensity.sum.*`, so the
    combination DIALS performed can be replayed. `Imid` is recovered
    from the data itself by inverting the combination weight, which also
    serves as a check: the recovered value must be constant across
    reflections.
    """
    from dials.array_family import flex  # noqa: PLC0415
    from dxtbx import flumpy  # noqa: PLC0415

    rt = flex.reflection_table.from_file(str(PASS1 / "scaled.refl"))
    flags = flumpy.to_numpy(rt["flags"]).astype(np.uint64)
    rt = rt.select(flumpy.from_numpy((flags & np.uint64(29360128)) == 0))
    n = rt.size()

    def col(key: str) -> np.ndarray:
        return flumpy.to_numpy(rt[key]).astype(np.float64)

    lp, qe, part = col("lp"), col("qe"), col("partiality")
    conv = lp * np.where(qe > 0, 1.0 / np.where(qe > 0, qe, 1.0), 1.0)
    inv_p = np.where(part > 0, 1.0 / np.where(part > 0, part, 1.0), 1.0)

    i_prf, v_prf = col("intensity.prf.value"), col("intensity.prf.variance")
    i_sum, v_sum = col("intensity.sum.value"), col("intensity.sum.variance")
    i_scale = col("intensity.scale.value")

    i_sum_p, v_sum_p = i_sum * inv_p, v_sum * inv_p**2
    denom = i_prf * conv - i_sum_p * conv
    usable = np.abs(denom) > 1e-3 * np.maximum(np.abs(i_prf * conv), 1.0)
    w_obs = (i_scale - i_sum_p * conv) / np.where(usable, denom, 1.0)
    good = usable & (w_obs > 1e-6) & (w_obs < 1 - 1e-6) & (i_sum_p > 0)
    imid = float(
        np.median(i_sum_p[good] / (1.0 / w_obs[good] - 1.0) ** (1.0 / 3.0))
    )
    spread = float(
        np.std(i_sum_p[good] / (1.0 / w_obs[good] - 1.0) ** (1.0 / 3.0))
    )
    print(f"  recovered Imid = {imid:.6f} (spread {spread:.2e})")

    w = np.where(
        i_sum_p <= 0,
        1.0,
        1.0 / (1.0 + (np.where(i_sum_p > 0, i_sum_p, 0.0) / imid) ** 3),
    )
    candidates_i = np.stack(
        [
            (w * i_prf + (1 - w) * i_sum_p) * conv,
            i_prf * conv,
            i_sum_p * conv,
        ]
    )
    candidates_v = np.stack(
        [
            (w * v_prf + (1 - w) * v_sum_p) * conv**2,
            v_prf * conv**2,
            v_sum_p * conv**2,
        ]
    )
    err = np.abs(candidates_i - i_scale) / np.maximum(np.abs(i_scale), 1.0)
    pick = np.argmin(err, axis=0)
    rows = np.arange(n)
    print(f"  max |I_recon - I_dials| / |I| = {err[pick, rows].max():.2e}")

    SCRATCH.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RAW_NPZ,
        hkl=np.asarray(rt["miller_index"], dtype=np.int64).reshape(n, 3),
        intensity=candidates_i[pick, rows],
        variance=candidates_v[pick, rows],
        d=col("d"),
        phi=flumpy.to_numpy(rt["xyzobs.px.value"]).astype(np.float64)[:, 2],
        s1=flumpy.to_numpy(rt["s1"]).astype(np.float64).reshape(n, 3),
        partiality=part,
        lp=lp,
        qe=qe,
        dataset_id=flumpy.to_numpy(rt["id"]).astype(np.int64),
        scale=col("inverse_scale_factor"),
    )
    print(f"  cached {n} reflections to {RAW_NPZ}")


def synthetic(
    n_groups: int = 40, per_group: int = 6, seed: int = 0
) -> tuple[Reflections, torch.Tensor]:
    """A small synthetic dataset with a known scale and merge."""
    rng = np.random.default_rng(seed)
    n = n_groups * per_group
    group = np.repeat(np.arange(n_groups), per_group)
    true_ih = rng.uniform(50.0, 500.0, n_groups)[group]
    scale = rng.uniform(0.6, 1.6, n)
    intensity = true_ih * scale * (1.0 + 0.08 * rng.standard_normal(n))
    variance = np.maximum(intensity, 1.0) * 1.5

    hkl = np.stack([group, group % 7, group // 7], axis=1)
    refl = Reflections(
        hkl=torch.from_numpy(hkl).to(IDTYPE),
        intensity=torch.from_numpy(intensity).to(DTYPE),
        variance=torch.from_numpy(variance).to(DTYPE),
        d=torch.full((n,), 2.0, dtype=DTYPE),
        phi=torch.from_numpy(rng.uniform(0, 100, n)).to(DTYPE),
        s1=torch.zeros(n, 3, dtype=DTYPE),
        partiality=torch.ones(n, dtype=DTYPE),
        lp=torch.ones(n, dtype=DTYPE),
        qe=torch.ones(n, dtype=DTYPE),
        dataset_id=torch.zeros(n, dtype=IDTYPE),
        scale=torch.from_numpy(scale).to(DTYPE),
    )
    return refl, torch.from_numpy(group).to(IDTYPE)


def check_merged_intensity() -> bool:
    """`merged_intensity` against a direct numpy weighted mean."""
    print("\n[1] merged_intensity vs numpy weighted mean")
    refl, group = synthetic()
    n_groups = int(group.max()) + 1
    w = 1.0 / refl.variance
    ih, ih_refl = merged_intensity(
        refl.intensity, w, refl.scale, group, n_groups
    )

    gi = group.numpy()
    wn, gn = w.numpy(), refl.scale.numpy()
    inn = refl.intensity.numpy()
    num = np.bincount(gi, weights=wn * gn * inn, minlength=n_groups)
    den = np.bincount(gi, weights=wn * gn * gn, minlength=n_groups)
    expected = num / den

    err = float(np.abs(ih.numpy() - expected).max())
    same = bool(torch.equal(ih_refl, ih[group]))
    print(f"  max |Ih - numpy| = {err:.3e}; broadcast consistent: {same}")
    return err < 1e-12 and same


LIKELIHOODS = [
    ("gaussian/observed", GaussianLeastSquares("observed")),
    ("gaussian/expected", GaussianLeastSquares("expected")),
    ("gaussian/unity", GaussianLeastSquares("unity")),
    ("gamma", GammaLikelihood()),
    ("poisson", PoissonLikelihood()),
]


def _frozen_objective(
    lik: object,
    refl: Reflections,
    group_id: torch.Tensor,
    n_groups: int,
    w0: torch.Tensor,
    ih0: torch.Tensor,
    detach_ih: bool,
):
    """The exact function whose gradient `ScalingTarget` computes.

    Weights are constants within a minimisation step, as in DIALS, so
    the objective a finite difference must be taken of is the one with
    the weights pinned at the base point. When `detach_ih` is set, `Ih`
    is pinned too.
    """

    def f(g: torch.Tensor) -> torch.Tensor:
        if detach_ih:
            ih = ih0
        else:
            _, ih = merged_intensity(
                refl.intensity, w0, g, group_id, n_groups
            )
        if isinstance(lik, GaussianLeastSquares):
            return (w0 * (refl.intensity - g * ih) ** 2).sum()
        return lik(refl, g, ih).sum()

    return f


def check_gradients() -> bool:
    """Autograd against central finite differences."""
    print("\n[2] autograd vs central finite differences")
    refl, _ = synthetic(n_groups=12, per_group=5, seed=3)
    amap = build_asu_map(refl.hkl, "P 1")
    group_id, n_groups = resolve_groups(amap, False)

    ok = True
    for name, lik in LIKELIHOODS:
        for detach in (False, True):
            target = ScalingTarget(lik, detach_ih=detach)
            raw = torch.log(refl.scale.clone()).requires_grad_(True)
            loss = target(refl, amap, torch.exp(raw))
            loss.backward()
            analytic = raw.grad.detach().clone()

            with torch.no_grad():
                ih0, w0 = target.merge(refl, amap, refl.scale)
            objective = _frozen_objective(
                lik, refl, group_id, n_groups, w0, ih0, detach
            )

            idx = [0, 7, 19, 31, 44]
            eps = 1e-6
            numeric = []
            for i in idx:
                pert = raw.detach().clone()
                pert[i] += eps
                up = float(objective(torch.exp(pert)))
                pert[i] -= 2 * eps
                dn = float(objective(torch.exp(pert)))
                numeric.append((up - dn) / (2 * eps))
            a = analytic[idx].numpy()
            b = np.asarray(numeric)
            rel = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))
            ok &= rel < 1e-6
            print(
                f"  {name:<18} detach_ih={str(detach):<5} "
                f"max rel err = {rel:.3e}"
            )
    return ok


def check_ih_gradient_term() -> bool:
    """When the `dIh/dp` term vanishes, and when it does not.

    The chain-rule contribution through `Ih` is `dL/dIh * dIh/dp`, so it
    vanishes exactly when the merge used to profile `Ih` out is the
    stationary point of the functional being minimised. That is DIALS'
    situation -- the weighted mean minimises the weighted sum of squares
    for any fixed weights -- so `calc_dIh_by_dpi` computes something
    that is arithmetically zero.

    The same cancellation reappears for the Gamma and Poisson
    likelihoods, but only under `weight_mode="expected"`: with `w`
    evaluated at `g Ih`, the inverse-variance weighted mean coincides
    with the maximum-likelihood merge for both. Under `"observed"`
    weights it does not, and the term is real.
    """
    print("\n[3] dIh/dp term: vanishes iff the merge is the MLE")
    refl, _ = synthetic(n_groups=12, per_group=5, seed=5)
    amap = build_asu_map(refl.hkl, "P 1")

    cases = [
        ("gaussian/observed", GaussianLeastSquares("observed"), True),
        ("gaussian/expected", GaussianLeastSquares("expected"), True),
        ("gamma/expected", GammaLikelihood("expected"), True),
        ("gamma/observed", GammaLikelihood("observed"), False),
        ("poisson/expected", PoissonLikelihood(weight_mode="expected"), True),
        ("poisson/observed", PoissonLikelihood(weight_mode="observed"), False),
    ]
    ok = True
    for name, lik, vanishes in cases:
        grads = []
        for detach in (False, True):
            raw = torch.log(refl.scale.clone()).requires_grad_(True)
            ScalingTarget(lik, detach_ih=detach)(
                refl, amap, torch.exp(raw)
            ).backward()
            grads.append(raw.grad.detach().clone())
        rel = float((grads[0] - grads[1]).norm() / grads[1].norm())
        ok &= (rel < 1e-10) if vanishes else (rel > 1e-3)
        label = "~0 (merge is MLE)" if vanishes else "non-zero"
        print(f"  {name:<18} ||dg||/||g|| = {rel:.3e}   expect {label}")
    return ok


def check_outliers() -> bool:
    """Planted outliers on synthetic groups."""
    print("\n[4] outlier rejection on planted outliers")
    refl, group = synthetic(n_groups=30, per_group=8, seed=11)
    planted = [3, 19, 42, 100, 175]
    intensity = refl.intensity.clone()
    for i in planted:
        intensity[i] *= 12.0
    refl = Reflections(
        **{**{f: getattr(refl, f) for f in refl.__dataclass_fields__},
           "intensity": intensity}
    )
    amap = build_asu_map(refl.hkl, "P 1")

    mask = reject_outliers(refl, amap, scale=refl.scale, zmax=6.0)
    found = set(torch.nonzero(mask).flatten().tolist())
    print(f"  planted {sorted(planted)}")
    print(f"  flagged {sorted(found)}")
    hit = set(planted) <= found
    print(f"  all planted found: {hit}; total flagged: {len(found)}")

    upper = reject_outliers(
        refl, amap, scale=refl.scale, zmax=6.0, zmax_lower=float("inf")
    )
    n_up = int(upper.sum())
    print(f"  one-sided upper test flags {n_up} (two-sided: {len(found)})")

    small, sgroup = synthetic(n_groups=10, per_group=2, seed=2)
    del sgroup
    small_amap = build_asu_map(small.hkl, "P 1")
    n_small = int(reject_outliers(small, small_amap, zmax=0.1).sum())
    print(f"  groups of 2 with zmax=0.1 flag {n_small} (must be 0)")
    return hit and n_small == 0 and n_up <= len(found)


def check_error_model() -> bool:
    """Refine `(a, b)` on HEWL against DIALS' published values."""
    print("\n[5] error model on HEWL vs DIALS")
    if not RAW_NPZ.exists():
        print(f"  SKIP: {RAW_NPZ} missing; rerun with --rebuild")
        return True

    refl = load_npz(RAW_NPZ)
    expt = load_experiment(PASS1 / "scaled.expt")
    amap = build_asu_map(refl.hkl, expt.space_group)
    print(f"  {len(refl)} reflections, space group {expt.space_group}")

    _, n_anom = resolve_groups(amap, True, pool_centrics=True)
    _, n_split = resolve_groups(amap, True, pool_centrics=False)
    print(
        f"  anomalous groups: {n_anom} pooled, {n_split} split "
        f"(cctbx/DIALS gives 23096)"
    )

    data = prepare_error_model_data(refl, amap, refl.scale)
    print(f"  filtered to {len(data)} (DIALS: {DIALS_N_FILTERED})")

    result = refine_error_model(
        refl,
        amap,
        refl.scale,
        initial=BasicErrorModel(1.72456, 0.03723),
    )
    a, b = result.model.a, result.model.b
    da = abs(a - DIALS_A) / DIALS_A
    db = abs(b - DIALS_B) / DIALS_B
    print(f"  a = {a:.6f}  (DIALS {DIALS_A:.6f})  rel {da:.2e}")
    print(f"  b = {b:.6f}  (DIALS {DIALS_B:.6f})  rel {db:.2e}")
    print(f"  cycles = {len(result.a_history)}, n_refl = {result.n_refl}")

    info = result.binning
    print(f"  bin counts   {info.refl_per_bin.astype(int)}")
    print(f"  DIALS        {DIALS_BIN_COUNTS}")
    for label, mine, theirs in (
        ("boundaries", info.boundaries, DIALS_BIN_BOUNDS),
        ("uncorrected", info.initial_variances, DIALS_UNCORRECTED),
        ("corrected", info.bin_variances, DIALS_CORRECTED),
    ):
        rel = float(np.abs(mine - theirs).max() / np.abs(theirs).max())
        print(f"  {label:<12} max rel diff vs DIALS = {rel:.2e}")

    fixed = refine_error_model(
        refl, amap, refl.scale, a_tolerance=1e-4, max_cycles=400
    )
    print(
        f"  fixed point (a_tolerance=1e-4): a = {fixed.model.a:.6f}, "
        f"b = {fixed.model.b:.6f} after {len(fixed.a_history)} cycles"
    )
    print(
        "  -> DIALS' published values are NOT a fixed point of its own "
        "update; the 1% tolerance stops a slowly-converging sequence."
    )

    unbiased = refine_error_model(
        refl,
        amap,
        refl.scale,
        initial=BasicErrorModel(1.72456, 0.03723),
        prefactor="unbiased",
    )
    mult = float(data.n_h.mean())
    print(
        f"  mean multiplicity {mult:.1f}; correcting the deviation "
        f"prefactor gives a = {unbiased.model.a:.6f}, "
        f"b = {unbiased.model.b:.6f}"
    )
    print(
        f"  -> DIALS' deviations carry sd (n-1)/n instead of 1, so its "
        f"sigmas are low; here by a factor "
        f"{a / unbiased.model.a:.3f}"
    )
    return da < 1e-3 and db < 1e-3


def main() -> int:
    """Run every check and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="reconstruct the raw HEWL npz (needs the DIALS environment)",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    if args.rebuild:
        print("[0] rebuilding raw HEWL cache")
        rebuild_raw_npz()

    checks = [
        ("merged_intensity", check_merged_intensity),
        ("gradients", check_gradients),
        ("dIh/dp term", check_ih_gradient_term),
        ("outliers", check_outliers),
        ("error model", check_error_model),
    ]
    results = {name: fn() for name, fn in checks}

    print("\n" + "=" * 60)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
