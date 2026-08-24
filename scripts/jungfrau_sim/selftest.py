"""Consistency checks for the JUNGFRAU forward/inverse model.

Run:  uv run python scripts/jungfrau_sim/selftest.py
"""

import torch

from detector import JungfrauConfig, calibrate, readout, simulate, stage_of, to_counts

torch.set_default_dtype(torch.float64)


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    return ok


def main() -> None:
    cfg = JungfrauConfig()
    passed = []

    print("derived constants")
    print(f"  e- per {cfg.photon_energy_kev} keV photon : {cfg.electrons_per_photon:.0f}")
    print(f"  ADU per photon (G0/G1/G2)     : {tuple(round(a, 2) for a in cfg.adu_per_photon)}")
    print(f"  read noise ADU                : {tuple(round(a, 2) for a in cfg.noise_adu())}")
    print(
        f"  read noise photons            : "
        f"{tuple(round(cfg.sigma_read_photons(g), 3) for g in range(3))}"
    )

    print("\npublished-number agreement")
    passed.append(
        check(
            "12.4 keV photon liberates ~3500 e-",
            3400 <= cfg.electrons_per_photon <= 3550,
            f"got {cfg.electrons_per_photon:.0f}",
        )
    )
    passed.append(
        check(
            "G0 read noise is a few % of a photon",
            0.01 <= cfg.sigma_read_photons(0) <= 0.05,
            f"got {cfg.sigma_read_photons(0):.4f} photons",
        )
    )

    # Headroom per stage: how many photons fit between pedestal and the ADC rail.
    print("\ndynamic range implied by pedestal + gain")
    for g, (lo, hi) in enumerate([(20, 30), (700, 1200), (8000, 12000)]):
        ped, gain = cfg.pedestal_adu[g], cfg.adu_per_photon[g]
        rail = cfg.adu_max if gain > 0 else 0.0
        headroom = abs(rail - ped) / abs(gain)
        passed.append(
            check(
                f"G{g} saturates in [{lo}, {hi}] photons",
                lo <= headroom <= hi,
                f"got {headroom:.0f}",
            )
        )

    print("\nnoiseless round-trip (photons -> raw -> photons)")
    noiseless = JungfrauConfig(read_noise_adu=(0.0, 0.0, 0.0))
    n = torch.tensor([0.0, 1.0, 5.0, 24.0, 26.0, 100.0, 799.0, 900.0, 5000.0])
    x, stage = calibrate(readout(n, noiseless), noiseless)
    passed.append(
        check(
            "stage decoded from gain bits matches the one written",
            torch.equal(stage, stage_of(n, noiseless)),
        )
    )
    # ADC rounding costs 1/(ADU per photon); G2 is coarse by design.
    err = (x - n).abs()
    tol = torch.tensor([1.0 / abs(noiseless.adu_per_photon[g]) for g in stage]) * 1.01
    passed.append(
        check(
            "photon-equivalents recovered to ADC quantization",
            bool((err <= tol).all()),
            f"max err {err.max():.2e} photons",
        )
    )

    print("\nnoisy behaviour at background level (the regime that matters)")
    g = torch.Generator().manual_seed(0)
    lam = torch.full((200_000,), 0.5)
    raw, n_true = simulate(lam, cfg, generator=g)
    x, stage = calibrate(raw, cfg)
    frac_neg = (x < 0).double().mean().item()
    exact = (to_counts(x) == n_true).double().mean().item()
    passed.append(check("all background pixels sit in G0", bool((stage == 0).all())))
    passed.append(
        check(
            "rounding recovers the true count almost always at real G0 noise",
            exact > 0.99,
            f"{100 * exact:.2f}% exact",
        )
    )
    print(f"   -> {100 * frac_neg:.1f}% of background pixels calibrate negative")

    print("\npedestal drift vs read noise")
    drift_photons = 100.0 / abs(cfg.adu_per_photon[0])
    passed.append(
        check(
            "100 ADU drift is ~0.2 photons and >> G0 read noise",
            0.15 <= drift_photons <= 0.25
            and drift_photons > 5 * cfg.sigma_read_photons(0),
            f"drift {drift_photons:.3f} vs noise {cfg.sigma_read_photons(0):.3f} photons",
        )
    )

    print("\nexact likelihood (underpins every claim, so check it hard)")
    import likelihoods as lk

    # It must be a normalized density in x for each lam.
    sigma = 0.3
    grid = torch.linspace(-6.0, 40.0, 240_001)
    for lam_v in (0.1, 0.5, 2.0, 8.0):
        lam = torch.full_like(grid, lam_v)
        dens = lk.exact(grid, lam, sigma).exp()
        mass = torch.trapezoid(dens, grid)
        passed.append(
            check(f"integrates to 1 at lam={lam_v}", abs(float(mass) - 1.0) < 1e-6,
                  f"got {float(mass):.8f}")
        )

    # Mean and variance must be lam and lam + sigma^2.
    lam_v = 3.0
    lam = torch.full_like(grid, lam_v)
    dens = lk.exact(grid, lam, sigma).exp()
    mean = torch.trapezoid(dens * grid, grid)
    var = torch.trapezoid(dens * (grid - mean) ** 2, grid)
    passed.append(check("mean == lam", abs(float(mean) - lam_v) < 1e-6, f"got {float(mean):.6f}"))
    passed.append(
        check(
            "var == lam + sigma^2",
            abs(float(var) - (lam_v + sigma**2)) < 1e-6,
            f"got {float(var):.6f} vs {lam_v + sigma**2:.6f}",
        )
    )

    # As sigma -> 0 the convolution collapses onto the Poisson pmf at integer x, up to
    # the density-vs-pmf normalization: a Normal density at its own mean is 1/(s*sqrt(2pi)),
    # which diverges. Subtract that off and the pmf must reappear exactly.
    import math

    n_int = torch.arange(0.0, 12.0)
    lam = torch.full_like(n_int, 2.5)
    tiny = 1e-4
    offset = -math.log(tiny) - 0.5 * math.log(2.0 * math.pi)
    diff = (lk.exact(n_int, lam, tiny) - offset - lk.poisson_counts(n_int, lam)).abs().max()
    passed.append(
        check("collapses to Poisson as sigma -> 0", float(diff) < 1e-8, f"max |d| {float(diff):.2e}")
    )

    print(f"\n{sum(passed)}/{len(passed)} checks passed")
    raise SystemExit(0 if all(passed) else 1)


if __name__ == "__main__":
    main()
