"""Forward model and calibration inverse for a JUNGFRAU charge-integrating detector.

JUNGFRAU has no discriminator: each pixel integrates deposited charge with one of
three automatically-selected gain stages, and an ADC digitizes the result. Integer
photon counts exist in the physics but never in the electronics. The readout word is
16 bits: the top 2 encode the gain stage (00=G0, 01=G1, 11=G2), the bottom 14 the ADU.

Forward chain (`simulate`):

    N ~ Poisson(lam)                     photons absorbed in the pixel
    q = N * photon_energy_kev            deposited energy
    g = stage(N)                         comparator picks the gain stage
    adu = pedestal[g] + gain[g] * q + e  e ~ Normal(0, read_noise_adu[g])
    raw = clip(adu, 0, 2^14 - 1) | (gain_bits[g] << 14)

Calibration inverse (`calibrate`) follows psana's
`energy = (code - pedestal) / gain`, which is real-valued and can be negative on
background pixels -- the property that makes the Poisson-vs-Normal question live.

Numbers and their sources:

  * 3.6 eV per electron-hole pair in Si; a 12.4 keV photon liberates ~3500 e-.
  * Gains (41.5, -1.39, -0.11) ADU/keV are the psana defaults for G0/G1/G2. The
    negative signs are real: the low-gain stages invert, so a brighter pixel reads
    a *lower* code. PSI ships the constants with the sign baked in.
  * G0 read noise is 83 e- RMS at 10 us integration, 200 e- at 840 us.
  * Pedestals drift up to ~100 ADU (2.5 keV, ~0.2 photons) as the system settles
    thermally -- an order of magnitude above the G0 read noise, and a *bias*.
  * G0 saturates near 25 photons at 10 keV, G2 near 8000-10000.

The default pedestals below are not published per-pixel constants; they are chosen
so each stage's headroom reproduces the published dynamic ranges (checked by
`selftest.py`). Only `read_noise_adu[0]` is anchored to a measured number, which is
the one that matters: the weak-data regime this study probes never leaves G0.

Refs: https://rtd.xfel.eu/docs/jungfrau-detector-documentation/en/latest/general_introduction.html
      https://pmc.ncbi.nlm.nih.gov/articles/PMC7044001/  (Struct. Dyn. 7, 014305)
      https://confluence.slac.stanford.edu/display/PSDM/Jungfrau
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

EH_PAIR_ENERGY_EV = 3.6
GAIN_BITS = (0, 1, 3)  # 00 -> G0, 01 -> G1, 11 -> G2


@dataclass(frozen=True)
class JungfrauConfig:
    """Detector constants. Gains carry psana's sign convention (G1/G2 invert)."""

    photon_energy_kev: float = 12.4
    gain_adu_per_kev: tuple[float, float, float] = (41.5, -1.39, -0.11)
    pedestal_adu: tuple[float, float, float] = (2000.0, 15000.0, 15000.0)
    switch_photons: tuple[float, float] = (25.0, 800.0)
    read_noise_adu: tuple[float, float, float] | None = None
    enc_electrons_g0: float = 83.0
    adc_bits: int = 14

    @property
    def adu_max(self) -> int:
        return (1 << self.adc_bits) - 1

    @property
    def electrons_per_photon(self) -> float:
        return self.photon_energy_kev * 1e3 / EH_PAIR_ENERGY_EV

    @property
    def adu_per_photon(self) -> tuple[float, float, float]:
        g0, g1, g2 = self.gain_adu_per_kev
        e = self.photon_energy_kev
        return (g0 * e, g1 * e, g2 * e)

    def noise_adu(self) -> tuple[float, float, float]:
        """Read noise per stage in ADU.

        G0 is anchored to `enc_electrons_g0`. Absent published G1/G2 figures we model
        the noise as ADC-dominated, i.e. constant in ADU across stages. In photon
        units that scales with the gain ratio (~0.02 / 0.7 / 9 photons), so read noise
        is negligible against Poisson noise wherever G1/G2 are actually selected.
        """
        if self.read_noise_adu is not None:
            return self.read_noise_adu
        kev = self.enc_electrons_g0 * EH_PAIR_ENERGY_EV * 1e-3
        adu = kev * abs(self.gain_adu_per_kev[0])
        return (adu, adu, adu)

    def sigma_read_photons(self, stage: int = 0) -> float:
        """Read noise in photon-equivalent units for one stage."""
        return self.noise_adu()[stage] / abs(self.adu_per_photon[stage])

    def with_sigma_read_photons_g0(self, sigma: float) -> JungfrauConfig:
        """Config whose G0 read noise is `sigma` photons, G1/G2 held constant in ADU."""
        adu_g0 = sigma * abs(self.adu_per_photon[0])
        return replace(self, read_noise_adu=(adu_g0, adu_g0, adu_g0))


def stage_of(photons: torch.Tensor, cfg: JungfrauConfig) -> torch.Tensor:
    """Gain stage index selected by the comparator for each pixel."""
    t0, t1 = cfg.switch_photons
    return torch.bucketize(photons, torch.tensor([t0, t1], dtype=photons.dtype))


def readout(
    n_photons: torch.Tensor,
    cfg: JungfrauConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Digitize known per-pixel photon counts into packed 16-bit words."""
    stage = stage_of(n_photons, cfg)

    gain = torch.tensor(cfg.gain_adu_per_kev, dtype=n_photons.dtype)[stage]
    ped = torch.tensor(cfg.pedestal_adu, dtype=n_photons.dtype)[stage]
    sigma = torch.tensor(cfg.noise_adu(), dtype=n_photons.dtype)[stage]

    q = n_photons * cfg.photon_energy_kev
    noise = torch.randn(n_photons.shape, dtype=n_photons.dtype, generator=generator)
    adu = torch.round(ped + gain * q + sigma * noise)
    adu = adu.clamp(0, cfg.adu_max).to(torch.int64)

    bits = torch.tensor(GAIN_BITS, dtype=torch.int64)[stage]
    return adu | (bits << cfg.adc_bits)


def simulate(
    lam: torch.Tensor,
    cfg: JungfrauConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw photons from per-pixel rates and digitize them.

    Returns `(raw, n_true)`: the packed words and the ground-truth counts behind them.
    """
    n_true = torch.poisson(lam, generator=generator)
    return readout(n_true, cfg, generator=generator), n_true


def calibrate(
    raw: torch.Tensor,
    cfg: JungfrauConfig,
    pedestal_error_adu: torch.Tensor | float | tuple[float, float, float] = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert the readout to photon-equivalent units.

    `pedestal_error_adu` is added to the assumed pedestal, modelling the thermal drift
    between the dark run and the exposure. Pass a scalar to apply the same ADU error to
    every stage, or a 3-tuple to set it per stage.

    Beware what a scalar means. The published ~100 ADU drift is a *G0* number (the same
    source quotes it as 2.5 keV, which is 100/41.5 using the G0 gain). A pedestal error
    is fixed in ADU but converts to photons through 1/gain, so the same 100 ADU is
    0.19 photons in G0 and 5.8 in G1 -- a 30x difference that lands entirely on the
    brightest pixels. Whether the G1/G2 pedestals really drift by the same ADU is not
    something the published figure settles, so it is a knob rather than a constant.

    Returns `(x, stage)` where `x` is real-valued and may be negative.
    """
    adu = (raw & ((1 << cfg.adc_bits) - 1)).to(torch.get_default_dtype())
    bits = raw >> cfg.adc_bits

    stage = torch.zeros_like(bits)
    for idx, b in enumerate(GAIN_BITS):
        stage = torch.where(bits == b, torch.full_like(bits, idx), stage)

    gain = torch.tensor(cfg.gain_adu_per_kev, dtype=adu.dtype)[stage]
    ped = torch.tensor(cfg.pedestal_adu, dtype=adu.dtype)[stage]

    if isinstance(pedestal_error_adu, (tuple, list)):
        ped = ped + torch.tensor(pedestal_error_adu, dtype=adu.dtype)[stage]
    else:
        ped = ped + pedestal_error_adu

    q = (adu - ped) / gain
    return q / cfg.photon_energy_kev, stage


def to_counts(x: torch.Tensor) -> torch.Tensor:
    """Round photon-equivalents to integer counts, clamping the negatives at zero.

    The clamp is not a detail: `(adu - pedestal) / gain` goes negative on roughly half
    the background pixels, and zeroing them biases the background estimate upward
    exactly where the Poisson likelihood was supposed to win.
    """
    return torch.round(x).clamp_min(0.0)
