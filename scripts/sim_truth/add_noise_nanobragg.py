"""Add background and detector noise to a datsimx noisy copy with nanoBragg's own add_noise, PSF included.

This is the noise call of datsimx `mx_simulate.py` without `--noNoise`, and of
the laue-dials paper's `sim_laue.py`, applied after the fact so the noiseless
truth stays beside it:

    SIM = nanoBragg(detector, beam)      # from geom_run<run>.expt
    SIM.raw_pixels = B + background      # photons
    SIM.add_noise()                      # Poisson, calibration, PSF, gain, ADC, readout (nanoBragg C order)

`braggaddnoise` re-implements the same steps in numpy without the PSF; the
paper's images had one (0.1 mm FWHM, 5 px kernel radius), and it is what
lowers their per-pixel shot-noise variance to 0.09 of Poisson.
Nothing of datsimx, easyBragg or nanoBragg is modified; this only calls it.

Seeds: Poisson and readout seed = 100000 * run + shot number; calibration seed
= 100000 * run, the same for every shot of a run (a fixed per-pixel response).

usage: python add_noise_nanobragg.py NOISY_DIR --run 1 [--gain 0.7 --adc 10 --readout 3
                                     --calib-pct 3 --psf-mm 0.1 --psf-radius 5]
Overwrites `sim_image` in NOISY_DIR/shots_<run>_*.h5 in place: run it on the copy only.
"""

from __future__ import annotations

import argparse
import glob
import os

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("noisy_dir")
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--gain", type=float, default=0.7)
    ap.add_argument("--adc", type=float, default=10.0)
    ap.add_argument("--readout", type=float, default=3.0)
    ap.add_argument("--calib-pct", type=float, default=3.0)
    ap.add_argument("--psf-mm", type=float, default=0.1)
    ap.add_argument("--psf-radius", type=int, default=5)
    a = ap.parse_args()

    from dxtbx.model.experiment_list import ExperimentListFactory
    from scitbx.array_family import flex
    from simtbx.nanoBragg import nanoBragg

    d = a.noisy_dir
    el = ExperimentListFactory.from_json_file(os.path.join(d, f"geom_run{a.run}.expt"), check_format=False)
    det, beam = el[0].detector, el[0].beam
    with h5py.File(os.path.join(d, f"background_{a.run}.h5"), "r") as f:
        bg = np.clip(f["background"][()].astype(np.float64), 0, None)

    for path in sorted(glob.glob(os.path.join(d, f"shots_{a.run}_*.h5"))):
        with h5py.File(path, "r+") as f:
            for name in sorted(f["sim_image"].keys()):
                shot = int(name.rsplit("_", 1)[1])
                bragg = np.clip(f[f"sim_image/{name}"][()].astype(np.float64), 0, None)
                SIM = nanoBragg(detector=det, beam=beam)
                SIM.adc_offset_adu = a.adc
                SIM.detector_psf_kernel_radius_pixels = a.psf_radius
                SIM.detector_calibration_noise_pct = a.calib_pct
                SIM.detector_psf_fwhm_mm = a.psf_mm
                SIM.quantum_gain = a.gain
                SIM.readout_noise_adu = a.readout
                SIM.seed = 100000 * a.run + shot
                SIM.calib_seed = 100000 * a.run
                SIM.raw_pixels = flex.double((bragg + bg).ravel())
                SIM.add_noise()
                img = SIM.raw_pixels.as_numpy_array().reshape(bragg.shape).astype(np.float32)
                SIM.free_all()
                del f[f"sim_image/{name}"]
                f.create_dataset(f"sim_image/{name}", data=img, compression="gzip", compression_opts=4, shuffle=True)
                print(f"{name}: ADU min {img.min():.1f}, median {np.median(img):.1f}, max {img.max():.1f}")


if __name__ == "__main__":
    main()
