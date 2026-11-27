"""Dump a whole-detector view and a full-resolution crop of simulated Laue shots, for fig_sim_image.py.

Each panel is in photons, (ADU - adc) / gain. The whole-detector view is the
mean over POOL x POOL blocks; the crop is CROP x CROP pixels at full resolution,
its centre offset by --offset pixels (fast, slow) from the beam centre, so two
detectors with the same angle per pixel show the same part of reciprocal space.

usage:
  python dump_image_view.py OUT.npz \\
      --paper NAME SHOT.cbf \\
      --datsimx NAME NOISY_DIR SHOT \\
      [--truth NAME TRUTH_DIR SHOT]          # noiseless Bragg, photons
      [--offset 500 500 --crop 400 --pool 8 --gain 0.7 --adc-paper 10 --adc-datsimx 10]
      [--centre X Y]                         # crop centred on this pixel instead
For datsimx photon-counting noise (noisy_pc) pass --gain 1 --adc-datsimx 0: those
shots are stored in photons.
Any of --paper/--datsimx/--truth may be repeated; panel order is the order given.
Runs in the simtbx environment.
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np


def beam_centre(panel, beam):
    return panel.get_beam_centre_px(beam.get_s0())


def load_paper(cbf, gain, adc):
    import dxtbx

    img = dxtbx.load(cbf)
    ph = (img.get_raw_data().as_numpy_array().astype(np.float64) - adc) / gain
    return ph, beam_centre(img.get_detector()[0], img.get_beam())


def load_datsimx(d, shot, gain, adc, noiseless=False):
    from dxtbx.model.experiment_list import ExperimentListFactory

    run = shot.split("_")[1]
    el = ExperimentListFactory.from_json_file(os.path.join(d, f"geom_run{run}.expt"), check_format=False)
    with h5py.File(os.path.join(d, f"shots_{run}_rank0.h5"), "r") as f:
        x = f[f"sim_image/{shot}"][()].astype(np.float64)
    ph = x if noiseless else (x - adc) / gain
    return ph, beam_centre(el[0].detector[0], el[0].beam)


def views(ph, bc, offset, crop, pool, centre=None):
    ny, nx = ph.shape
    py, px = ny // pool * pool, nx // pool * pool
    pooled = ph[:py, :px].reshape(py // pool, pool, px // pool, pool).mean(axis=(1, 3))
    if centre is not None:
        cx, cy = int(round(centre[0])), int(round(centre[1]))
    else:
        cx, cy = int(round(bc[0] + offset[0])), int(round(bc[1] + offset[1]))
    h = crop // 2
    return pooled.astype(np.float32), ph[cy - h:cy + h, cx - h:cx + h].astype(np.float32), (cx, cy)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--paper", nargs=2, action="append", default=[], metavar=("NAME", "CBF"))
    ap.add_argument("--datsimx", nargs=3, action="append", default=[], metavar=("NAME", "NOISY_DIR", "SHOT"))
    ap.add_argument("--truth", nargs=3, action="append", default=[], metavar=("NAME", "TRUTH_DIR", "SHOT"))
    ap.add_argument("--offset", nargs=2, type=float, default=[500, 500])
    ap.add_argument("--crop", type=int, default=400)
    ap.add_argument("--centre", nargs=2, type=float, default=None, metavar=("X", "Y"),
                    help="crop centre in pixels (fast, slow); default beam centre + --offset")
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--gain", type=float, default=0.7)
    ap.add_argument("--adc-paper", type=float, default=10.0)
    ap.add_argument("--adc-datsimx", type=float, default=10.0)
    a = ap.parse_args()

    out, names = {}, []
    for name, cbf in a.paper:
        ph, bc = load_paper(cbf, a.gain, a.adc_paper)
        out[f"{name}_kind"] = np.array("noisy")
        names.append(name)
        out[f"{name}_pooled"], out[f"{name}_crop"], c = views(ph, bc, a.offset, a.crop, a.pool, a.centre)
        out[f"{name}_centre"] = np.array(c)
    for name, d, shot in a.datsimx:
        ph, bc = load_datsimx(d, shot, a.gain, a.adc_datsimx)
        out[f"{name}_kind"] = np.array("noisy")
        names.append(name)
        out[f"{name}_pooled"], out[f"{name}_crop"], c = views(ph, bc, a.offset, a.crop, a.pool, a.centre)
        out[f"{name}_centre"] = np.array(c)
    for name, d, shot in a.truth:
        ph, bc = load_datsimx(d, shot, a.gain, 0.0, noiseless=True)
        out[f"{name}_kind"] = np.array("noiseless")
        names.append(name)
        out[f"{name}_pooled"], out[f"{name}_crop"], c = views(ph, bc, a.offset, a.crop, a.pool, a.centre)
        out[f"{name}_centre"] = np.array(c)
    out["names"] = np.array(names)
    out["pool"], out["crop"] = np.array(a.pool), np.array(a.crop)
    if a.centre is not None:
        out["centre_xy"] = np.array(a.centre)
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out}: {names}")


if __name__ == "__main__":
    main()
