"""Add the columns `laue.optimize_indexing` needs to a stills spot list.

In the ordinary laue-dials pipeline the `s1` column arrives from `laue.index`,
which maps centroids into reciprocal space through a monochromatic indexing
solution, and `laue.sequence_to_stills` carries it onto the stills. Neither step
can run here: dxtbx resolves these CBFs to `FormatCBFFullStill`, so there is no
sweep for `dials.index` to work with.

Nothing is lost by computing it directly, because `s1` is not a fitted quantity.
`LaueAssigner` normalizes both `s0` and `s1` to unit vectors (`laue.py`,
`LaueBase.__init__` and `LaueAssigner.__init__`), so only the DIRECTION from the
sample to the observed pixel is used, and that is fixed by the detector model and
the centroid alone -- no crystal, no wavelength, no indexing.

Writes the four columns the assigner and the predictor read: `s1`, and
`imageset_id`/`xyzobs.mm.value`/`panel` if spotfinding left any of them out.
"""

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("expt")
    ap.add_argument("refl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentListFactory

    el = ExperimentListFactory.from_json_file(a.expt, check_format=False)
    refls = flex.reflection_table.from_file(a.refl)
    print(f"{len(refls):,} spots over {len(el)} experiments")
    print("columns in:", sorted(refls.keys()))

    ids = refls["id"].as_numpy_array()
    px = refls["xyzobs.px.value"].as_numpy_array()
    panel = (refls["panel"].as_numpy_array() if "panel" in refls
             else np.zeros(len(refls), dtype=int))

    s1 = np.zeros((len(refls), 3))
    mm = np.zeros((len(refls), 3))
    for i in np.unique(ids):
        sel = ids == i
        det = el[int(i)].detector
        for p in np.unique(panel[sel]):
            q = sel & (panel == p)
            pan = det[int(p)]
            # lab coordinate of the observed centroid, then the unit direction
            # from the sample; the assigner renormalizes anyway
            lab = np.array([pan.get_pixel_lab_coord((x, y))
                            for x, y in px[q][:, :2]])
            s1[q] = lab / np.linalg.norm(lab, axis=1)[:, None]
            xy = np.array([pan.pixel_to_millimeter((x, y))
                           for x, y in px[q][:, :2]])
            mm[q, :2] = xy

    # flex.double rejects a non-contiguous array, and a column slice of an
    # (n, 3) array is never contiguous
    def _v3(a):
        return flex.vec3_double(
            flex.double(np.ascontiguousarray(a[:, 0])),
            flex.double(np.ascontiguousarray(a[:, 1])),
            flex.double(np.ascontiguousarray(a[:, 2])),
        )

    refls["s1"] = _v3(s1)
    if "imageset_id" not in refls:
        refls["imageset_id"] = refls["id"]
    if "xyzobs.mm.value" not in refls:
        refls["xyzobs.mm.value"] = _v3(mm)
    if "xyzobs.mm.variance" not in refls:
        v = refls["xyzobs.px.variance"].as_numpy_array()
        ps = el[0].detector[0].get_pixel_size()
        refls["xyzobs.mm.variance"] = _v3(
            np.stack([v[:, 0] * ps[0] ** 2,
                      v[:, 1] * ps[1] ** 2,
                      np.zeros(len(refls))], axis=1))

    per = np.bincount(ids, minlength=len(el))
    print(f"spots per image: min {per.min()} median {int(np.median(per))} "
          f"max {per.max()}  images with 0: {(per == 0).sum()}")
    print("columns out:", sorted(refls.keys()))
    refls.as_file(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
