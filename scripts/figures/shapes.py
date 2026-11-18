"""Shape statistics of a profile image, shared by the figure scripts.

The orientation comes from the central second-moment tensor and the agreement
between two shapes from the modulus of their Fourier transform with the DC
term dropped. Both are translation invariant, which is the point: comparing
profiles by cosine similarity reads back 0.42 to 0.65 on this data, which is
the chance level for nonnegative vectors and not a result.

The axial orientation is defined modulo 180 degrees, so two unrelated
directions differ by 45 degrees on average, not 90.
"""

import numpy as np


def moments(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(theta, axis_ratio)` of each image in a stack `(n, h, w)`.

    `theta` is the direction of the major axis in radians, `axis_ratio` is
    `sqrt(l1 / l2)` of the central second-moment tensor.
    """
    a = np.asarray(img, dtype=np.float64)
    n, h, w = a.shape
    f = np.maximum(a.reshape(n, -1), 0.0)
    tot = f.sum(1)
    safe = np.where(tot > 0, tot, 1.0)
    yy, xx = np.mgrid[0:h, 0:w]
    y, x = yy.ravel().astype(np.float64), xx.ravel().astype(np.float64)
    cy, cx = f @ y / safe, f @ x / safe
    dy, dx = y[None] - cy[:, None], x[None] - cx[:, None]
    myy = (f * dy * dy).sum(1) / safe
    mxx = (f * dx * dx).sum(1) / safe
    mxy = (f * dx * dy).sum(1) / safe
    theta = 0.5 * np.arctan2(2 * mxy, mxx - myy)
    tr, det = mxx + myy, mxx * myy - mxy * mxy
    disc = np.maximum(tr * tr / 4.0 - det, 0.0)
    l1 = tr / 2.0 + np.sqrt(disc)
    l2 = np.maximum(tr / 2.0 - np.sqrt(disc), 1e-12)
    return theta, np.sqrt(l1 / l2)


def dtheta_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Axial difference in degrees between two orientation arrays."""
    d = np.degrees(a - b) % 180.0
    return np.minimum(d, 180.0 - d)


def dtheta_images(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Orientation difference between each image and its reference image."""
    ta, _ = moments(img)
    tb, _ = moments(ref)
    return dtheta_deg(ta, tb)
