from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.spatial import cKDTree


def _tqdm(iterable, **kwargs):
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def _process_one_image(work):
    """Compute neighbor overlap for the reflections of a single image.

    Args:
        work: tuple (idx, cx, cy, bx0, by0, H, W) where idx are the global
            reflection indices on this image, (cx, cy) their predicted
            centroids in detector pixels, (bx0, by0) their window origins,
            and (H, W) the fixed window shape.

    Returns:
        List of (global_idx, local_mask) tuples, one per reflection that has
        any overlapped pixel. `local_mask` is a (H, W) bool array.
    """
    idx, cx, cy, bx0, by0, H, W = work
    n = len(idx)
    if n < 2:
        return []

    tree_2d = cKDTree(np.column_stack([cx, cy]))
    max_r = float(np.sqrt(W * W + H * H))
    pairs = tree_2d.query_pairs(r=max_r, output_type="ndarray")
    if len(pairs) == 0:
        return []

    ii = pairs[:, 0]
    jj = pairs[:, 1]

    ix0 = np.maximum(bx0[ii], bx0[jj])
    ix1 = np.minimum(bx0[ii] + W, bx0[jj] + W)
    iy0 = np.maximum(by0[ii], by0[jj])
    iy1 = np.minimum(by0[ii] + H, by0[jj] + H)

    has_overlap = (ix0 < ix1) & (iy0 < iy1)
    pairs_ov = np.where(has_overlap)[0]
    if len(pairs_ov) == 0:
        return []

    # Accumulate overlap per local reflection.
    local_overlap = np.zeros((n, H, W), dtype=bool)

    for p in pairs_ov:
        i_local, j_local = ii[p], jj[p]
        ox0, ox1 = int(ix0[p]), int(ix1[p])
        oy0, oy1 = int(iy0[p]), int(iy1[p])

        xs = np.arange(ox0, ox1, dtype=np.float32)
        ys = np.arange(oy0, oy1, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")

        d_i = (xx - cx[i_local]) ** 2 + (yy - cy[i_local]) ** 2
        d_j = (xx - cx[j_local]) ** 2 + (yy - cy[j_local]) ** 2
        j_wins = d_j < d_i

        i_y0, i_x0 = int(by0[i_local]), int(bx0[i_local])
        local_overlap[
            i_local,
            oy0 - i_y0 : oy1 - i_y0,
            ox0 - i_x0 : ox1 - i_x0,
        ] |= j_wins

        j_y0, j_x0 = int(by0[j_local]), int(bx0[j_local])
        local_overlap[
            j_local,
            oy0 - j_y0 : oy1 - j_y0,
            ox0 - j_x0 : ox1 - j_x0,
        ] |= ~j_wins

    affected = local_overlap.any(axis=(1, 2))
    results = []
    for k in np.where(affected)[0]:
        results.append((idx[k], local_overlap[k]))
    return results


def compute_overlap_mask(
    bboxes,
    centroids,
    D,
    H,
    W,
    nproc=1,
    image_ids=None,
):
    """Compute the per-pixel neighbor-overlap mask for every reflection.

    Args:
        bboxes: (N, 6) int array of (x0, x1, y0, y1, z0, z1) per reflection.
        centroids: (N, 3) float array of predicted (x, y, z) in pixels.
        D: window depth (frames).
        H: window height (pixels).
        W: window width (pixels).
        nproc: number of parallel workers.
        image_ids: (N,) int array used to group reflections per image. If
            None, groups by bbox z0 (`bboxes[:, 4]`). For single-frame stills
            (all z0 == 0) pass image_ids explicitly, otherwise every
            reflection lands in one group and cross-image overlaps are bogus.

    Returns:
        (N, D, H, W) bool array, True at pixels within a reflection's window
        that are owned by another reflection.
    """
    N = len(bboxes)
    overlap = np.zeros((N, D, H, W), dtype=bool)

    if image_ids is not None:
        group_keys = np.asarray(image_ids, dtype=np.int64)
    else:
        group_keys = bboxes[:, 4]
    unique_groups = np.unique(group_keys)

    # One work item per image.
    work = []
    for g in unique_groups:
        idx = np.where(group_keys == g)[0]
        if len(idx) < 2:
            continue
        cx = centroids[idx, 0].astype(np.float32)
        cy = centroids[idx, 1].astype(np.float32)
        bx0 = bboxes[idx, 0].astype(np.float32)
        by0 = bboxes[idx, 2].astype(np.float32)
        work.append((idx, cx, cy, bx0, by0, H, W))

    print(f"  overlap: {len(work)} images to process, {nproc} workers")

    if nproc <= 1:
        for w in _tqdm(work, desc="overlap detection"):
            for global_idx, mask_2d in _process_one_image(w):
                overlap[global_idx] |= mask_2d
    else:
        with ProcessPoolExecutor(max_workers=nproc) as executor:
            futures = {executor.submit(_process_one_image, w): w for w in work}
            for f in _tqdm(
                as_completed(futures),
                total=len(futures),
                desc="overlap detection",
            ):
                for global_idx, mask_2d in f.result():
                    overlap[global_idx] |= mask_2d

    return overlap
