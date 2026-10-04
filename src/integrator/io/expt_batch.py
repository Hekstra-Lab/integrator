"""Reconstruct the laue-dials `BATCH` convention from a DIALS experiment list.

laue-dials writes `BATCH = refls["id"] + 1` (`laue_dials/command_line/integrate.py`),
that is the **dense index** of the experiment in the list handed to `laue.integrate`.
Our shoebox metadata instead stores `image_num`, the trailing integer of the image
filename minus one (refltorch's `_path_to_image_num`), which is the **true image
number** and stays stable when upstream refinement drops experiments.

The two agree only when nothing was dropped. When something was, laue-dials'
index compresses and ours does not, so the offset drifts along the sweep. On PDZ2
it drifts from -4 to +3 depending on the sweep, worst in the largest pass.

That silently breaks two things downstream, which is why this matters more than a
cosmetic column difference:

- `careless` takes `BATCH` as a scaling metadata key, so mismatched labels pool
  observations from physically different images under one per-image scale.
- The laue-dials tutorial's frame cuts are expressed **by BATCH**
  (`BATCH <= 85` for one pass, `<= 95` for another, plus an RMSD drop list read
  from `rmsds.log`), so a shifted `BATCH` cuts the wrong frames.

The experiment list is the authority for the map, so it is rebuilt from there. The
`.expt` file is JSON, so it is parsed directly and DIALS is not required (DIALS is
only installed in the `integrator-dev` environment).
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# same convention as refltorch's `_path_to_image_num`, so the two agree by
# construction rather than by coincidence
_TRAILING_INT_RE = re.compile(r"_(\d+)\.[A-Za-z0-9]+$")


def image_number_from_path(path: str) -> int:
    """Image number of a laue-dials image filename, zero-indexed.

    `e35f_off_001.mccd` -> 0, matching refltorch's extraction so that the numbers
    in shoebox metadata and the numbers derived here are the same quantity.

    Raises:
        ValueError: if the filename carries no trailing `_<digits>.<ext>`.
    """
    m = _TRAILING_INT_RE.search(path)
    if m is None:
        raise ValueError(f"could not parse an image number from filename: {path}")
    return int(m.group(1)) - 1


def batch_map_from_expt(expt_path: str | Path) -> dict[int, int]:
    """Map `image_num` to the `BATCH` laue-dials would write for it.

    Args:
        expt_path: A DIALS `.expt` JSON, the same one passed to `laue.integrate`
            (for the reference pipeline, `poly_refined.expt`).

    Returns:
        `{image_num: experiment_index + 1}`, one entry per experiment.

    Raises:
        ValueError: if two experiments resolve to the same image number, which
            would make the map ambiguous.
    """
    doc = json.loads(Path(expt_path).read_text())
    imagesets = doc.get("imageset", [])
    out: dict[int, int] = {}
    for expt_index, expt in enumerate(doc.get("experiment", [])):
        images = imagesets[expt["imageset"]].get("images") or []
        if not images:
            raise ValueError(
                f"experiment {expt_index} in {expt_path} has no image path; "
                "BATCH cannot be reconstructed"
            )
        num = image_number_from_path(images[0])
        if num in out:
            raise ValueError(
                f"two experiments in {expt_path} resolve to image_num {num}; "
                "the BATCH map would be ambiguous"
            )
        out[num] = expt_index + 1
    logger.info(
        "BATCH map from %s: %d experiments, image_num %d-%d",
        Path(expt_path).name,
        len(out),
        min(out) if out else -1,
        max(out) if out else -1,
    )
    return out
