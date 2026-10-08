"""Connect shoebox batches to the nanoBragg kernel.

A shoebox row holds `h * w` pixels flattened slow-major from the window origin
`(bbox.0, bbox.2)` on the panel, and its image number keys the crystal setting
matrix of the shot it came from. These helpers turn a batch's metadata and the
`physics` block of `dataset.yaml` into the pixel indices, panel and per-image
crystals that `render_pixels` takes.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from integrator.physics.nanobragg import Detector


def shoebox_pixel_indices(metadata: dict, h: int, w: int) -> tuple[Tensor, Tensor]:
    """Fast and slow panel indices of every shoebox pixel, each shape (B, h * w)."""
    x0 = metadata["bbox.0"].reshape(-1, 1).round().long()
    y0 = metadata["bbox.2"].reshape(-1, 1).round().long()
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return x0 + xs.reshape(1, -1).to(x0.device), y0 + ys.reshape(1, -1).to(y0.device)


def detector_from_spec(physics: dict, dtype=torch.float64) -> Detector:
    """Panel geometry from the `physics` block of `dataset.yaml`, in millimetres."""
    det = physics["detector"]
    fdet = torch.tensor(det["fast"], dtype=dtype)
    sdet = torch.tensor(det["slow"], dtype=dtype)
    odet = torch.linalg.cross(fdet, sdet)
    pix0 = torch.tensor(det["pix0_mm"], dtype=dtype)
    if torch.dot(pix0, odet) < 0:
        odet = -odet
    return Detector(pix0=pix0, fdet=fdet, sdet=sdet, odet=odet, pixel_size=float(det["pixel_size_mm"]))


@dataclass
class CrystalModels:
    """Per-image crystal setting matrices, looked up by image number."""

    image_num: Tensor
    A: Tensor
    """Setting matrices with columns `a*, b*, c*`, inverse Angstrom, shape (M, 3, 3)."""
    mos_spread_deg: Tensor

    @classmethod
    def load(cls, data_dir: str | Path, physics: dict, dtype=torch.float64) -> "CrystalModels":
        z = np.load(Path(data_dir) / physics["orientation_file"])
        return cls(
            image_num=torch.as_tensor(z["image_num"]).long(),
            A=torch.as_tensor(z["A"], dtype=dtype),
            mos_spread_deg=torch.as_tensor(z["mos_spread_deg"], dtype=dtype),
        )

    def index(self, image_num: Tensor) -> Tensor:
        """Row of each image number in `A`, shape like `image_num`."""
        order = torch.argsort(self.image_num)
        pos = torch.searchsorted(self.image_num[order], image_num.long().reshape(-1))
        found = order[pos.clamp(max=len(order) - 1)]
        if not torch.equal(self.image_num[found], image_num.long().reshape(-1)):
            raise KeyError("image numbers in the batch are missing from the crystal models")
        return found.reshape(image_num.shape)

    def cells(self, image_num: Tensor) -> Tensor:
        """Real-space cells (rows `a, b, c`, Angstrom) for each image, shape (..., 3, 3)."""
        return torch.linalg.inv(self.A[self.index(image_num)])

    def recip(self, image_num: Tensor) -> Tensor:
        """Reciprocal vectors as rows `a*, b*, c*`, shape (..., 3, 3)."""
        return self.A[self.index(image_num)].transpose(-1, -2)
