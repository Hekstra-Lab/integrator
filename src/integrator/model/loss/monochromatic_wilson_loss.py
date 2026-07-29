import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from integrator.model.loss.wilson_loss import WilsonLoss


class MonochromaticWilsonLoss(WilsonLoss):
    """Wilson loss for monochromatic data with optional per-image B/G."""

    def __init__(
        self,
        *,
        init_log_G: float = 0.0,
        lp_correction: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._apply_lp = lp_correction

        # Global G parameter used when image_level_wilson=False.
        self.raw_G = nn.Parameter(torch.tensor(float(init_log_G)))

        # Optional per-image B/G parameters.
        # Requires WilsonLoss.__init__ to define:
        #   self.image_level_wilson
        #   self.n_images
        #   self.init_log_B
        if self.image_level_wilson:
            if self.n_images is None:
                raise ValueError(
                    "image_level_wilson=True requires n_images."
                )

            self.raw_B_by_image = nn.Embedding(self.n_images, 1)
            self.raw_G_by_image = nn.Embedding(self.n_images, 1)

            nn.init.constant_(
                self.raw_B_by_image.weight,
                float(self.init_log_B),
            )
            nn.init.constant_(
                self.raw_G_by_image.weight,
                float(init_log_G),
            )

    def get_G(self) -> Tensor:
        return F.softplus(self.raw_G)

    def _get_tau(
        self,
        metadata: dict,
        s_sq: Tensor,
        device: torch.device,
    ) -> Tensor:
        if self.image_level_wilson:
            if "image_id" not in metadata:
                raise ValueError(
                    "image_level_wilson=True requires metadata['image_id']."
                )

            image_id = metadata["image_id"].to(device).long()

            B = (
                F.softplus(self.raw_B_by_image(image_id)).squeeze(-1)
                + self.b_min
            )
            G = F.softplus(self.raw_G_by_image(image_id)).squeeze(-1)

        else:
            B = self.get_B()
            G = self.get_G()

        tau = (1.0 / G.clamp(min=self.eps)) * torch.exp(2.0 * B * s_sq)

        if self._apply_lp:
            if "lp" not in metadata:
                raise ValueError("lp_correction=True requires metadata['lp'].")
            lp = metadata["lp"].to(device).clamp(min=1e-8)
            tau = tau * lp

        return tau