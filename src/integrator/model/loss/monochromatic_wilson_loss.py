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
        #init_log_G is captured directly.
        init_log_G: float = 1000.0,
        lp_correction: bool = False,
        #init_log_B goes inside kwargs
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.init_log_G = init_log_G
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
                float(self.init_log_G),
            )

    def get_wilson_stats(self) -> dict[str, Tensor]:
        """Return summary statistics for the learned Wilson B/G values
            it reads the current (trained) values and summarizes them

            So the structure is:
            monochromatic_wilson_loss.py -> calculates B/G statistics
            WilsonParamLogger callback -> calls get_wilson_stats() every epoch
            train.py -> already registers the callback  
        """
        with torch.no_grad():
            if self.image_level_wilson:
                #torch.arange: creates a tensor of integers from 0 to n - 1.
                image_id = torch.arange(self.n_images, device= self.raw_B_by_image.weight.device,)

                B, G = self._get_image_B_G(image_id)

            else:
                B = self.get_B().reshape(1)
                G = self.get_G().reshape(1)
            return{
                "B_mean": B.mean(),
                #population standard deviation formula: divide by N,
                #if = True means: sample standard deviation formula: divide by N - 1
                "B_std": B.std(unbiased=False),                                             
                "B_min": B.min(),
                "B_max": B.max(),
                "G_mean":G.mean(),
                "G_std": G.std(unbiased=False),
                "G_min": G.min(),
                "G_max": G.max(),
            }
                

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

            B, G = self._get_image_B_G(image_id)
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


    def _get_image_B_G(self, image_id: Tensor,) -> tuple[Tensor, Tensor]:
        raw_B = self.raw_B_by_image(image_id).squeeze(-1)
        raw_G = self.raw_G_by_image(image_id).squeeze(-1)

        B = F.softplus(raw_B) + self.b_min
        G = F.softplus(raw_G)

        return B, G
        
        

