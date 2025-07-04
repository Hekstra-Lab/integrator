import torch
import torch.nn as nn

from integrator.model.integrators import BaseIntegrator


class Integrator(BaseIntegrator):
    def __init__(
        self,
        qbg,
        qp,
        qI,
        intensity_encoder,
        profile_encoder,
        loss_fn,
        mc_samples: int = 100,
        lr: float = 1e-3,
        max_iterations: int = 4,
        renyi_scale: float = 0.00,
        d: int = 3,
        h: int = 21,
        w: int = 21,
        weight_decay=1e-8,
    ):
        """
        Args:
            qbg ():
            qp ():
            qI ():
            intensity_encoder ():
            profile_encoder ():
            loss_fn ():
            weight_decay (float):
            mc_samples (int):
            lr (float):
            max_iterations (int):
            renyi_scale (float):
            d (int):
            h (int):
            w (int):
        """
        """
        Args:
            intensity_encoder ():
            profile_encoder ():
            loss ():
            qbg ():
            qp ():
            qI ():
            weight_decay ():
            mc_samples:
            lr:
            max_iterations:
            renyi_scale:
            d:
            h:
            w:
        """
        super().__init__(
            qbg=qbg,
            qp=qp,
            qI=qI,
            loss_fn=loss_fn,
            d=d,
            h=h,
            w=w,
            lr=lr,
            weight_decay=weight_decay,
            mc_samples=mc_samples,
            max_iterations=max_iterations,
            renyi_scale=renyi_scale,
        )
        # Save hyperparameters
        self.save_hyperparameters(
            ignore=[
                "image_encoder",
                "profile_model",
                "unet",
                "signal_preprocessor",
            ]
        )

        # Model components
        self.intensity_encoder = intensity_encoder
        self.profile_encoder = profile_encoder
        self.encoder3 = nn.Linear(11, 64)

    # def forward(self, counts, shoebox, metadata, masks, reference):
    def forward(self, counts, shoebox, masks, reference):
        """
        Args:
            counts (): The raw shoebox data
            shoebox (): The standardized shoebox data
            masks (): Dead pixel mask
            reference ():

        Returns:

        """
        # Unpack batch
        counts = torch.clamp(counts, min=0) * masks
        device = counts.device

        profile_rep = self.profile_encoder(
            shoebox.reshape(shoebox.shape[0], 1, self.d, self.h, self.w), masks
        )
        intensity_rep = self.intensity_encoder(
            shoebox.reshape(shoebox.shape[0], 1, self.d, self.h, self.w), masks
        )

        # qbg = self.qbg(rep2,metarep=rep3)
        qbg = self.qbg(intensity_rep)
        qp = self.qp(profile_rep)
        # qI = self.qI(rep2, metarep=rep3)
        qI = self.qI(intensity_rep)
        # qI = self.qI(rep2)

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)
        zI = qI.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        intensity_mean = qI.mean  # [batch_size]
        intensity_var = qI.variance  # [batch_size]

        rate = zI * zp + zbg

        return {
            "rates": rate,
            "counts": counts,
            "masks": masks,
            "qbg": qbg,
            "qp": qp,
            "qp_mean": qp.mean,
            "qI": qI,
            "intensity_mean": intensity_mean,
            "intensity_mean": intensity_mean,
            "intensity_var": intensity_var,
            "dials_I_sum_value": reference[:, 6],
            "dials_I_sum_var": reference[:, 7],
            "dials_I_prf_value": reference[:, 8],
            "dials_I_prf_var": reference[:, 9],
            "refl_ids": reference[:, -1],
            "profile": qp.mean,
            "zp": zp,
            "x_c": reference[:, 0],
            "y_c": reference[:, 1],
            "z_c": reference[:, 2],
            "x_c_mm": reference[:, 3],
            "y_c_mm": reference[:, 4],
            "z_c_mm": reference[:, 5],
            "dials_bg_mean": reference[:, 10],
            "dials_bg_sum_value": reference[:, 11],
            "dials_bg_sum_var": reference[:, 12],
            "d": reference[:, 13],
        }


if __name__ == "__main__":
    from integrator.utils import load_config

    config = load_config(
        "/Users/luis/integratorv3/integrator/src/integrator/config/config.yaml"
    )
