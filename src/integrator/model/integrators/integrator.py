import torch

from integrator.model.integrators import BaseIntegrator


class Integrator(BaseIntegrator):
    def __init__(
        self,
        qbg,
        qp,
        qi,
        encoder1,
        encoder2,
        loss,
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
            qi ():
            encoder1 ():
            encoder2 ():
            loss ():
            weight_decay (float):
            mc_samples (int):
            lr (float):
            max_iterations (int):
            renyi_scale (float):
            d (int):
            h (int):
            w (int):
        """
        super().__init__(
            qbg=qbg,
            qp=qp,
            qi=qi,
            loss=loss,
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
        self.encoder1 = encoder1
        self.encoder2 = encoder2

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

        profile_rep = self.encoder1(
            shoebox.reshape(shoebox.shape[0], 1, self.d, self.h, self.w), masks
        )
        intensity_rep = self.encoder2(
            shoebox.reshape(shoebox.shape[0], 1, self.d, self.h, self.w), masks
        )

        qbg = self.qbg(intensity_rep)
        qp = self.qp(profile_rep)
        qi = self.qi(intensity_rep)

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)
        zI = qi.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        rate = zI * zp + zbg

        # calculate profile renyi entropy
        avg_reynyi_entropy = (-(zp.pow(2).sum(-1).log())).mean(-1)

        return {
            "rates": rate,
            "counts": counts,
            "masks": masks,
            "qbg": qbg,
            "qp": qp,
            "qp_mean": qp.mean,
            "qi": qi,
            "intensity_mean": qi.mean,
            "intensity_var": qi.variance,
            "dials_I_sum_value": reference[:, 6],
            "dials_I_sum_var": reference[:, 7],
            "dials_I_prf_value": reference[:, 8],
            "dials_I_prf_var": reference[:, 9],
            "refl_ids": reference[:, -1].int().tolist(),
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
            "avg_reynyi_entropy": avg_reynyi_entropy,
        }


class Model2(BaseIntegrator):
    def __init__(
        self,
        qbg,
        qp,
        qi,
        encoder,
        intensity_encoder,
        loss,
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
            qi ():
            intensity_encoder ():
            encoder2 ():
            loss ():
            weight_decay (float):
            mc_samples (int):
            lr (float):
            max_iterations (int):
            renyi_scale (float):
            d (int):
            h (int):
            w (int):
        """
        super().__init__(
            qbg=qbg,
            qp=qp,
            qi=qi,
            loss=loss,
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
        self.encoder1 = intensity_encoder
        self.encoder2 = encoder

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

        # encode input data
        profile_rep = self.encoder(
            shoebox.reshape(shoebox.shape[0], 1, self.d, self.h, self.w), masks
        )
        intensity_rep = self.encoder(
            shoebox.reshape(shoebox.shape[0], 1, self.d, self.h, self.w), masks
        )

        # build distributions
        qbg = self.qbg(intensity_rep)
        qp = self.qp(profile_rep)
        qi = self.qi(intensity_rep)

        # estimate rate
        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)
        zI = qi.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        return {
            "rates": zI * zp + zbg,
            "counts": counts,
            "masks": masks,
            "qbg": qbg,
            "qp": qp,
            "qp_mean": qp.mean,
            "qi": qi,
            "intensity_mean": qi.mean,
            "intensity_var": qi.variance,
            "dials_I_sum_value": reference[:, 6],
            "dials_I_sum_var": reference[:, 7],
            "dials_I_prf_value": reference[:, 8],
            "dials_I_prf_var": reference[:, 9],
            "refl_ids": reference[:, -1].int().tolist(),
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


class Integrator2D(BaseIntegrator):
    def __init__(
        self,
        qbg,
        qp,
        qi,
        encoder1,
        encoder2,
        loss,
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
            qi ():
            encoder1 ():
            encoder2 ():
            loss ():
            weight_decay (float):
            mc_samples (int):
            lr (float):
            max_iterations (int):
            renyi_scale (float):
            d (int):
            h (int):
            w (int):
        """
        super().__init__(
            qbg=qbg,
            qp=qp,
            qi=qi,
            loss=loss,
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
        self.encoder1 = encoder1
        self.encoder2 = encoder2

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

        profile_rep = self.encoder1(
            shoebox.reshape(shoebox.shape[0], 1, self.h, self.w), masks
        )

        intensity_rep = self.encoder2(
            shoebox.reshape(shoebox.shape[0], 1, self.h, self.w), masks
        )
        qbg = self.qbg(intensity_rep)
        qp = self.qp(profile_rep)
        qi = self.qi(intensity_rep)

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)  # [B,S,1]
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)  # [B,S,Pix]
        zI = qi.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)  # [B,S,1]
        rate = zI * zp + zbg

        return {
            "rates": rate,
            "counts": counts,
            "masks": masks,
            "qbg": qbg,
            "qp": qp,
            "qp_mean": qp.mean,
            "qi": qi,
            "intensity_mean": qi.mean,
            "intensity_var": qi.variance,
            "dials_I_sum_value": reference[:, 3],
            "dials_I_sum_var": reference[:, 4],
            "dials_I_prf_value": reference[:, 3],
            "dials_I_prf_var": reference[:, 4],
            "refl_ids": reference[:, -1].tolist(),
            "profile": qp.mean,
            "zp": zp,
            "x_c": reference[:, 9],
            "y_c": reference[:, 10],
            "z_c": reference[:, 11],
            # "x_c_mm": reference[:, 3],
            # "y_c_mm": reference[:, 4],
            # "z_c_mm": reference[:, 5],
            "dials_bg_mean": reference[:, 0],
            "dials_bg_sum_value": reference[:, 0],
            "dials_bg_sum_var": reference[:, 1],
            # "d": reference[:, 13]
        }


# Including metadata
class Model3(BaseIntegrator):
    def __init__(
        self,
        qbg,
        qp,
        qi,
        encoder1,
        encoder2,
        encoder3,
        loss,
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
            qi ():
            encoder1 ():
            encoder2 ():
            loss ():
            weight_decay (float):
            mc_samples (int):
            lr (float):
            max_iterations (int):
            renyi_scale (float):
            d (int):
            h (int):
            w (int):
        """
        super().__init__(
            qbg=qbg,
            qp=qp,
            qi=qi,
            loss=loss,
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
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.encoder3 = encoder3

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
        metadata = reference[:, [0, 1, 2, 13]]
        device = counts.device

        profile_rep = self.encoder1(
            shoebox.reshape(shoebox.shape[0], 1, self.d, self.h, self.w), masks
        )
        intensity_rep = self.encoder2(
            shoebox.reshape(shoebox.shape[0], 1, self.d, self.h, self.w), masks
        )
        metadata_rep = self.encoder3(metadata)
        met_inten_rep = intensity_rep + metadata_rep

        qbg = self.qbg(met_inten_rep)
        qp = self.qp(profile_rep)
        qi = self.qi(met_inten_rep)

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)
        zI = qi.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        rate = zI * zp + zbg

        # calculate profile renyi entropy
        avg_reynyi_entropy = (-(zp.pow(2).sum(-1).log())).mean(-1)

        return {
            "rates": rate,
            "counts": counts,
            "masks": masks,
            "qbg": qbg,
            "qp": qp,
            "qp_mean": qp.mean,
            "qi": qi,
            "intensity_mean": qi.mean,
            "intensity_var": qi.variance,
            "dials_I_sum_value": reference[:, 6],
            "dials_I_sum_var": reference[:, 7],
            "dials_I_prf_value": reference[:, 8],
            "dials_I_prf_var": reference[:, 9],
            "refl_ids": reference[:, -1].int().tolist(),
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
            "avg_reynyi_entropy": avg_reynyi_entropy,
        }


if __name__ == "__main__":
    import torch

    from integrator.utils import create_integrator, load_config

    config = load_config(
        "/Users/luis/integratorv3/integrator/src/integrator/config/model3.yaml"
    )

    integrator = create_integrator(config)

    counts = torch.exp(torch.randn(10, 1323))
    shoebox = torch.exp(torch.randn(10, 1323))
    masks = torch.randint(2, (10, 1323))
    reference = torch.randn(10, 14)

    integrator(counts, shoebox, masks, reference)
