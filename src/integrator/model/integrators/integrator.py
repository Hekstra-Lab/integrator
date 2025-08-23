import torch
from torch import Tensor, nn

from integrator.model.distributions import BaseDistribution
from integrator.model.integrators import BaseIntegrator
from integrator.model.loss import BaseLoss


def get_outputs(vars: dict, data_dim: str) -> dict:
    # default network outputs
    out = {
        "rates": vars["rate"],
        "counts": vars["counts"],
        "masks": vars["masks"],
        "qbg": vars["qbg"],
        "qp": vars["qp"],
        "qp_mean": vars["qp"].mean,
        "qi": vars["qi"],
        "intensity_mean": vars["qi"].mean,
        "intensity_var": vars["qi"].variance,
        "profile": vars["qp"].mean,
        "zp": vars["zp"],
    }

    if vars["reference"] is not None:
        reference = vars["reference"]

        if data_dim == "3d":
            ref_3d = {
                "dials_I_sum_value": reference[:, 6],
                "dials_I_sum_var": reference[:, 7],
                "dials_I_prf_value": reference[:, 8],
                "dials_I_prf_var": reference[:, 9],
                "refl_ids": reference[:, -1].int().tolist(),
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
            for k, v in ref_3d.items():
                out[k] = v

        elif data_dim == "2d":
            ref_2d = {
                "dials_I_sum_value": reference[:, 3],
                "dials_I_sum_var": reference[:, 4],
                "dials_I_prf_value": reference[:, 3],
                "dials_I_prf_var": reference[:, 4],
                "refl_ids": reference[:, -1].tolist(),
                "x_c": reference[:, 9],
                "y_c": reference[:, 10],
                "z_c": reference[:, 11],
                "dials_bg_mean": reference[:, 0],
                "dials_bg_sum_value": reference[:, 0],
                "dials_bg_sum_var": reference[:, 1],
            }

            for k, v in ref_2d.items():
                out[k] = v

    elif vars["reference"] is None:
        return out

    else:
        print("Invalid output data")

    return out


class Integrator(BaseIntegrator):
    def __init__(
        self,
        qbg: BaseDistribution,
        qp: BaseDistribution,
        qi: BaseDistribution,
        encoder1: nn.Module,
        encoder2: nn.Module,
        loss: BaseLoss,
        mc_samples: int = 100,
        lr: float = 1e-3,
        max_iterations: int = 4,
        renyi_scale: float = 0.00,
        d: int = 3,
        h: int = 21,
        w: int = 21,
        weight_decay=1e-8,
    ):
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

        self.data_dim: str = "3d"
        # Save hyperparameters
        self.save_hyperparameters(
            ignore=[
                "encoder1",
                "encoder2",
                "qp",
                "qi",
                "qbg",
                "loss",
            ]
        )

        # Model components
        self.encoder1 = encoder1
        self.encoder2 = encoder2

    # def forward(self, counts, shoebox, metadata, masks, reference):
    def forward(
        self,
        counts: Tensor,
        shoebox: Tensor,
        masks: Tensor,
        reference: Tensor | None = None,
    ) -> dict:
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
        out = get_outputs(locals(), self.data_dim)
        return out


class Model2(BaseIntegrator):
    def __init__(
        self,
        qbg: BaseDistribution,
        qp: BaseDistribution,
        qi: BaseDistribution,
        encoder1: nn.Module,
        encoder2: nn.Module,
        loss: BaseLoss,
        mc_samples: int = 100,
        lr: float = 1e-3,
        max_iterations: int = 4,
        renyi_scale: float = 0.00,
        d: int = 3,
        h: int = 21,
        w: int = 21,
        weight_decay=1e-8,
    ):
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
                "qbg",
                "qp",
                "qi",
                "loss",
                "encoder1",
                "encoder2",
            ]
        )

        self.data_dim: str = "3d"

        # Model components
        self.encoder1 = encoder1
        self.encoder2 = encoder2

    def forward(
        self,
        counts: Tensor,
        shoebox: Tensor,
        masks: Tensor,
        reference: Tensor | None = None,
    ) -> dict:
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

        # encode input data
        profile_rep = self.encoder1(
            shoebox.reshape(shoebox.shape[0], 1, self.d, self.h, self.w), masks
        )
        intensity_rep = self.encoder2(
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

        out = get_outputs(vars=locals(), data_dim=self.data_dim)

        return out


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
        self.data_dim: str = "2d"
        self.save_hyperparameters(
            ignore=[
                "encoder1",
                "encoder2",
                "loss",
                "qbg",
                "qp",
                "qi",
            ]
        )

        # Model components
        self.encoder1 = encoder1
        self.encoder2 = encoder2

    # def forward(self, counts, shoebox, metadata, masks, reference):
    def forward(
        self,
        counts: Tensor,
        shoebox: Tensor,
        masks: Tensor,
        reference: Tensor | None = None,
    ) -> dict:
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

        profile_rep = self.encoder1(shoebox.reshape(shoebox.shape[0], 1, self.h, self.w))

        intensity_rep = self.encoder2(shoebox.reshape(shoebox.shape[0], 1, self.h, self.w))
        qbg = self.qbg(intensity_rep)
        qp = self.qp(profile_rep)
        qi = self.qi(intensity_rep)

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)  # [B,S,1]
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)  # [B,S,Pix]
        zI = qi.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)  # [B,S,1]
        rate = zI * zp + zbg
        out = get_outputs(
            vars=locals(),
            data_dim=self.data_dim,
        )
        return out


# Including metadata
class Model3(BaseIntegrator):
    """
    Integrator for 3D shoeboxes
    Attributes:
        encoder1:
        encoder2:
        encoder3:
    """

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
        self.data_dim: str = "3d"
        self.save_hyperparameters(
            ignore=[
                "encoder1",
                "encoder2",
                "encoder3",
                "qbg",
                "qp",
                "qi",
                "loss",
            ]
        )

        # Model components
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.encoder3 = encoder3

    def forward(
        self,
        counts: Tensor,
        shoebox: Tensor,
        masks: Tensor,
        reference: Tensor | None = None,
    ) -> dict:
        if masks is not None:
            masks = masks

        counts = torch.clamp(counts, min=0) * masks

        metadata = torch.tensor([0.0])
        if reference is not None:
            metadata = reference[:, [0, 1, 2, 13]]
        else:
            print("You must use a reference.pt with Model3!")

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

        out = get_outputs(vars=locals(), data_dim=self.data_dim)

        return out


if __name__ == "__main__":
    import torch

    from integrator.utils import create_data_loader, create_integrator, load_config
    from utils import CONFIGS, ROOT_DIR

    data_path = ROOT_DIR / "tests/data/3d/hewl_9b7c"

    # load 3d model
    config = load_config(CONFIGS["config3d"])

    data_loader = create_data_loader(config.dict())

    integrator = create_integrator(config.dict())

    # ld data
    config = load_config(CONFIGS["config2d"])
    integrator = create_integrator(config.dict())

    data_loader = create_data_loader(config.dict())

    counts, shoebox, masks, reference = next(iter(data_loader.train_dataloader()))

    out = integrator(counts, shoebox, masks, reference)
