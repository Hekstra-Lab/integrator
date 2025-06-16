import math

import torch
import torch.nn as nn
from lightning.pytorch.utilities import grad_norm

from integrator.model.integrators import BaseIntegrator


def int_to_20bit_binary(x: torch.Tensor) -> torch.Tensor:
    """
    Converts an integer tensor to its 20-bit binary representation.

    Args:
        x: A tensor of non-negative integers with values in [0, 2^20)

    Returns:
        A tensor of shape (*x.shape, 20) with 0/1 binary values
    """
    assert torch.all((x >= 0) & (x < 2**20)), (
        "All values must be in the range [0, 2^20)"
    )

    # Create a tensor with the bit masks [2^19, ..., 2^0]
    device = x.device
    bit_positions = torch.arange(19, -1, -1, device=device)
    bit_masks = 2**bit_positions

    # Unsigned right shift and mask
    return ((x.unsqueeze(-1) & bit_masks) > 0).to(torch.float32)


def binary_to_int(bits: torch.Tensor) -> torch.Tensor:
    """
    Converts 20-bit binary representation back to integer.

    Args:
        bits: Tensor of shape (..., 20) with 0/1 float or bool

    Returns:
        Tensor of integers
    """
    bit_positions = torch.arange(19, -1, -1, device=bits.device)
    weights = 2**bit_positions
    return torch.sum(bits * weights, dim=-1).long()


# dirichlet version
class Integrator(BaseIntegrator):
    """
    Attributes:
        d: An integer indicating the depth of the input shoebox
        h: An integer indicating the height of the input shoebox
        w: An integer indicating the width of the input shoebox
        learning_rate: A float indicating the learning rate of the optimizer
        mc_samples: An integer indicating the number of Monte Carlo samples to use
        intensity_encoder: Encoder object
        profile_encoder: Encoder object
        qp: Profile distribution object
        qI: Intensity distribution object
        qbg: Background distribution object
        automatic_optimization:
        loss_fn: Loss function object
        max_iterations: Max number of iterations for Kabsch summation
        encoder3:
        renyi_scale: Scaling for Renyi entropy regularization
    """

    def __init__(
        self,
        intensity_encoder,
        profile_encoder,
        loss,
        qbg,
        qp,
        qI,
        mc_samples: int = 100,
        learning_rate: float = 1e-3,
        max_iterations: int = 4,
        renyi_scale: float = 0.00,
        d: int = 3,
        h: int = 21,
        w: int = 21,
    ):
        super().__init__()
        # Save hyperparameters
        self.d = d
        self.h = h
        self.w = w
        self.save_hyperparameters(
            ignore=[
                "image_encoder",
                "mlp_encoder",
                "profile_model",
                "unet",
                "signal_preprocessor",
            ]
        )
        self.learning_rate = learning_rate
        self.mc_samples = mc_samples

        # Model components
        self.intensity_encoder = intensity_encoder
        self.profile_encoder = profile_encoder
        self.qp = qp
        self.qI = qI
        self.qbg = qbg
        self.automatic_optimization = True
        self.loss_fn = loss
        self.max_iterations = max_iterations
        self.encoder3 = nn.Linear(11, 64)
        self.renyi_scale = renyi_scale

    def calculate_intensities(self, counts, qbg, qp, masks):
        with torch.no_grad():
            counts = counts * masks  # [B,P]
            zbg = (
                qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
            )  # [B,S,1]
            zp = qp.mean.unsqueeze(1)  # [B,1,P]

            vi = zbg + 1e-6

            # kabsch sum
            for i in range(self.max_iterations):
                num = (counts.unsqueeze(1) - zbg) * zp * masks.unsqueeze(1) / vi
                denom = zp.pow(2) / vi
                I = num.sum(-1) / denom.sum(-1)  # [batch_size, mc_samples]
                vi = (I.unsqueeze(-1) * zp) + zbg
                vi = vi.mean(-1, keepdim=True)
            kabsch_sum_mean = I.mean(-1)
            kabsch_sum_var = I.var(-1)

            # profile masking
            zp = zp * masks.unsqueeze(1)  # profiles
            thresholds = torch.quantile(
                zp, 0.99, dim=-1, keepdim=True
            )  # threshold values
            profile_mask = zp > thresholds

            masked_counts = counts.unsqueeze(1) * profile_mask

            profile_masking_I = (masked_counts - zbg * profile_mask).sum(-1)

            profile_masking_mean = profile_masking_I.mean(-1)

            profile_masking_var = profile_masking_I.var(-1)

            intensities = {
                "profile_masking_mean": profile_masking_mean,
                "profile_masking_var": profile_masking_var,
                "kabsch_sum_mean": kabsch_sum_mean,
                "kabsch_sum_var": kabsch_sum_var,
            }

            return intensities

    # def forward(self, counts, shoebox, metadata, masks, reference):
    def forward(self, counts, shoebox, masks, reference):
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

    def training_step(self, batch, batch_idx):
        counts, shoebox, masks, reference = batch
        outputs = self(counts, shoebox, masks, reference)

        # Calculate loss
        (loss, neg_ll, kl, kl_bg, kl_I, kl_p) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_I=outputs["qI"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Track gradient norms here
        norms = grad_norm(self, norm_type=2)
        for name, norm in norms.items():
            self.log(f"grad_norm/{name}", norm)

        renyi_loss = (
            (-torch.log(outputs["qp"].rsample([100]).permute(1, 0, 2).pow(2).sum(-1)))
            .mean(1)
            .sum()
        ) * self.renyi_scale
        self.log("renyi_loss", renyi_loss)

        # Log metrics
        self.log("Train: -ELBO", loss.mean())
        self.log("Train: NLL", neg_ll.mean())
        self.log("Train: KL", kl.mean())
        self.log("Train: KL Bg", kl_bg.mean())
        self.log("Train: KL I", kl_I.mean())
        self.log("Train: KL Prf", kl_p.mean())
        self.log("Mean(qI.mean)", outputs["qI"].mean.mean())
        self.log("Min(qI.mean)", outputs["qI"].mean.min())
        self.log("Max(qI.mean)", outputs["qI"].mean.max())
        self.log("Mean(qbg.mean)", outputs["qbg"].mean.mean())
        self.log("Min(qbg.mean)", outputs["qbg"].mean.min())
        self.log("Max(qbg.mean)", outputs["qbg"].mean.max())
        self.log("Mean(qbg.variance)", outputs["qbg"].variance.mean())
        self.train_loss.append(loss.mean())
        self.train_kl.append(kl.mean())
        self.train_nll.append(neg_ll.mean())
        return loss.mean() + renyi_loss.sum()

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        counts, shoebox, masks, reference = batch
        outputs = self(counts, shoebox, masks, reference)

        (
            loss,
            neg_ll,
            kl,
            kl_bg,
            kl_I,
            kl_p,
        ) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_I=outputs["qI"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Log metrics
        self.log("Val: -ELBO", loss.mean())
        self.log("Val: NLL", neg_ll.mean())
        self.log("Val: KL", kl.mean())
        self.log("Val: KL bg", kl_bg.mean())
        self.log("Val: KL I", kl_I.mean())
        self.log("Val: KL prf", kl_p.mean())
        self.log("val_loss", neg_ll.mean())

        self.val_loss.append(loss.mean())
        self.val_kl.append(kl.mean())
        self.val_nll.append(neg_ll.mean())

        return outputs

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-8
        )

    def predict_step(self, batch, batch_idx):
        counts, shoebox, masks, reference = batch
        outputs = self(counts, shoebox, masks, reference)

        intensities = self.calculate_intensities(
            counts=outputs["counts"],
            qbg=outputs["qbg"],
            qp=outputs["qp"],
            masks=outputs["masks"],
        )

        return {
            "intensity_mean": outputs["intensity_mean"],  # qI.mean
            "intensity_var": outputs["intensity_var"],  # qI.variance
            "refl_ids": outputs["refl_ids"],
            "dials_I_sum_var": outputs["dials_I_sum_var"],
            "dials_I_prf_value": outputs["dials_I_prf_value"],
            "dials_I_prf_var": outputs["dials_I_prf_var"],
            "qbg": outputs["qbg"].mean,
            "qbg_scale": outputs["qbg"].scale,  # halfnormal param
            "profile_masking_mean": intensities["profile_masking_mean"],
            "profile_masking_var": intensities["profile_masking_var"],
            "kabsch_sum_mean": intensities["kabsch_sum_mean"],
            "kabsch_sum_var": intensities["kabsch_sum_var"],
            "x_c": outputs["x_c"],
            "y_c": outputs["y_c"],
            "z_c": outputs["z_c"],
        }


# %%
class IntegratorFourierFeatures(BaseIntegrator):
    def __init__(
        self,
        encoder,
        encoder2,
        loss,
        qbg,
        qp,
        qI,
        mc_samples=100,
        learning_rate=1e-3,
        max_iterations=4,
        profile_threshold=0.001,
        renyi_scale=0.00,
        ff_scale=1.0,
        num_fourier_features=10,
    ):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters(
            ignore=[
                "image_encoder",
                "mlp_encoder",
                "profile_model",
                "unet",
                "signal_preprocessor",
            ]
        )
        self.learning_rate = learning_rate
        self.mc_samples = mc_samples

        # Model components
        self.encoder = encoder
        self.qp = qp
        self.qI = qI
        self.qbg = qbg
        self.automatic_optimization = True
        self.loss_fn = loss
        self.max_iterations = max_iterations
        self.intensity_encoder = encoder2
        # self.linear = Linear(64 * 2, 64)
        self.renyi_scale = renyi_scale
        B = torch.distributions.Normal(0, ff_scale).sample((num_fourier_features, 3))
        self.register_buffer("B", B, persistent=True)

    def calculate_intensities(self, counts, qbg, qp, masks):
        with torch.no_grad():
            counts = counts * masks  # [B,P]
            zbg = (
                qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
            )  # [B,S,1]
            # zp = qp.rsample([self.mc_samples]).permute(1, 0, 2) #
            zp = qp.mean.unsqueeze(1)  # [B,1,P]

            vi = zbg + 1e-6

            # kabsch sum
            for i in range(4):
                num = (counts.unsqueeze(1) - zbg) * zp * masks.unsqueeze(1) / vi
                denom = zp.pow(2) / vi
                I = num.sum(-1) / denom.sum(-1)  # [batch_size, mc_samples]
                vi = (I.unsqueeze(-1) * zp) + zbg
                vi = vi.mean(-1, keepdim=True)
            kabsch_sum_mean = I.mean(-1)
            kabsch_sum_var = I.var(-1)

            # profile masking
            zp = zp * masks.unsqueeze(1)  # profiles
            thresholds = torch.quantile(
                zp, 0.99, dim=-1, keepdim=True
            )  # threshold values
            profile_mask = zp > thresholds

            masked_counts = counts.unsqueeze(1) * profile_mask

            profile_masking_I = (masked_counts - zbg * profile_mask).sum(-1)

            profile_masking_mean = profile_masking_I.mean(-1)

            profile_masking_var = profile_masking_I.var(-1)

            intensities = {
                "profile_masking_mean": profile_masking_mean,
                "profile_masking_var": profile_masking_var,
                "kabsch_sum_mean": kabsch_sum_mean,
                "kabsch_sum_var": kabsch_sum_var,
            }

            return intensities

    def forward(self, counts, shoebox, metadata, masks, reference):
        # Unpack batch
        counts_ = torch.clamp(counts[..., -1].clone(), min=0) * masks
        device = counts_.device

        proj = 2 * torch.pi * counts[..., :3] @ self.B.to(device).T
        features = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        samples_ = torch.concat([features, counts[..., -1].unsqueeze(-1)], dim=-1)
        samples_ = samples_.view(samples_.shape[0], 3, 21, 21, 21)
        samples_ = samples_.permute(0, 4, 1, 2, 3)

        rep = self.encoder(samples_, masks)

        intensity_rep = self.intensity_encoder(samples_)

        qbg = self.qbg(intensity_rep)
        qp = self.qp(rep)
        qI = self.qI(intensity_rep)

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)
        zI = qI.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        intensity_mean = qI.mean  # [batch_size]
        intensity_var = qI.variance  # [batch_size]

        rate = zI * zp + zbg

        return {
            "rates": rate,
            "counts": counts_,
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

    def training_step(self, batch, batch_idx):
        counts, shoebox, metadata, masks, reference = batch
        outputs = self(counts, shoebox, metadata, masks, reference)

        # Calculate loss
        (loss, neg_ll, kl, kl_bg, kl_I, kl_p) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_I=outputs["qI"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Track gradient norms here
        norms = grad_norm(self, norm_type=2)
        for name, norm in norms.items():
            self.log(f"grad_norm/{name}", norm)

        renyi_loss = (
            (-torch.log(outputs["qp"].rsample([100]).permute(1, 0, 2).pow(2).sum(-1)))
            .mean(1)
            .sum()
        ) * self.renyi_scale
        self.log("renyi_loss", renyi_loss)

        # Log metrics
        self.log("Train: -ELBO", loss.mean())
        self.log("Train: NLL", neg_ll.mean())
        self.log("Train: KL", kl.mean())
        self.log("Train: KL Bg", kl_bg.mean())
        self.log("Train: KL I", kl_I.mean())
        self.log("Train: KL Prf", kl_p.mean())
        self.log("Mean(qI.mean)", outputs["qI"].mean.mean())
        self.log("Min(qI.mean)", outputs["qI"].mean.min())
        self.log("Max(qI.mean)", outputs["qI"].mean.max())
        self.log("Mean(qbg.mean)", outputs["qbg"].mean.mean())
        self.log("Min(qbg.mean)", outputs["qbg"].mean.min())
        self.log("Max(qbg.mean)", outputs["qbg"].mean.max())
        self.log("Mean(qbg.variance)", outputs["qbg"].variance.mean())

        return loss.mean() + renyi_loss.sum()

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        counts, shoebox, metadata, masks, reference = batch
        outputs = self(counts, shoebox, metadata, masks, reference)

        (
            loss,
            neg_ll,
            kl,
            kl_bg,
            kl_I,
            kl_p,
        ) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_I=outputs["qI"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Log metrics
        self.log("Val: -ELBO", loss.mean())
        self.log("Val: NLL", neg_ll.mean())
        self.log("Val: KL", kl.mean())
        self.log("Val: KL bg", kl_bg.mean())
        self.log("Val: KL I", kl_I.mean())
        self.log("Val: KL prf", kl_p.mean())
        self.log("val_loss", neg_ll.mean())

        return outputs

    def predict_step(self, batch, batch_idx):
        counts, shoebox, metadata, masks, reference = batch
        outputs = self(counts, shoebox, metadata, masks, reference)
        intensities = self.calculate_intensities(
            counts=outputs["counts"],
            qbg=outputs["qbg"],
            qp=outputs["qp"],
            masks=outputs["masks"],
        )

        return {
            "intensity_mean": outputs["intensity_mean"],  # qI.mean
            "intensity_var": outputs["intensity_var"],  # qI.variance
            "refl_ids": outputs["refl_ids"],
            "dials_I_sum_var": outputs["dials_I_sum_var"],
            "dials_I_prf_value": outputs["dials_I_prf_value"],
            "dials_I_prf_var": outputs["dials_I_prf_var"],
            "qbg": outputs["qbg"].mean,
            "qbg_scale": outputs["qbg"].scale,  # halfnormal param
            "profile_masking_mean": intensities["profile_masking_mean"],
            "profile_masking_var": intensities["profile_masking_var"],
            "kabsch_sum_mean": intensities["kabsch_sum_mean"],
            "kabsch_sum_var": intensities["kabsch_sum_var"],
            "x_c": outputs["x_c"],
            "y_c": outputs["y_c"],
            "z_c": outputs["z_c"],
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# %%


class IntegratorBinaryEncoding(BaseIntegrator):
    """
    Attributes:
        d: An integer indicating the depth of the input shoebox
        h: An integer indicating the height of the input shoebox
        w: An integer indicating the width of the input shoebox
        learning_rate: A float indicating the learning rate of the optimizer
        mc_samples: An integer indicating the number of Monte Carlo samples to use
        intensity_encoder: Encoder object
        profile_encoder: Encoder object
        qp: Profile distribution object
        qI: Intensity distribution object
        qbg: Background distribution object
        automatic_optimization:
        loss_fn: Loss function object
        max_iterations: Max number of iterations for Kabsch summation
        encoder3:
        renyi_scale: Scaling for Renyi entropy regularization
    """

    def __init__(
        self,
        intensity_encoder,
        profile_encoder,
        loss,
        qbg,
        qp,
        qI,
        mc_samples: int = 100,
        learning_rate: float = 1e-3,
        max_iterations: int = 4,
        renyi_scale: float = 0.00,
        d: int = 3,
        h: int = 21,
        w: int = 21,
    ):
        super().__init__()
        # Save hyperparameters
        self.d = d
        self.h = h
        self.w = w
        self.save_hyperparameters(
            ignore=[
                "image_encoder",
                "mlp_encoder",
                "profile_model",
                "unet",
                "signal_preprocessor",
            ]
        )
        self.learning_rate = learning_rate
        self.mc_samples = mc_samples

        # Model components
        self.intensity_encoder = intensity_encoder
        self.profile_encoder = profile_encoder
        self.qp = qp
        self.qI = qI
        self.qbg = qbg
        self.automatic_optimization = True
        self.loss_fn = loss
        self.max_iterations = max_iterations
        self.encoder3 = nn.Linear(11, 64)
        self.renyi_scale = renyi_scale

    def calculate_intensities(self, counts, qbg, qp, masks):
        with torch.no_grad():
            counts = counts * masks  # [B,P]
            zbg = (
                qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
            )  # [B,S,1]
            zp = qp.mean.unsqueeze(1)  # [B,1,P]

            vi = zbg + 1e-6

            # kabsch sum
            for i in range(self.max_iterations):
                num = (counts.unsqueeze(1) - zbg) * zp * masks.unsqueeze(1) / vi
                denom = zp.pow(2) / vi
                I = num.sum(-1) / denom.sum(-1)  # [batch_size, mc_samples]
                vi = (I.unsqueeze(-1) * zp) + zbg
                vi = vi.mean(-1, keepdim=True)
            kabsch_sum_mean = I.mean(-1)
            kabsch_sum_var = I.var(-1)

            # profile masking
            zp = zp * masks.unsqueeze(1)  # profiles
            thresholds = torch.quantile(
                zp, 0.99, dim=-1, keepdim=True
            )  # threshold values
            profile_mask = zp > thresholds

            masked_counts = counts.unsqueeze(1) * profile_mask

            profile_masking_I = (masked_counts - zbg * profile_mask).sum(-1)

            profile_masking_mean = profile_masking_I.mean(-1)

            profile_masking_var = profile_masking_I.var(-1)

            intensities = {
                "profile_masking_mean": profile_masking_mean,
                "profile_masking_var": profile_masking_var,
                "kabsch_sum_mean": kabsch_sum_mean,
                "kabsch_sum_var": kabsch_sum_var,
            }

            return intensities

    # def forward(self, counts, shoebox, metadata, masks, reference):
    def forward(self, counts, shoebox, masks, reference):
        # Unpack batch
        counts = torch.clamp(counts, min=0) * masks
        device = counts.device

        shoebox = int_to_20bit_binary(counts.type(torch.int32))

        if shoebox.dim() == 2:
            num_channels = 1
        elif shoebox.dim() == 3:
            num_channels = shoebox.shape[-1]

        profile_rep = self.profile_encoder(
            shoebox.reshape(shoebox.shape[0], num_channels, self.d, self.h, self.w),
            masks,
        )
        intensity_rep = self.intensity_encoder(
            shoebox.reshape(shoebox.shape[0], num_channels, self.d, self.h, self.w),
            masks,
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

    def training_step(self, batch, batch_idx):
        counts, shoebox, masks, reference = batch
        outputs = self(counts, shoebox, masks, reference)

        # Calculate loss
        (loss, neg_ll, kl, kl_bg, kl_I, kl_p) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_I=outputs["qI"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Track gradient norms here
        norms = grad_norm(self, norm_type=2)
        for name, norm in norms.items():
            self.log(f"grad_norm/{name}", norm)

        renyi_loss = (
            (-torch.log(outputs["qp"].rsample([100]).permute(1, 0, 2).pow(2).sum(-1)))
            .mean(1)
            .sum()
        ) * self.renyi_scale
        self.log("renyi_loss", renyi_loss)

        # Log metrics
        self.log("Train: -ELBO", loss.mean())
        self.log("Train: NLL", neg_ll.mean())
        self.log("Train: KL", kl.mean())
        self.log("Train: KL Bg", kl_bg.mean())
        self.log("Train: KL I", kl_I.mean())
        self.log("Train: KL Prf", kl_p.mean())
        self.log("Mean(qI.mean)", outputs["qI"].mean.mean())
        self.log("Min(qI.mean)", outputs["qI"].mean.min())
        self.log("Max(qI.mean)", outputs["qI"].mean.max())
        self.log("Mean(qbg.mean)", outputs["qbg"].mean.mean())
        self.log("Min(qbg.mean)", outputs["qbg"].mean.min())
        self.log("Max(qbg.mean)", outputs["qbg"].mean.max())
        self.log("Mean(qbg.variance)", outputs["qbg"].variance.mean())
        self.train_loss.append(loss.mean())
        return loss.mean() + renyi_loss.sum()

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        counts, shoebox, masks, reference = batch
        outputs = self(counts, shoebox, masks, reference)

        (
            loss,
            neg_ll,
            kl,
            kl_bg,
            kl_I,
            kl_p,
        ) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_I=outputs["qI"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Log metrics
        self.log("Val: -ELBO", loss.mean())
        self.log("Val: NLL", neg_ll.mean())
        self.log("Val: KL", kl.mean())
        self.log("Val: KL bg", kl_bg.mean())
        self.log("Val: KL I", kl_I.mean())
        self.log("Val: KL prf", kl_p.mean())
        self.log("val_loss", neg_ll.mean())

        return outputs

    def predict_step(self, batch, batch_idx):
        counts, shoebox, masks, reference = batch
        outputs = self(counts, shoebox, masks, reference)

        intensities = self.calculate_intensities(
            counts=outputs["counts"],
            qbg=outputs["qbg"],
            qp=outputs["qp"],
            masks=outputs["masks"],
        )

        return {
            "intensity_mean": outputs["intensity_mean"],  # qI.mean
            "intensity_var": outputs["intensity_var"],  # qI.variance
            "refl_ids": outputs["refl_ids"],
            "dials_I_sum_var": outputs["dials_I_sum_var"],
            "dials_I_prf_value": outputs["dials_I_prf_value"],
            "dials_I_prf_var": outputs["dials_I_prf_var"],
            "qbg": outputs["qbg"].mean,
            "qbg_scale": outputs["qbg"].scale,  # halfnormal param
            "profile_masking_mean": intensities["profile_masking_mean"],
            "profile_masking_var": intensities["profile_masking_var"],
            "kabsch_sum_mean": intensities["kabsch_sum_mean"],
            "kabsch_sum_var": intensities["kabsch_sum_var"],
            "x_c": outputs["x_c"],
            "y_c": outputs["y_c"],
            "z_c": outputs["z_c"],
        }


# -


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)  # even dims
        pe[:, 1::2] = torch.cos(pos * div_term)  # odd dims
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer("pe", pe)  # not a Parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class BinaryPositionalEncoding(nn.Module):
    def __init__(self, bit_depth: int = 20):
        super().__init__()
        self.bit_depth = bit_depth
        # Create k = 2^(-0), 2^(-1), 2^(-2), ... = [1, 0.5, 0.25, 0.125, ...]
        k = 2.0 ** (-torch.arange(bit_depth, dtype=torch.float32))
        self.register_buffer("k", k)

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        """
        counts: tensor of photon counts of shape (..., )
        Returns: (..., 2*bit_depth) with binary-style positional encoding
        """
        # Add dimension for broadcasting: (..., 1) * (bit_depth,) -> (..., bit_depth)
        scaled_counts = counts.unsqueeze(-1) * self.k  # (..., bit_depth)

        # Apply sin and cos
        sin_part = torch.sin(math.pi * scaled_counts)  # (..., bit_depth)
        cos_part = torch.cos(math.pi * scaled_counts)  # (..., bit_depth)

        # Concatenate along last dimension
        out = torch.cat([sin_part, cos_part], dim=-1)  # (..., 2*bit_depth)

        return out


# -
class IntegratorPositionalEncoding(BaseIntegrator):
    """
    Attributes:
        d: An integer indicating the depth of the input shoebox
        h: An integer indicating the height of the input shoebox
        w: An integer indicating the width of the input shoebox
        learning_rate: A float indicating the learning rate of the optimizer
        mc_samples: An integer indicating the number of Monte Carlo samples to use
        intensity_encoder: Encoder object
        profile_encoder: Encoder object
        qp: Profile distribution object
        qI: Intensity distribution object
        qbg: Background distribution object
        automatic_optimization:
        loss_fn: Loss function object
        max_iterations: Max number of iterations for Kabsch summation
        encoder3:
        renyi_scale: Scaling for Renyi entropy regularization
    """

    def __init__(
        self,
        intensity_encoder,
        profile_encoder,
        loss,
        qbg,
        qp,
        qI,
        mc_samples: int = 100,
        learning_rate: float = 1e-3,
        max_iterations: int = 4,
        renyi_scale: float = 0.00,
        d: int = 3,
        h: int = 21,
        w: int = 21,
    ):
        super().__init__()
        # Save hyperparameters
        self.d = d
        self.h = h
        self.w = w
        self.save_hyperparameters(
            ignore=[
                "image_encoder",
                "mlp_encoder",
                "profile_model",
                "unet",
                "signal_preprocessor",
            ]
        )
        self.learning_rate = learning_rate
        self.mc_samples = mc_samples

        # Model components
        self.intensity_encoder = intensity_encoder
        self.profile_encoder = profile_encoder
        self.qp = qp
        self.qI = qI
        self.qbg = qbg
        self.automatic_optimization = True
        self.loss_fn = loss
        self.max_iterations = max_iterations
        self.renyi_scale = renyi_scale
        self.pos_encoder = BinaryPositionalEncoding()

    def calculate_intensities(self, counts, qbg, qp, masks):
        with torch.no_grad():
            counts = counts * masks  # [B,P]
            zbg = (
                qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
            )  # [B,S,1]
            zp = qp.mean.unsqueeze(1)  # [B,1,P]

            vi = zbg + 1e-6

            # kabsch sum
            for i in range(self.max_iterations):
                num = (counts.unsqueeze(1) - zbg) * zp * masks.unsqueeze(1) / vi
                denom = zp.pow(2) / vi
                I = num.sum(-1) / denom.sum(-1)  # [batch_size, mc_samples]
                vi = (I.unsqueeze(-1) * zp) + zbg
                vi = vi.mean(-1, keepdim=True)
            kabsch_sum_mean = I.mean(-1)
            kabsch_sum_var = I.var(-1)

            # profile masking
            zp = zp * masks.unsqueeze(1)  # profiles
            thresholds = torch.quantile(
                zp, 0.99, dim=-1, keepdim=True
            )  # threshold values
            profile_mask = zp > thresholds

            masked_counts = counts.unsqueeze(1) * profile_mask

            profile_masking_I = (masked_counts - zbg * profile_mask).sum(-1)

            profile_masking_mean = profile_masking_I.mean(-1)

            profile_masking_var = profile_masking_I.var(-1)

            intensities = {
                "profile_masking_mean": profile_masking_mean,
                "profile_masking_var": profile_masking_var,
                "kabsch_sum_mean": kabsch_sum_mean,
                "kabsch_sum_var": kabsch_sum_var,
            }

            return intensities

    # def forward(self, counts, shoebox, metadata, masks, reference):
    def forward(self, counts, shoebox, masks, reference):
        # Unpack batch
        counts = torch.clamp(counts, min=0) * masks
        device = counts.device

        # shoebox = int_to_20bit_binary(counts.type(torch.int32))
        pos_encoding = self.pos_encoder(shoebox)

        if shoebox.dim() == 2:
            num_channels = 1
        elif shoebox.dim() == 3:
            num_channels = shoebox.shape[-1]

        if pos_encoding.dim() == 2:
            num_channels_2 = 1
        elif pos_encoding.dim() == 3:
            num_channels_2 = pos_encoding.shape[-1]

        profile_rep = self.profile_encoder(
            shoebox.reshape(shoebox.shape[0], num_channels, self.d, self.h, self.w),
            masks,
        )
        intensity_rep = self.intensity_encoder(
            pos_encoding.reshape(
                pos_encoding.shape[0], num_channels_2, self.d, self.h, self.w
            ),
            masks,
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

    def training_step(self, batch, batch_idx):
        counts, shoebox, masks, reference = batch
        outputs = self(counts, shoebox, masks, reference)

        # Calculate loss
        (loss, neg_ll, kl, kl_bg, kl_I, kl_p) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_I=outputs["qI"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Track gradient norms here
        norms = grad_norm(self, norm_type=2)
        for name, norm in norms.items():
            self.log(f"grad_norm/{name}", norm)

        renyi_loss = (
            (-torch.log(outputs["qp"].rsample([100]).permute(1, 0, 2).pow(2).sum(-1)))
            .mean(1)
            .sum()
        ) * self.renyi_scale
        self.log("renyi_loss", renyi_loss)

        # Log metrics
        self.log("Train: -ELBO", loss.mean())
        self.log("Train: NLL", neg_ll.mean())
        self.log("Train: KL", kl.mean())
        self.log("Train: KL Bg", kl_bg.mean())
        self.log("Train: KL I", kl_I.mean())
        self.log("Train: KL Prf", kl_p.mean())
        self.log("Mean(qI.mean)", outputs["qI"].mean.mean())
        self.log("Min(qI.mean)", outputs["qI"].mean.min())
        self.log("Max(qI.mean)", outputs["qI"].mean.max())
        self.log("Mean(qbg.mean)", outputs["qbg"].mean.mean())
        self.log("Min(qbg.mean)", outputs["qbg"].mean.min())
        self.log("Max(qbg.mean)", outputs["qbg"].mean.max())
        self.log("Mean(qbg.variance)", outputs["qbg"].variance.mean())
        self.train_loss.append(loss.mean())
        return loss.mean() + renyi_loss.sum()

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        counts, shoebox, masks, reference = batch
        outputs = self(counts, shoebox, masks, reference)

        (
            loss,
            neg_ll,
            kl,
            kl_bg,
            kl_I,
            kl_p,
        ) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_I=outputs["qI"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Log metrics
        self.log("Val: -ELBO", loss.mean())
        self.log("Val: NLL", neg_ll.mean())
        self.log("Val: KL", kl.mean())
        self.log("Val: KL bg", kl_bg.mean())
        self.log("Val: KL I", kl_I.mean())
        self.log("Val: KL prf", kl_p.mean())
        self.log("val_loss", neg_ll.mean())

        return outputs

    def predict_step(self, batch, batch_idx):
        counts, shoebox, masks, reference = batch
        outputs = self(counts, shoebox, masks, reference)

        intensities = self.calculate_intensities(
            counts=outputs["counts"],
            qbg=outputs["qbg"],
            qp=outputs["qp"],
            masks=outputs["masks"],
        )

        return {
            "intensity_mean": outputs["intensity_mean"],  # qI.mean
            "intensity_var": outputs["intensity_var"],  # qI.variance
            "refl_ids": outputs["refl_ids"],
            "dials_I_sum_var": outputs["dials_I_sum_var"],
            "dials_I_prf_value": outputs["dials_I_prf_value"],
            "dials_I_prf_var": outputs["dials_I_prf_var"],
            "qbg": outputs["qbg"].mean,
            "qbg_scale": outputs["qbg"].scale,  # halfnormal param
            "profile_masking_mean": intensities["profile_masking_mean"],
            "profile_masking_var": intensities["profile_masking_var"],
            "kabsch_sum_mean": intensities["kabsch_sum_mean"],
            "kabsch_sum_var": intensities["kabsch_sum_var"],
            "x_c": outputs["x_c"],
            "y_c": outputs["y_c"],
            "z_c": outputs["z_c"],
        }
