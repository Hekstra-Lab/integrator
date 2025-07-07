from abc import ABC, abstractmethod
from typing import Any

import polars as plr
import pytorch_lightning as pl
import torch


class BaseIntegrator(pl.LightningModule, ABC):
    @abstractmethod
    def __init__(
        self,
        qbg: Any,
        qp: Any,
        qi: Any,
        loss: Any,
        d: int = 3,
        h: int = 21,
        w: int = 21,
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        mc_samples: int = 100,
        max_iterations: int = 4,
        renyi_scale: float = 0.00,
    ):
        super().__init__()
        self.qbg = qbg
        self.qp = qp
        self.qi = qi
        self.d = d
        self.h = h
        self.w = w
        self.loss = loss
        self.renyi_scale = renyi_scale

        # lists to track avg traning metrics
        self.train_loss = []
        self.train_kl = []
        self.train_nll = []

        # lists to track avg validation metrics
        self.val_loss = []
        self.val_kl = []
        self.val_nll = []
        self.lr = lr
        self.automatic_optimization = True
        self.weight_decay = weight_decay
        self.mc_samples = mc_samples
        self.max_iterations = max_iterations

        # dataframes to keep track of val/train epoch metrics
        self.schema = [
            ("epoch", int),
            ("avg_loss", float),
            ("avg_kl", float),
            ("avg_nll", float),
        ]
        self.train_df = plr.DataFrame(schema=self.schema)
        self.val_df = plr.DataFrame(schema=self.schema)

    def calculate_intensities(self, counts, qbg, qp, masks):
        with torch.no_grad():
            counts = counts * masks  # [B,P]
            zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
            zp = qp.mean.unsqueeze(1)

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

    @abstractmethod
    def forward(self, *args, **kwargs):
        "Forward method to be implemented by the subclass integrator"
        pass

    def on_train_epoch_end(self):
        # calculate epoch averages
        avg_train_loss = sum(self.train_loss) / len(self.train_loss)
        avg_kl = sum(self.train_kl) / len(self.train_kl)
        avg_nll = sum(self.train_nll) / len(self.train_nll)

        # log averages to weights & biases
        self.log("train_loss", avg_train_loss)
        self.log("avg_kl", avg_kl)
        self.log("avg_nll", avg_nll)

        # create epoch dataframe
        epoch_df = plr.DataFrame(
            {
                "epoch": self.current_epoch,
                "avg_loss": avg_train_loss,
                "avg_kl": avg_kl,
                "avg_nll": avg_nll,
            }
        )

        # udpate training dataframe
        self.train_df = plr.concat([self.train_df, epoch_df])

        # clear all lists
        self.train_loss = []
        self.train_kl = []
        self.train_nll = []

    def on_validation_epoch_end(self):
        avg_val_loss = sum(self.val_loss) / len(self.val_loss)
        avg_kl = sum(self.val_kl) / len(self.val_kl)
        avg_nll = sum(self.val_nll) / len(self.val_nll)

        self.log("validation_loss", avg_val_loss)
        self.log("validation_avg_kl", avg_kl)
        self.log("validation_avg_nll", avg_nll)

        epoch_df = plr.DataFrame(
            {
                "epoch": self.current_epoch,
                "avg_loss": avg_val_loss,
                "avg_kl": avg_kl,
                "avg_nll": avg_nll,
            }
        )
        self.val_df = plr.concat([self.val_df, epoch_df])

        self.val_loss = []
        self.avg_kl = []
        self.val_nll = []

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch ():
            batch_idx ():

        Returns:

        """
        counts, shoebox, masks, reference = batch
        outputs = self(counts, shoebox, masks, reference)

        # Calculate loss
        (loss, neg_ll, kl, kl_bg, kl_i, kl_p) = self.loss(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_i=outputs["qi"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

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
        self.log("Train: KL I", kl_i.mean())
        self.log("Train: KL Prf", kl_p.mean())
        self.log("Mean(qi.mean)", outputs["qi"].mean.mean())
        self.log("Min(qi.mean)", outputs["qi"].mean.min())
        self.log("Max(qi.mean)", outputs["qi"].mean.max())
        self.log("Mean(qbg.mean)", outputs["qbg"].mean.mean())
        self.log("Min(qbg.mean)", outputs["qbg"].mean.min())
        self.log("Max(qbg.mean)", outputs["qbg"].mean.max())
        self.log("Mean(qbg.variance)", outputs["qbg"].variance.mean())
        self.train_loss.append(loss.mean())
        self.train_kl.append(kl.mean())
        self.train_nll.append(neg_ll.mean())
        return loss.mean() + renyi_loss.sum()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def validation_step(self, batch, batch_idx):
        """

        Args:
            batch ():
            batch_idx ():

        Returns:

        """
        # Unpack batch
        counts, shoebox, masks, reference = batch
        outputs = self(counts, shoebox, masks, reference)

        (
            loss,
            neg_ll,
            kl,
            kl_bg,
            kl_i,
            kl_p,
        ) = self.loss(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_i=outputs["qi"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Log metrics
        self.log("Val: -ELBO", loss.mean())
        self.log("Val: NLL", neg_ll.mean())
        self.log("Val: KL", kl.mean())
        self.log("Val: KL bg", kl_bg.mean())
        self.log("Val: KL I", kl_i.mean())
        self.log("Val: KL prf", kl_p.mean())
        self.log("val_loss", neg_ll.mean())

        self.val_loss.append(loss.mean())
        self.val_kl.append(kl.mean())
        self.val_nll.append(neg_ll.mean())

        return outputs

    def predict_step(self, batch, batch_idx):
        """

        Args:
            batch ():
            batch_idx ():

        Returns:

        """
        counts, shoebox, masks, reference = batch
        outputs = self(counts, shoebox, masks, reference)

        return {
            "intensity_mean": outputs["intensity_mean"],  # qi.mean
            "intensity_var": outputs["intensity_var"],  # qi.variance
            "refl_ids": outputs["refl_ids"],
            "dials_I_sum_var": outputs["dials_I_sum_var"],
            "dials_I_prf_value": outputs["dials_I_prf_value"],
            "dials_I_prf_var": outputs["dials_I_prf_var"],
            "qbg": outputs["qbg"],
            "qbg_scale": outputs["qbg"].scale,  # halfnormal param
            "x_c": outputs["x_c"],
            "y_c": outputs["y_c"],
            "z_c": outputs["z_c"],
        }


if __name__ == "__main__":
    pass
