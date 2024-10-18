import torch
import polars as pl
import numpy as np
from dials.array_family import flex


class OutWriter:
    """
    Writes the output of the neural network to a dials reflection file.

    Attributes:
        predictions:
        reflection_file:
        out_file_name:
        out_file_name2:
        out_file_name3:
        dirichlet:
    """

    def __init__(
        self,
        predictions,
        reflection_file,  # dials reflections file
        out_file_name,
        out_file_name2="nn_only.refl",
        out_file_name3="dials_sum_nn_weak.refl",
        dirichlet=False,
    ):
        self.predictions = predictions
        self.reflection_file = reflection_file
        self.out_file_name = out_file_name
        self.out_file_name2 = out_file_name2
        self.out_file_name3 = out_file_name3
        self.dirichlet = dirichlet

    def tensor_to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().detach().numpy()
        return tensor

    def write_output(self):
        res_df = pl.DataFrame(
            {
                "refl_id": self.tensor_to_numpy(self.predictions["refl_id"]),
                "q_I_mean": self.tensor_to_numpy(self.predictions["q_I_mean"]),
                "q_I_stddev": self.tensor_to_numpy(self.predictions["q_I_stddev"]),
                "I_weighted_sum": self.tensor_to_numpy(
                    self.predictions["weighted_sum"]
                ),
                "I_masked_sum": self.tensor_to_numpy(self.predictions["masked_sum"]),
            }
        )

        # adding variance column
        res_df = res_df.with_columns((pl.col("q_I_stddev") ** 2).alias("q_I_variance"))

        res_df = res_df.sort(pl.col("refl_id"))
        tbl = flex.reflection_table.from_file(self.reflection_file)
        sel = np.asarray([False] * len(tbl))
        reflection_ids = res_df["refl_id"].cast(pl.Int32).to_list()
        intensity_preds = res_df["q_I_mean"].to_list()
        intensity_variance = res_df["q_I_variance"].to_list()
        intensity_prf_preds = res_df["I_masked_sum"].to_list()
        intensity_wsum_preds = res_df["I_weighted_sum"].to_list()

        for id in reflection_ids:
            sel[id] = True

        # Replaces both intensity.sum.value and intensity.sum.variance columns with our model's predictions

        temp = tbl.select(flex.bool(sel))

        # combination of dials.sum and network predictions
        temp["intensity.prf.value"] = flex.double(intensity_preds)
        temp["intensity.prf.variance"] = flex.double(intensity_variance)
        temp["I_masked_sum"] = flex.double(intensity_prf_preds)
        temp["I_weighted_sum"] = flex.double(intensity_wsum_preds)
        temp.as_file(self.out_file_name)

        # network only predictions
        temp["intensity.sum.value"] = flex.double(intensity_preds)
        temp["intensity.sum.variance"] = flex.double(intensity_variance)
        temp["I_masked_sum"] = flex.double(intensity_prf_preds)
        temp["I_weighted_sum"] = flex.double(intensity_wsum_preds)
        temp.as_file(self.out_file_name2)

        # Replaces intensity.prf.value and intensity.prf.variance columns with intensity.sum.value
        # and intensity.sum.variance
        # Then, it replaces reflections used during training

        indices = np.where(sel)
        intensity_dials_sum = np.array(list(tbl["intensity.sum.value"]))
        intensity_dials_variance = np.array(list(tbl["intensity.sum.variance"]))
        intensity_dials_sum[indices] = intensity_preds
        intensity_dials_variance[indices] = intensity_variance
        temp2 = tbl
        temp2["intensity.prf.value"] = flex.double(intensity_dials_sum)
        temp2["intensity.prf.variance"] = flex.double(intensity_dials_variance)
        intensity_dials_sum[indices] = intensity_prf_preds

        temp2["I_masked_sum"] = flex.double(intensity_dials_sum)
        temp["I_weighted_sum"] = flex.double(intensity_wsum_preds)
        temp2.as_file(self.out_file_name3)

        return sel
