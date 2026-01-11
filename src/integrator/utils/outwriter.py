import numpy as np
import polars as pl
from dials.array_family import flex


class OutWriter:
    def __init__(
        self,
        output_dict,
        refl_file_name,
        out_file_name,
        out_file_name2="nn_only.refl",
        out_file_name3="dials_sum_nn_weak.refl",
    ):
        self.output_dict = output_dict
        self.refl_file_name = refl_file_name
        self.out_file_name = out_file_name
        self.out_file_name2 = out_file_name2
        self.out_file_name3 = out_file_name3

    def write_output(self):
        train_res_df = pl.DataFrame(
            {
                "refl_id": self.output_dict.training_preds["refl_id"],
                "q_I_mean": self.output_dict.training_preds["q_I_mean"],
                "q_I_stddev": self.output_dict.training_preds["q_I_stddev"],
            }
        )
        val_res_df = pl.DataFrame(
            {
                "refl_id": self.output_dict.validation_preds["refl_id"],
                "q_I_mean": self.output_dict.validation_preds["q_I_mean"],
                "q_I_stddev": self.output_dict.validation_preds["q_I_stddev"],
            }
        )

        # adding variance column
        train_res_df = train_res_df.with_columns(
            (pl.col("q_I_stddev") ** 2).alias("q_I_variance")
        )
        val_res_df = val_res_df.with_columns(
            (pl.col("q_I_stddev") ** 2).alias("q_I_variance")
        )

        res_df = pl.concat([train_res_df, val_res_df]).sort(pl.col("refl_id"))
        tbl = flex.reflection_table.from_file(self.refl_file_name)
        sel = np.asarray([False] * len(tbl))
        reflection_ids = res_df["refl_id"].cast(pl.Int32).to_list()
        intensity_preds = res_df["q_I_mean"].to_list()
        intensity_variance = res_df["q_I_variance"].to_list()

        for id in reflection_ids:
            sel[id] = True

        # Replaces both intensity.sum.value and intensity.sum.variance columns with our model's predictions
        temp = tbl.select(flex.bool(sel))

        # combination of dials.sum and network predictions
        temp["intensity.prf.value"] = flex.double(intensity_preds)
        temp["intensity.prf.variance"] = flex.double(intensity_variance)
        temp.as_file(self.out_file_name)

        # network only predictions
        temp["intensity.sum.value"] = flex.double(intensity_preds)
        temp["intensity.sum.variance"] = flex.double(intensity_variance)
        temp.as_file(self.out_file_name2)

        # Replaces intensity.prf.value and intensity.prf.variance columns with intensity.sum.value
        # and intensity.sum.variance
        # Then, it replaces reflections used during training

        indices = np.where(sel)
        intensity_dials_sum = np.array(list(tbl["intensity.sum.value"]))
        intensity_dials_variance = np.array(
            list(tbl["intensity.sum.variance"])
        )
        intensity_dials_sum[indices] = intensity_preds
        intensity_dials_variance[indices] = intensity_variance
        temp2 = tbl
        temp2["intensity.prf.value"] = flex.double(intensity_dials_sum)
        temp2["intensity.prf.variance"] = flex.double(intensity_dials_variance)
        temp2.as_file(self.out_file_name3)

        return
