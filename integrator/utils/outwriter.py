from dials.array_family import flex
import polars as pl
import numpy as np


class OutWriter:
    def __init__(self, output_dict, refl_file_name, out_file_name):
        self.output_dict = output_dict
        self.refl_file_name = refl_file_name
        self.out_file_name = out_file_name

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
        res_df = pl.concat([train_res_df, val_res_df]).sort(pl.col("refl_id"))
        tbl = flex.reflection_table.from_file(self.refl_file_name)
        sel = np.asarray([False] * len(tbl))
        reflection_ids = res_df["refl_id"].cast(pl.Int32).to_list()
        intensity_preds = res_df["q_I_mean"].to_list()
        intensity_stddev = res_df["q_I_stddev"].to_list()
        for id in reflection_ids:
            sel[id] = True
        temp = tbl.select(flex.bool(sel))
        temp["intensity.sum.value"] = flex.double(intensity_preds)
        temp["intensity.sum.variance"] = flex.double(intensity_stddev)
        temp.as_file(self.out_file_name)
        return


# def OutWriter(output_dict, refl_file_name, out_file_name):
# """
# Write the output of the model to a reflection file.

# Args:
# output_dict (): The output dictionary from the model.
# refl_file_name (): The name of the input reflection file.
# out_file_name (): The name of the output reflection file.
# """
# train_res_df = pl.DataFrame(
# {
# "refl_id": output_dict.training_preds["refl_id"],
# "q_I_mean": output_dict.training_preds["q_I_mean"],
# "q_I_stddev": output_dict.training_preds["q_I_stddev"],
# }
# )
# val_res_df = pl.DataFrame(
# {
# "refl_id": output_dict.validation_preds["refl_id"],
# "q_I_mean": output_dict.validation_preds["q_I_mean"],
# "q_I_stddev": output_dict.validation_preds["q_I_stddev"],
# }
# )
# res_df = pl.concat([train_res_df, val_res_df]).sort(pl.col("refl_id"))
# tbl = flex.reflection_table.from_file(refl_file_name)
# sel = np.asarray([False] * len(tbl))
# reflection_ids = res_df["refl_id"].cast(pl.Int32).to_list()
# intensity_preds = res_df["q_I_mean"].to_list()
# intensity_stddev = res_df["q_I_stddev"].to_list()
# for id in reflection_ids:
# sel[id] = True
# temp = tbl.select(flex.bool(sel))
# temp["intensity.sum.value"] = flex.double(intensity_preds)
# temp["intensity.sum.variance"] = flex.double(intensity_stddev)
# temp.as_file(out_file_name)
# return
