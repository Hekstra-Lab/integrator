from dials.array_family import flex
import torch
import polars as plr
import numpy as np
import glob
import os

def reflection_file_writer(prediction_directories, prediction_files, refl_file):
    # load reference reflection table
    refl_tbl = flex.reflection_table.from_file(refl_file)

    # check if prediction files exist
    if (len(prediction_files) == 0) or (len(prediction_directories) == 0):
        raise FileNotFoundError("No prediction files found")
    else:
        for pred_dir in prediction_directories:
            # Reinitialize empty_df for each prediction directory
            empty_df = plr.DataFrame()

            # store all predictions in a dataframe
            for pred in glob.glob(pred_dir + "/*.pt"):
                empty_df = empty_df.vstack(plr.DataFrame(torch.load(pred)))

            if empty_df.is_empty():
                continue  # Skip this iteration if no predictions were loaded

            # create a boolean array to select reflections used during training
            sel = np.asarray([False] * len(refl_tbl))
            reflection_ids = empty_df["refl_ids"].cast(plr.Int32).to_list()

            qI_mean_list = empty_df["qI_mean"].to_list()
            qI_variance_list = empty_df["qI_variance"].to_list()
            weighted_sum_mean = empty_df["weighted_sum_mean"].to_list()
            weighted_sum_var = empty_df["weighted_sum_var"].to_list()
            thresholded_mean = empty_df["thresholded_mean"].to_list()
            thresholded_var = empty_df["thresholded_var"].to_list()

            for id in reflection_ids:
                sel[id] = True

            temp = refl_tbl.select(flex.bool(sel))
            os.makedirs(pred_dir + "/reflections", exist_ok=True)

            temp["intensity.sum.value"] = flex.double(qI_mean_list)
            temp["intensity.sum.variance"] = flex.double(qI_variance_list)
            temp["intensity.prf.value"] = flex.double(qI_mean_list)
            temp["intensity.prf.variance"] = flex.double(qI_variance_list)
            temp.as_file(pred_dir + "/reflections/posterior_.refl")

            temp["intensity.sum.value"] = flex.double(thresholded_mean)
            temp["intensity.sum.variance"] = flex.double(thresholded_var)
            temp["intensity.prf.value"] = flex.double(thresholded_mean)
            temp["intensity.prf.variance"] = flex.double(thresholded_var)
            temp.as_file(pred_dir + "/reflections/thresholded_.refl")

            temp["intensity.sum.value"] = flex.double(weighted_sum_mean)
            temp["intensity.sum.variance"] = flex.double(weighted_sum_var)
