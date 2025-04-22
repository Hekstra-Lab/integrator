from dials.array_family import flex
from pathlib import Path
import torch
import json
import polars as plr
import numpy as np
import glob
import os


def reflection_file_writer(
    prediction_directories,
    prediction_files,
    refl_file,
    # optional job_id string
    job_id="",
):
    # load reference reflection table
    refl_tbl = flex.reflection_table.from_file(refl_file)
    file_dict = {"refl_files": []}

    # check if prediction files exist
    if (len(prediction_files) == 0) or (len(prediction_directories) == 0):
        raise FileNotFoundError("No prediction files found")
    else:
        for pred_dir in prediction_directories:
            # Reinitialize empty_df for each prediction directory
            empty_df = plr.DataFrame()

            # store all predictions in a dataframe
            for pred in glob.glob(pred_dir + "/*.pt"):
                preds = torch.load(pred, weights_only=False)
                data_dict = {key: np.concatenate(preds[key]) for key in preds}
                empty_df = empty_df.vstack(plr.DataFrame(data_dict))

            if empty_df.is_empty():
                continue  # Skip this iteration if no predictions were loaded

            empty_df = empty_df.sort(plr.col("refl_ids"))

            # create a boolean array to select reflections used during training
            sel = np.asarray([False] * len(refl_tbl))
            reflection_ids = empty_df["refl_ids"].explode().cast(plr.Int32).to_list()

            qI_mean_list = empty_df["qI_mean"].explode().to_list()
            qI_variance_list = empty_df["qI_variance"].explode().to_list()
            weighted_sum_mean = empty_df["weighted_sum_mean"].explode().to_list()
            weighted_sum_var = empty_df["weighted_sum_var"].explode().to_list()
            thresholded_mean = empty_df["thresholded_mean"].explode().to_list()
            thresholded_var = empty_df["thresholded_var"].explode().to_list()

            for id in reflection_ids:
                sel[id] = True

            temp = refl_tbl.select(flex.bool(sel))
            os.makedirs(pred_dir + "/reflections", exist_ok=True)

            temp["intensity.sum.value"] = flex.double(qI_mean_list)
            temp["intensity.sum.variance"] = flex.double(qI_variance_list)
            temp["intensity.prf.value"] = flex.double(qI_mean_list)
            temp["intensity.prf.variance"] = flex.double(qI_variance_list)
            temp.as_file(pred_dir + "/reflections/posterior_.refl")
            posterior_refl = Path(pred_dir + "/reflections/posterior_.refl")

            file_dict["refl_files"].append(posterior_refl.resolve().as_posix())

            temp["intensity.sum.value"] = flex.double(thresholded_mean)
            temp["intensity.sum.variance"] = flex.double(thresholded_var)
            temp["intensity.prf.value"] = flex.double(thresholded_mean)
            temp["intensity.prf.variance"] = flex.double(thresholded_var)
            temp.as_file(pred_dir + "/reflections/thresholded_.refl")
            thresholded_refl = Path(pred_dir + "/reflections/thresholded_.refl")

            file_dict["refl_files"].append(posterior_refl.resolve().as_posix())

            temp["intensity.sum.value"] = flex.double(weighted_sum_mean)
            temp["intensity.sum.variance"] = flex.double(weighted_sum_var)
            temp["intensity.prf.value"] = flex.double(weighted_sum_mean)
            temp["intensity.prf.variance"] = flex.double(weighted_sum_var)
            temp.as_file(pred_dir + "/reflections/weighted_.refl")
            weighted_refl = Path(pred_dir + "/reflections/weighted_.refl")
            file_dict["refl_files"].append(weighted_refl.resolve().as_posix())

        # write the paths to a json file
        # refl_file_path = Path(weighted_refl.parents[2] / "paths.json")
        with open(f"paths_{job_id}.json", "w") as f:
            json.dump(file_dict, f, indent=4)
