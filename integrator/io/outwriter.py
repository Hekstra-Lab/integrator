import polars as pl
import numpy as np
import glob
import os
from dials.array_family import flex

# Assume model.training_preds and model.validation_preds are already defined
# Convert to numpy arrays and ensure they are of integer type

intensity_preds = np.array(model.training_preds["q_I_mean"], dtype=np.float32)

refl_ids_train = np.array(model.training_preds["refl_id"], dtype=np.int32)
tbl_ids_train = np.array(model.training_preds["tbl_id"], dtype=np.int32)
refl_ids_val = np.array(model.validation_preds["refl_id"], dtype=np.int32)
tbl_ids_val = np.array(model.validation_preds["tbl_id"], dtype=np.int32)

# Training predictions
train_res_df = pl.DataFrame(
    {
        "tbl_id": tbl_ids_train,
        "refl_id": refl_ids_train,
        "q_I_mean": model.training_preds["q_I_mean"],
        "q_I_stddev": model.training_preds["q_I_stddev"],
    }
)

# Validation predictions
val_res_df = pl.DataFrame(
    {
        "tbl_id": tbl_ids_val,
        "refl_id": refl_ids_val,
        "q_I_mean": model.validation_preds["q_I_mean"],
        "q_I_stddev": model.validation_preds["q_I_stddev"],
    }
)

# Concatenate train_res_df and val_res_df
res_df = pl.concat([train_res_df, val_res_df])

tbl_ids = res_df["tbl_id"].unique().to_list()

# Directory to shoeboxes
refl_dir = "/Users/luis/integrator/rotation_data_examples/data/"

# shoebox filenames
filenames = glob.glob(os.path.join(refl_dir, "shoebox*"))

refl_tables = [flex.reflection_table.from_file(filename) for filename in filenames]

# Iterate over reflection id
for tbl_id in tbl_ids:
    sel = np.asarray([False] * len(refl_tables[tbl_id]))

    filtered_df = res_df.filter(res_df["tbl_id"] == tbl_id)

    # Reflection ids
    reflection_ids = filtered_df["refl_id"].to_list()

    # Intensity predictions
    intensity_preds = filtered_df["q_I_mean"].to_list()
    intensity_stddev = filtered_df["q_I_stddev"].to_list()

    for i, id in enumerate(reflection_ids):
        sel[id] = True

    refl_temp_tbl = refl_tables[tbl_id].select(flex.bool(sel))

    refl_temp_tbl["intensity.sum.value"] = flex.double(intensity_preds)

    refl_temp_tbl["intensity.sum.variance"] = flex.double(intensity_stddev)

    # save the updated reflection table
    refl_temp_tbl.as_file(f"integrator_preds_{tbl_id}_test.refl")
