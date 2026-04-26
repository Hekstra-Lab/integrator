# Script to randomly sample the same number of refl ids from resolution bins
import polars as pl
import torch

# should contain either a resolution 'd' vector or another target to group by
# For monochromatic, we can use d
# For Laue we can bin by image and then by detector panel

meta = torch.load("/path/to/metadta")

df = pl.DataFrame({"d": meta["d"], "refl_ids": meta["refl_ids"]})

quantiles = torch.linspace(0.005, 1, 20).tolist()
res_bins = [df.get_column("d").quantile(q) for q in quantiles]

df = df.with_columns(pl.col("d").cut(breaks=res_bins).alias("db

test_refl_ids = []
for (dbin,), grouped_df in df.group_by("dbin"):
    n = grouped_df.height
    n_samples = int(n * 0.05)
    test_refl_ids.extend(grouped_df.sample(n_samples, seed=55)["refl_ids"].to_list())

test_ids = pl.Series("refl_ids", test_refl_ids)

df = df.with_columns(pl.col("refl_ids").is_in(test_ids.implode()).alias("is_test"))
