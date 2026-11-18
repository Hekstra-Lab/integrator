"""Derive a predict config that asks for the Rao-Blackwellized columns.

`--rb-export` only tells the MTZ writer which columns to read out of the
prediction parquet; the parquet's columns come from `predict_keys`, so a
config that does not list `i_rb` and `var_rb` produces a parquet without
them and the export fails on a KeyError at the very end.

The config is derived from the RUN'S OWN config_log.yaml, never from the
repository config: the checkpoint has to load into the architecture it was
trained with, and the PDZ2 arms differ from each other and from the
repository file in window and kernel size.
"""

import sys

import yaml

RB_KEYS = ("i_rb", "var_rb", "var_rb_cond", "var_rb_profile")

run_dir, out = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(f"{run_dir}/config_log.yaml").read())
args = cfg["integrator"]["args"]
keys = list(args.get("predict_keys") or [])
added = [k for k in RB_KEYS if k not in keys]
args["predict_keys"] = keys + added
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
print(f"  predict_keys += {added}")
