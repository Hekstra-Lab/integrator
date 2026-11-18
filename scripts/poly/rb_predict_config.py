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

# Values that mean "this behaviour is off". An argument the current code no
# longer accepts can be dropped when it holds one of these, because the code
# that read it is gone and the run was not using it either. Any other value
# means the run used behaviour this version does not have, and re-exporting
# it would silently produce something else.
INERT = (0, 0.0, False, None, "")


def strip_unknown(name: str, args: dict) -> list[str]:
    """Drop args the installed surrogate no longer takes, if they were off.

    Code drift is the reason this exists. `geometric_basis_profile` took
    `smoothness_weight` when the PDZ2 arms were trained and does not now
    (the profile basis penalties were removed), so a September config cannot
    be rebuilt by today's factory at all.
    """
    from integrator.registry import REGISTRY

    cls = REGISTRY["surrogate"].get(name)
    if cls is None:
        return []
    import inspect

    valid = set(inspect.signature(cls.__init__).parameters)
    dropped = []
    for key in [k for k in args if k not in valid and k != "self"]:
        if args[key] in INERT:
            dropped.append(key)
            args.pop(key)
        else:
            raise SystemExit(
                f"{name}.{key} = {args[key]!r} is not accepted by the "
                f"installed version and is not an inert value. This run used "
                f"behaviour the current code does not have; re-exporting it "
                f"would produce something else. Check out the commit it was "
                f"trained with, or decide the difference does not matter and "
                f"say so explicitly."
            )
    return dropped


run_dir, out = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(f"{run_dir}/config_log.yaml").read())
args = cfg["integrator"]["args"]
keys = list(args.get("predict_keys") or [])
added = [k for k in RB_KEYS if k not in keys]
args["predict_keys"] = keys + added

for slot, spec in (cfg.get("surrogates") or {}).items():
    if isinstance(spec, dict) and isinstance(spec.get("args"), dict):
        gone = strip_unknown(str(spec.get("name", "")), spec["args"])
        if gone:
            print(f"  {slot}: dropped {gone} (inert, and removed from the code)")

yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
print(f"  predict_keys += {added}")
