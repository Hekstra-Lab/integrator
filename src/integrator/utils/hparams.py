"""Hyperparameters out of a run's own config log, in long format.

A paper needs a table of what each model was configured with. The settings
are already recorded -- `integrator.train` writes `config_log.yaml` into
every run directory -- so nothing new has to be logged during training and
no run has to be repeated. What was missing is the reading.

**Read the run's `config_log.yaml`, never the config in the repository.**
They are not the same file. The repository config is edited, and
`configs/poly/pdz2_1116.yaml` today says `hierarchical_3enc` while every
PDZ2 run on disk trained `hierarchical`. The run's own log is what actually
ran; the repository config is what someone later thought it was.

Three things this does beyond flattening a dict:

- **Repeated encoders collapse.** A config may list the same encoder four
  times through a YAML anchor -- the hierarchical integrator takes `k_i`,
  `r_i`, `k_bg` and `r_bg` -- and four identical rows in a paper table are
  noise. Identical consecutive entries become one row with `n` recorded.
- **Bookkeeping is dropped.** Output directories, checkpoint paths and
  logger settings are not hyperparameters and would swamp the table.
- **`varies` is computed across the set.** A parameter that is the same in
  every arm compared is context; one that differs is the comparison. A
  table of twenty arms is unreadable without that flag, and which
  parameters vary is a property of the SET, not of any one run, so it
  cannot be computed one run at a time.

The output is the long format the figures and tables read: one row per
(`method`, `dataset`, `group`, `parameter`), with `value`, `unit` and
`varies`, plus a labels file declaring how each parameter is set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "COLUMNS",
    "GROUPS",
    "LABEL_COLUMNS",
    "SKIP_CONTAINS",
    "SKIP_PREFIX",
    "flatten",
    "hparam_labels",
    "hparam_rows",
    "mark_varying",
    "read_config_log",
]

COLUMNS = ("method", "dataset", "group", "parameter", "value", "unit",
           "varies")
LABEL_COLUMNS = ("parameter", "header", "unit", "group", "fmt",
                 "description")

# Top-level config sections, and the group each is reported under. A section
# that is not listed is still read, under its own name, so a section added to
# the config later appears rather than vanishing.
GROUPS = {
    "integrator": "integrator",
    "encoders": "encoder",
    "surrogates": "surrogate",
    "loss": "loss",
    "optimizer": "optimizer",
    "data_loader": "data",
    "trainer": "trainer",
    "early_stop": "trainer",
    "checkpoint": "trainer",
}

# Paths that record where output went rather than what was configured.
# Matched as a prefix of the dotted path, or anywhere for the bare names.
SKIP_PREFIX = ("output", "checkpoint.dirpath", "trainer.default_root_dir",
               "trainer.logger", "mode")
SKIP_CONTAINS = ("data_dir", "out_dir", "run_dir", "save_dir", "log_dir",
                 "_path", "path", "wandb", "project")

# Units for the parameters that have one. Everything else is unitless and
# gets an empty string rather than a guess.
UNITS = {
    "max_epochs": "epochs",
    "warmup_epochs": "epochs",
    "patience": "epochs",
    "d": "px",
    "h": "px",
    "w": "px",
    "dmin": "AA",
    "lr": "",
    "batch_size": "reflections",
}

# Headers for the parameters a paper table is likely to name. Anything not
# listed falls back to its own parameter name, so the table is never missing
# a row because a label was not written.
HEADERS = {
    "integrator.name": "integrator",
    "surrogate.qp.name": "profile model",
    "surrogate.qi.name": "intensity model",
    "surrogate.qbg.name": "background model",
    "loss.name": "loss",
    "optimizer.lr": "learning rate",
    "trainer.max_epochs": "epochs",
    "encoder.profile_encoder.conv1_kernel_size": "first kernel",
}


def read_config_log(run_dir: str | Path) -> dict:
    """The config a run actually trained with, from its own directory."""
    import yaml

    path = Path(run_dir) / "config_log.yaml"
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _skip(path: str) -> bool:
    if any(path == p or path.startswith(p + ".") for p in SKIP_PREFIX):
        return True
    tail = path.rsplit(".", 1)[-1].lower()
    return any(s in tail for s in SKIP_CONTAINS)


def _same(a: Any, b: Any) -> bool:
    return repr(a) == repr(b)


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """A nested config as dotted paths.

    A list of dicts is indexed by its entries' `name` where they have one,
    because `encoders.0` says nothing and `encoders.profile_encoder` says
    what it is. Consecutive identical entries collapse to one path with an
    `n` recorded beside it, so four anchored copies of one encoder do not
    become four identical rows.
    """
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, sub in value.items():
            if key == "args" and isinstance(sub, dict):
                # `args` is a container, not a level a reader cares about
                out.update(flatten(sub, prefix))
                continue
            out.update(flatten(sub, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list) and value and isinstance(value[0], dict):
        seen: list[tuple[str, dict, int]] = []
        for entry in value:
            name = str(entry.get("name", len(seen)))
            if seen and seen[-1][0] == name and _same(seen[-1][1], entry):
                seen[-1] = (name, entry, seen[-1][2] + 1)
            else:
                seen.append((name, entry, 1))
        for name, entry, n in seen:
            out.update(flatten(entry, f"{prefix}.{name}" if prefix else name))
            if n > 1:
                out[f"{prefix}.{name}.n" if prefix else f"{name}.n"] = n
    else:
        out[prefix] = value
    return out


def _as_text(value: Any) -> str:
    """A value as the table stores it: faithful, not formatted.

    How a list of two integers is shown -- `[3, 3]` or `3x3` -- is a
    presentation decision and belongs with whatever sets the table, not
    here, so the stored value stays close to the config.
    """
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(_as_text(v) for v in value)
    return str(value)


def hparam_rows(cfg: dict, method: str, dataset: str = "") -> list[dict]:
    """One row per configured parameter, before `varies` is known."""
    rows = []
    for section, value in (cfg or {}).items():
        group = GROUPS.get(section, str(section))
        for path, raw in flatten(value, str(section)).items():
            if _skip(path):
                continue
            # the group replaces the section at the head of the path, so a
            # parameter is named the same way whatever file it came from
            tail = path.split(".", 1)[1] if "." in path else path
            name = f"{group}.{tail}"
            rows.append({
                "method": method,
                "dataset": dataset,
                "group": group,
                "parameter": name,
                "value": _as_text(raw),
                "unit": UNITS.get(tail.rsplit(".", 1)[-1], ""),
                "varies": "",
            })
    return rows


def mark_varying(rows: list[dict]) -> list[dict]:
    """Set `varies` per parameter, across the methods present.

    A parameter missing from one arm counts as varying: an arm that does not
    have a setting differs from one that does, and hiding that would make
    two different models look identically configured.
    """
    methods = {r["method"] for r in rows}
    seen: dict[str, set] = {}
    where: dict[str, set] = {}
    for r in rows:
        seen.setdefault(r["parameter"], set()).add(r["value"])
        where.setdefault(r["parameter"], set()).add(r["method"])
    for r in rows:
        name = r["parameter"]
        varies = len(seen[name]) > 1 or where[name] != methods
        r["varies"] = int(bool(varies))
    return rows


def hparam_labels(rows: list[dict]) -> list[dict]:
    """The labels file: how each parameter is named and set."""
    out, seen = [], set()
    for r in sorted(rows, key=lambda r: (r["group"], r["parameter"])):
        name = r["parameter"]
        if name in seen:
            continue
        seen.add(name)
        tail = name.rsplit(".", 1)[-1]
        out.append({
            "parameter": name,
            "header": HEADERS.get(name, tail.replace("_", " ")),
            "unit": r["unit"],
            "group": r["group"],
            "fmt": "s",
            "description": "",
        })
    return out
