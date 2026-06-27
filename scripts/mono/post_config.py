import argparse
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from integrator.cli.utils.logger import setup_logging
from integrator.utils import load_config

logger = logging.getLogger(__name__)

_COLUMN_SPECS = {
    "intensity": ("I(+),SIGI(+),I(-),SIGI(-)", "intensity"),
    "amplitude": ("F(+),SIGF(+),F(-),SIGF(-)", "amplitude"),
}


@dataclass
class Config:
    """Schema for `dials_phenix_cfg.yaml`"""

    refl_files: list  # list of refl files to process
    expt_file: str  # dials.expt file for data used to make dataset
    dials_env: str  # environment with dials
    phenix_env: str  # path to phenix
    phenix_eff: str  # phenix.eff file used on reference data
    pdb: str  # reference pdb file
    columns: str = "intensity"
    merged_mtz: list | None = None
    user_selected_labels: str = ""
    french_wilson_scale: bool = True


def open_run(
    run_dir,
    run_paths: str = "run_paths.yaml",
) -> tuple[dict, Path]:
    """Resolve a run directory to its config and W&B log directory.

    Args:
        run_dir: Path to a run directory containing `run_paths.yaml`.
    """
    run_dir = Path(run_dir)
    run_metadata = next(iter(run_dir.glob(run_paths)))
    config = load_config(run_metadata)

    # path to predictions
    pred_dir = Path(config["predictions_dir"])
    return config, pred_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare dials_phenix_cfg.yaml file to run DIALS/PHENIX",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory (contains run_metadata.yaml and config_copy.yaml)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
    )

    # Intensity vs amplitude
    parser.add_argument(
        "--columns",
        choices=["intensity", "amplitude"],
        default="intensity",
        help="Which merged columns phenix should read. 'amplitude' uses "
        "F(+)/F(-) from the model's merged MTZ and turns French-Wilson off.",
    )
    parser.add_argument(
        "--process-cfg",
        type=str,
        help="configuration file for post-processing",
    )

    parser.add_argument(
        "--french-wilson",
        choices=["auto", "true", "false"],
        default="auto",
        help="Override french_wilson_scale. 'auto' = off for amplitude, on for "
        "intensity.",
    )
    parser.add_argument(
        "--merged-mtz",
        type=str,
        default=None,
        help="Explicit merged MTZ path (amplitude mode). Default: discover "
        "predictions/**/merged.mtz under the run's log dir.",
    )

    # Dataset paths — override what's in config_copy.yaml
    parser.add_argument("--expt-file", type=str, default=None)
    parser.add_argument("--pdb", type=str, default=None)
    parser.add_argument("--phenix-eff", type=str, default=None)
    parser.add_argument("--paired-ref-eff", type=str, default=None)
    parser.add_argument("--paired-model-eff", type=str, default=None)

    # Tool paths — override env vars DIALS_ENV / PHENIX_ENV
    parser.add_argument("--dials-env", type=str, default=None)
    parser.add_argument("--phenix-env", type=str, default=None)
    return parser.parse_args()


def _resolve(cli_val, cfg_val, env_var, label):
    if cli_val is not None:
        return cli_val
    if cfg_val is not None:
        return str(cfg_val)
    env = os.environ.get(env_var) if env_var else None
    if env is not None:
        return env
    raise ValueError(
        f"Missing '{label}'. Set it in the training config output section, "
        f"pass --{label.replace('_', '-')}, "
        + (f"or set ${env_var}" if env_var else "")
    )


# phenix.eff rendering
def _set_array_type_star(line: str, star_token: str) -> str:
    """Move the `*` in an array_type listing to the requested token."""
    tokens = re.findall(r"\S+", line)
    out_tokens = []
    for t in tokens:
        bare = t.lstrip("*")
        out_tokens.append(f"*{bare}" if bare == star_token else bare)
    leading = line[: len(line) - len(line.lstrip())]
    return leading + " ".join(out_tokens) + "\n"


def render_eff(
    template: str,
    labels: str,
    star_token: str,
    fw_scale: bool,
) -> str:
    """Render a phenix.eff from the template for the chosen columns."""
    miller_array_count = 0
    in_data_block = False
    array_type_pending = False
    out_lines = []

    for line in template.splitlines(keepends=True):
        stripped = line.strip()

        if stripped.startswith("miller_array"):
            miller_array_count += 1
            in_data_block = miller_array_count == 1
            out_lines.append(line)
            continue

        if in_data_block and stripped.startswith("name ="):
            out_lines.append(
                re.sub(r'name = "[^"]*"', f'name = "{labels}"', line)
            )
            array_type_pending = True
            continue

        if in_data_block and array_type_pending:
            out_lines.append(_set_array_type_star(line, star_token))
            if not line.rstrip().endswith("\\"):
                array_type_pending = False
            continue

        if in_data_block and stripped.startswith("user_selected_labels"):
            out_lines.append(
                re.sub(
                    r'user_selected_labels = "[^"]*"',
                    f'user_selected_labels = "{labels}"',
                    line,
                )
            )
            in_data_block = False
            continue

        if "french_wilson_scale" in line:
            out_lines.append(
                re.sub(
                    r"french_wilson_scale\s*=\s*\w+",
                    f"french_wilson_scale = {fw_scale}",
                    line,
                )
            )
            continue

        out_lines.append(line)

    return "".join(out_lines)


def main():
    # get args
    args = parse_args()

    # setup logger
    setup_logging(args.verbose)

    # path to run-dir
    run_dir = args.run_dir.resolve()
    logger.info(f"Run directory: {run_dir}")

    # Load run paths for wandb log dir
    _, pred_dir = open_run(run_dir)
    logger.info(f"Prediction directory: {pred_dir}")

    # get per-observation .refl files or mtz files
    refl_files = sorted(x.as_posix() for x in pred_dir.glob("**/preds*.refl"))
    if args.merged_mtz is not None:
        merged_mtzs = [str(Path(args.merged_mtz).resolve())]
    else:
        merged_mtzs = sorted(
            x.as_posix() for x in pred_dir.glob("**/merged.mtz")
        )
    logger.info(
        f"Found {len(refl_files)} .refl files, {len(merged_mtzs)} merged MTZs"
    )
    if not refl_files and not merged_mtzs:
        raise FileNotFoundError(
            f"No preds*.refl and no merged.mtz found in {pred_dir}"
        )

    # Resolve French-Wilson: 'auto' -> off for amplitude, on for intensity.
    if args.french_wilson == "auto":
        fw_scale = args.columns == "intensity"
    else:
        fw_scale = args.french_wilson == "true"
    labels, star_token = _COLUMN_SPECS[args.columns]

    # Load training config for dataset-specific paths
    if args.process_cfg:
        output_cfg = load_config(args.process_cfg)
    else:
        logger.warning(f"No config_log.yaml in {run_dir}, using CLI/env only")
        output_cfg = {}

    phenix_eff_template = _resolve(
        args.phenix_eff,
        output_cfg.get("phenix_eff"),
        None,
        "phenix_eff",
    )

    phenix_eff = phenix_eff_template
    if args.columns == "amplitude" or not fw_scale:
        if not merged_mtzs:
            logger.warning(
                "columns=%s but no merged.mtz found; phenix has no amplitude "
                "data to read.",
                args.columns,
            )
        template_text = Path(phenix_eff_template).read_text()
        rendered = render_eff(template_text, labels, star_token, fw_scale)
        rendered_path = run_dir / f"phenix_{args.columns}.eff"
        rendered_path.write_text(rendered)
        phenix_eff = str(rendered_path)
        logger.info(
            "Rendered %s (labels=%s, array_type=%s, french_wilson_scale=%s)",
            rendered_path,
            labels,
            star_token,
            fw_scale,
        )

    config = Config(
        refl_files=refl_files,
        expt_file=_resolve(
            args.expt_file,
            output_cfg.get("expt_file"),
            None,
            "expt_file",
        ),
        dials_env=_resolve(
            args.dials_env,
            output_cfg.get("dials_env"),
            "DIALS_ENV",
            "dials_env",
        ),
        phenix_env=_resolve(
            args.phenix_env,
            output_cfg.get("phenix_env"),
            "PHENIX_ENV",
            "phenix_env",
        ),
        phenix_eff=phenix_eff,
        pdb=_resolve(
            args.pdb,
            output_cfg.get("pdb"),
            None,
            "pdb",
        ),
        # paired_ref_eff=_resolve_optional(
        #     args.paired_ref_eff,
        #     output_cfg.get("paired_ref_eff"),
        # ),
        # paired_model_eff=_resolve_optional(
        #     args.paired_model_eff,
        #     output_cfg.get("paired_model_eff"),
        # ),
        columns=args.columns,
        merged_mtz=merged_mtzs,
        user_selected_labels=labels,
        french_wilson_scale=fw_scale,
    )

    cfg_fname = run_dir / "dials_phenix_cfg.yaml"
    cfg_fname.write_text(yaml.safe_dump(asdict(config)))
    logger.info(f"Wrote config to: {cfg_fname}")


if __name__ == "__main__":
    main()
