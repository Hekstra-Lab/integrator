"""Print the fully-resolved config for a YAML file.

Expands `mode`, synthesized `encoders:`/`surrogates:`, and injected `encoder_out`
into the complete config that training would build, without running the model.

This is the same expansion saved to `config_log.yaml` during a run.
"""

import argparse
import sys

import yaml

from integrator.utils import load_config, resolve_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to the config YAML to resolve.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write the resolved config here instead of stdout.",
    )
    args = parser.parse_args()

    cfg = resolve_config(load_config(args.config))
    text = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
