"""``--config file.toml`` support. Values given on the command line override the TOML file.

TOML sections are only for readability; keys are matched to argument names.
"""

import argparse
import logging
from typing import Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def parse_with_config(parser: argparse.ArgumentParser, argv: Optional[list] = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)

    parser.add_argument("--config", metavar="FILE", help="TOML config file; CLI arguments override it.")
    if known.config:
        with open(known.config, "rb") as f:
            toml = tomllib.load(f)
        flat = {}
        for key, value in toml.items():
            if isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value

        dests = {a.dest for a in parser._actions}
        unknown = sorted(set(flat) - dests)
        if unknown:
            logging.getLogger(__name__).warning(f"{known.config}: ignoring unknown keys {unknown}")
        values = {k: v for k, v in flat.items() if k in dests}
        parser.set_defaults(**values)
        for action in parser._actions:
            if action.dest in values:
                action.required = False

    return parser.parse_args(argv)
