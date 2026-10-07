"""Loading config.yaml, the record of the experiment's parameters (spec v0.3, 2.6, 7.5).

The modules keep their constants as defaults; config.yaml repeats them in one place, and
tests/test_config.py checks the two agree. Pipeline scripts load this file and pass values in.
"""

from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
SECTIONS = ("data", "retrieval", "signals", "fuzzy", "routing", "actions", "calibration",
            "splits", "evaluation", "fitted")


def load_config(path=CONFIG):
    """config.yaml as a dict. Raises if a section is missing."""
    with open(path, encoding="utf8") as f:
        cfg = yaml.safe_load(f)
    missing = [s for s in SECTIONS if s not in (cfg or {})]
    if missing:
        raise ValueError(f"{path}: missing section(s) {missing}")
    return cfg
