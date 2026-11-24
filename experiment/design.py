"""Shared circuit design and paths for acquisition and offline analysis."""

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Each sweep specifies a block angle in radians and the sampled block counts.
SWEEPS = {
    "small": (math.pi / 16, (0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16)),
    "large": (math.pi / 4, range(5)),
    "reset": (math.pi / 4, range(5)),
}
INPUTS = ("ground", "excited", "plus")
BASES = ("X", "Y", "Z")
SEED = 550


def run_path(name):
    """Resolve a local run name to a path under the experiment's runs directory.

    The name starts with a letter or digit and can also contain hyphens and
    underscores. This function resolves the path without creating it.
    """
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name):
        raise ValueError("Use letters, digits, hyphens or underscores for the run name.")
    return ROOT / "runs" / name


def read_json(path):
    """Read a UTF-8 JSON file from the supplied Path and return its parsed value."""
    return json.loads(path.read_text(encoding="utf-8"))
