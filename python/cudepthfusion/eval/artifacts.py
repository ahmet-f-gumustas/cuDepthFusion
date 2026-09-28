"""Run artifacts: everything needed to reproduce and audit a result (spec 8, 11)."""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cudepthfusion import _core
from cudepthfusion.config import LoadedConfig

SUMMARY_NAME = "summary.json"
PER_FRAME_NAME = "per_frame.csv"
CONFIG_NAME = "config_resolved.yaml"
ENVIRONMENT_NAME = "environment.json"
JETSON_RELEASE_FILE = Path("/etc/nv_tegra_release")


def environment_info() -> dict[str, Any]:
    jetson = (
        JETSON_RELEASE_FILE.read_text(encoding="utf-8").strip()
        if JETSON_RELEASE_FILE.exists()
        else None
    )
    return {
        "cudepthfusion": _core.build_info(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "jetson_l4t_release": jetson,
    }


def git_commit() -> str | None:
    """The commit the code came from, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() or None


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def finite(value: Any) -> Any:
    """JSON has no Infinity or NaN: a number that does not exist is written as null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [finite(item) for item in value]
    return value


def write_run(
    output: Path, summary: dict[str, Any], rows: list[dict[str, Any]], config: LoadedConfig
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / SUMMARY_NAME).write_text(
        json.dumps(finite(summary), indent=2) + "\n", encoding="utf-8"
    )
    (output / CONFIG_NAME).write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    (output / ENVIRONMENT_NAME).write_text(
        json.dumps(environment_info(), indent=2) + "\n", encoding="utf-8"
    )
    if rows:
        with (output / PER_FRAME_NAME).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
