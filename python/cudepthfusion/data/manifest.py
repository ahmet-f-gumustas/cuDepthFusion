"""Dataset manifest (manifest.json) and validation report (validation.json) helpers.

Paths stored in a manifest are relative to the manifest's directory so a dataset folder
can be moved as a whole.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cudepthfusion.data.download import sha256_file

MANIFEST_NAME = "manifest.json"
VALIDATION_NAME = "validation.json"
MANIFEST_SCHEMA = 1
ADAPTER_VERSION = "icl-nuim-tum-png/1"

REQUIRED_FIELDS = (
    "schema",
    "dataset",
    "sequence",
    "split",
    "license",
    "files",
    "conventions",
    "frames",
    "timestamp_rule",
    "command",
    "adapter_version",
)


class ManifestError(ValueError):
    """A manifest is missing, unreadable or lacks required fields."""


def file_sha256(path: Path) -> str:
    return sha256_file(path)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write via a temporary file so a crash never leaves half a JSON document."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ManifestError(f"cannot read manifest {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ManifestError(f"{path} is not valid JSON: {error}") from error
    missing = [key for key in REQUIRED_FIELDS if key not in manifest]
    if missing:
        raise ManifestError(f"{path} lacks required field(s) {missing}")
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise ManifestError(f"{path}: unsupported schema {manifest['schema']!r}")
    return manifest
