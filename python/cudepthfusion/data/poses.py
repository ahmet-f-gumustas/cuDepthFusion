"""TUM-format trajectory parsing: ``key tx ty tz qx qy qz qw`` -> 4x4 float64 transforms.

The first column is returned verbatim as a string key. Whether it is a timestamp in
seconds (TUM RGB-D) or an integer frame id (ICL-NUIM) is the adapter's decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

QUATERNION_MIN_NORM = 1e-6
TUM_FIELDS = 8


class TrajectoryFormatError(ValueError):
    """A trajectory line is malformed or holds a degenerate rotation."""


@dataclass(frozen=True, eq=False)
class PoseRecord:
    key: str
    line_number: int
    transform: np.ndarray  # float64 (4, 4), direction as stored in the file


def quaternion_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Rotation matrix of a unit quaternion; the input is normalised first."""
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(norm) or norm < QUATERNION_MIN_NORM:
        raise TrajectoryFormatError(f"quaternion norm {norm} is not usable")
    x, y, z, w = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_matrix(translation: tuple[float, float, float], quaternion_xyzw: tuple) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_rotation(*quaternion_xyzw)
    transform[:3, 3] = translation
    return transform


def parse_tum_trajectory(path: Path) -> list[PoseRecord]:
    """Parse every non-comment line; raises TrajectoryFormatError naming the line."""
    records: list[PoseRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            records.append(_parse_line(path, line_number, text))
    return records


def _parse_line(path: Path, line_number: int, text: str) -> PoseRecord:
    fields = text.split()
    where = f"{path}:{line_number}"
    if len(fields) != TUM_FIELDS:
        raise TrajectoryFormatError(f"{where}: expected {TUM_FIELDS} fields, got {len(fields)}")
    try:
        values = [float(value) for value in fields[1:]]
    except ValueError as error:
        raise TrajectoryFormatError(f"{where}: non-numeric value ({error})") from error
    if not all(math.isfinite(value) for value in values):
        raise TrajectoryFormatError(f"{where}: non-finite value")
    try:
        transform = pose_matrix(tuple(values[:3]), tuple(values[3:]))
    except TrajectoryFormatError as error:
        raise TrajectoryFormatError(f"{where}: {error}") from error
    return PoseRecord(key=fields[0], line_number=line_number, transform=transform)


def invert_rigid(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ transform[:3, 3]
    return inverse
