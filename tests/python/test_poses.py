from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from cudepthfusion.data import poses


def test_identity_and_known_rotation() -> None:
    np.testing.assert_allclose(poses.quaternion_to_rotation(0, 0, 0, 1), np.eye(3))
    half = math.sqrt(0.5)
    rz90 = poses.quaternion_to_rotation(0, 0, half, half)
    np.testing.assert_allclose(rz90 @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)


def test_quaternion_is_normalised() -> None:
    np.testing.assert_allclose(poses.quaternion_to_rotation(0, 0, 0, 5.0), np.eye(3))


@pytest.mark.parametrize("q", [(0, 0, 0, 0), (1e-9, 0, 0, 0), (math.nan, 0, 0, 1)])
def test_degenerate_quaternion_is_rejected(q: tuple) -> None:
    with pytest.raises(poses.TrajectoryFormatError):
        poses.quaternion_to_rotation(*q)


def test_parse_tum_trajectory_keeps_keys_verbatim(tmp_path: Path) -> None:
    path = tmp_path / "traj.txt"
    path.write_text(
        "# comment\n1 0 0 -2.25 0 0 0 1\n\n1305031102.175304 1 2 3 0 0 0 2\n", encoding="utf-8"
    )
    records = poses.parse_tum_trajectory(path)
    assert [r.key for r in records] == ["1", "1305031102.175304"]
    assert [r.line_number for r in records] == [2, 4]
    np.testing.assert_allclose(records[0].transform[:3, 3], [0, 0, -2.25])
    np.testing.assert_allclose(records[1].transform[:3, :3], np.eye(3))


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("1 0 0 0 0 0 1", "expected 8 fields"),
        ("1 0 0 x 0 0 0 1", "non-numeric"),
        ("1 0 0 inf 0 0 0 1", "non-finite"),
        ("1 0 0 0 0 0 0 0", "quaternion norm"),
    ],
)
def test_malformed_lines_name_the_line(tmp_path: Path, line: str, message: str) -> None:
    path = tmp_path / "traj.txt"
    path.write_text(f"1 0 0 0 0 0 0 1\n{line}\n", encoding="utf-8")
    with pytest.raises(poses.TrajectoryFormatError, match=rf"traj.txt:2: .*{message}"):
        poses.parse_tum_trajectory(path)


@pytest.mark.parametrize(
    "quaternion",
    [
        (0, 0, 0, 1),
        (0.1, 0.2, 0.3, 0.9),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0.5, -0.5, 0.5, -0.5),
    ],
    ids=["identity", "generic", "x180", "y180", "z180", "negative-w"],
)
def test_rotation_to_quaternion_round_trips(quaternion: tuple) -> None:
    rotation = poses.quaternion_to_rotation(*quaternion)
    recovered = poses.rotation_to_quaternion(rotation)
    assert recovered[3] >= 0
    np.testing.assert_allclose(poses.quaternion_to_rotation(*recovered), rotation, atol=1e-12)


def test_invert_rigid_round_trips() -> None:
    transform = poses.pose_matrix((1.0, -2.0, 0.5), (0.1, 0.2, 0.3, 0.9))
    np.testing.assert_allclose(poses.invert_rigid(transform) @ transform, np.eye(4), atol=1e-12)
