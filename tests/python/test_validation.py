from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fake_icl import FY_ABS, build_fake_icl

from cudepthfusion import cli
from cudepthfusion.data.fetch import fetch_sequence
from cudepthfusion.data.icl_nuim import IclInputSequence
from cudepthfusion.data.manifest import MANIFEST_NAME, VALIDATION_NAME
from cudepthfusion.data.registry import VARIANTS
from cudepthfusion.data.validation import validate_sequence


def fetch(fake, root: Path) -> Path:
    return fetch_sequence(
        fake.spec,
        VARIANTS,
        root / "data",
        command="test",
        downloader=fake.downloader,
        log=lambda _: None,
    )


def test_true_conventions_pass_and_unlock_filter_input(tmp_path: Path) -> None:
    manifest_path = fetch(build_fake_icl(tmp_path), tmp_path)
    report = validate_sequence(manifest_path)

    assert report["passed"], report["reasons"]
    assert report["geometry"]["best"]["candidate"]["pose_id_offset"] == -1
    assert all(r["declared_win_rate"] >= 0.9 for r in report["geometry"]["rivals"])
    assert (manifest_path.parent / VALIDATION_NAME).exists()
    assert len(IclInputSequence(manifest_path)) == 48


@pytest.mark.parametrize(
    ("truth", "declared", "field", "expected"),
    [
        ({"pose_offset": -1}, {"pose_id_offset": 0}, "pose_id_offset", -1),
        ({"depth_kind": "ray_distance"}, {"depth_kind": "z"}, "depth_kind", "ray_distance"),
        ({"fy_sign": 1}, {"fy": -FY_ABS}, "fy_sign", 1),
        ({}, {"pose_direction": "T_camera_world"}, "pose_direction", "T_world_camera"),
        ({}, {"depth_units_per_meter": 1000.0}, "depth_units_per_meter", 5000.0),
    ],
    ids=["offset", "ray-distance", "fy-sign", "pose-direction", "scale"],
)
def test_wrong_declared_convention_is_caught(
    tmp_path: Path, truth: dict, declared: dict, field: str, expected: object
) -> None:
    manifest_path = fetch(build_fake_icl(tmp_path, declared=declared, **truth), tmp_path)
    report = validate_sequence(manifest_path)
    assert not report["passed"]
    assert any("differs from the declared" in reason for reason in report["reasons"])
    assert report["geometry"]["best"]["candidate"][field] == expected


def test_swapped_noisy_frames_fail_consistency(tmp_path: Path) -> None:
    manifest_path = fetch(build_fake_icl(tmp_path, swap_noisy=(3, 30)), tmp_path)
    report = validate_sequence(manifest_path)
    assert not report["passed"]
    assert any("differ from clean" in reason for reason in report["reasons"])


def test_duplicate_frames_fail_validation(tmp_path: Path) -> None:
    manifest_path = fetch(build_fake_icl(tmp_path, duplicate_noisy=7), tmp_path)
    report = validate_sequence(manifest_path)
    assert not report["passed"]
    assert any("duplicate" in reason for reason in report["reasons"])


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON constant {token}")


def test_cli_output_is_strict_json_even_with_unusable_frames(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = fetch(build_fake_icl(tmp_path), tmp_path)
    # A noisy frame with no valid pixel has no noisy/clean median (infinite internally).
    blank = manifest_path.parent / "noisy" / "depth" / "4.png"
    cv2.imwrite(str(blank), np.zeros(cv2.imread(str(blank), cv2.IMREAD_UNCHANGED).shape, np.uint16))

    assert cli.main(["validate-data", "--manifest", str(manifest_path)]) == cli.EXIT_FAILED_CHECK
    summary = json.loads(capsys.readouterr().out, parse_constant=_reject_constant)
    assert summary["passed"] is False
    report = (manifest_path.parent / VALIDATION_NAME).read_text(encoding="utf-8")
    json.loads(report, parse_constant=_reject_constant)


def test_declared_offset_outside_default_candidates_is_still_scored(tmp_path: Path) -> None:
    fake = build_fake_icl(tmp_path, pose_offset=-1, declared={"pose_id_offset": 3})
    report = validate_sequence(fetch(fake, tmp_path))
    assert not report["passed"]
    assert report["geometry"]["declared"]["candidate"]["pose_id_offset"] == 3


def test_validate_data_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = fetch(build_fake_icl(tmp_path), tmp_path)
    assert cli.main(["validate-data", "--manifest", str(manifest_path)]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["passed"] is True

    missing = tmp_path / "nope" / MANIFEST_NAME
    assert cli.main(["validate-data", "--manifest", str(missing)]) == cli.EXIT_USAGE_ERROR
    assert "cannot read manifest" in capsys.readouterr().err
