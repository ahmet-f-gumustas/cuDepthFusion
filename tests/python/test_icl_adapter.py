from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fake_icl import FY_ABS, build_fake_icl, camera_pose

from cudepthfusion.data.fetch import fetch_sequence
from cudepthfusion.data.icl_nuim import (
    DatasetError,
    IclGroundTruth,
    IclInputSequence,
    pose_table,
)
from cudepthfusion.data.manifest import (
    MANIFEST_NAME,
    VALIDATION_NAME,
    file_sha256,
    load_manifest,
    write_json,
)
from cudepthfusion.data.poses import parse_tum_trajectory
from cudepthfusion.data.registry import VARIANTS

REPO_ROOT = Path(__file__).resolve().parents[2]


def fetch(fake, root: Path, log=lambda _: None) -> Path:
    return fetch_sequence(
        fake.spec, VARIANTS, root / "data", command="test", downloader=fake.downloader, log=log
    )


def mark_validation(manifest_path: Path, *, passed: bool) -> None:
    """Stand-in for a validate-data run, to test the adapter's gating on its own."""
    report = {"passed": passed, "reasons": [], "manifest_sha256": file_sha256(manifest_path)}
    write_json(manifest_path.parent / VALIDATION_NAME, report)


def test_manifest_records_sources_hashes_and_conventions(tmp_path: Path) -> None:
    manifest_path = fetch(build_fake_icl(tmp_path), tmp_path)
    manifest = load_manifest(manifest_path)

    assert manifest_path.name == MANIFEST_NAME
    assert manifest["files"]["clean"]["top_level_entries"] == ["depth"]
    assert manifest["files"]["noisy"]["publisher_sha256"] is None
    assert len(manifest["files"]["poses"]["sha256"]) == 64
    assert manifest["conventions"]["fy"] == -FY_ABS
    assert manifest["conventions_status"].startswith("declared")
    assert manifest["frames"]["matched_ids"] == list(range(48))
    assert manifest["license"]["name"]
    assert "frame id, not seconds" in manifest["timestamp_rule"]


def test_fetch_reuses_an_unchanged_extraction(tmp_path: Path) -> None:
    fake = build_fake_icl(tmp_path)
    fetch(fake, tmp_path)
    messages: list[str] = []
    fetch(fake, tmp_path, log=messages.append)
    assert sum("reusing extracted" in m for m in messages) == 2


def test_fetch_requires_all_variants(tmp_path: Path) -> None:
    fake = build_fake_icl(tmp_path)
    with pytest.raises(ValueError, match="missing"):
        fetch_sequence(fake.spec, ["clean"], tmp_path, command="t", downloader=fake.downloader)


def test_pairing_reports_missing_and_duplicate_frames(tmp_path: Path) -> None:
    fake = build_fake_icl(tmp_path, missing_noisy=(5,), duplicate_noisy=7)
    pairing = load_manifest(fetch(fake, tmp_path))["frames"]["pairing"]

    assert pairing["ok"] is False
    assert pairing["duplicates"]["noisy"] == [7]
    assert 5 in pairing["only_in"]["clean"] and 7 in pairing["only_in"]["clean"]
    assert pairing["matched_count"] == 46


def test_pose_keys_must_be_integer_frame_ids(tmp_path: Path) -> None:
    path = tmp_path / "traj.txt"
    path.write_text("1305031102.1753 0 0 0 0 0 0 1\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="not an integer frame id"):
        pose_table(parse_tum_trajectory(path), 0)


def test_filter_input_is_refused_until_validated(tmp_path: Path) -> None:
    manifest_path = fetch(build_fake_icl(tmp_path), tmp_path)
    with pytest.raises(DatasetError, match="has not been validated"):
        IclInputSequence(manifest_path)
    mark_validation(manifest_path, passed=False)
    with pytest.raises(DatasetError, match="failed validation"):
        IclInputSequence(manifest_path)


def test_validated_input_frames_follow_the_contract(tmp_path: Path) -> None:
    manifest_path = fetch(build_fake_icl(tmp_path), tmp_path)
    mark_validation(manifest_path, passed=True)

    sequence = IclInputSequence(manifest_path)
    frame = sequence.frame(10)
    assert len(sequence) == 48
    assert frame.depth_m.dtype == np.float32 and frame.depth_m.flags.c_contiguous
    assert frame.intrinsics.fy == -FY_ABS
    assert frame.timestamp_s == pytest.approx(10 / 30.0)
    # The fake stores pose id = image id + 1; offset -1 maps it back to image 10.
    np.testing.assert_allclose(frame.T_world_camera, camera_pose(10), atol=1e-9)
    assert not any("clean" in name for name in vars(frame))
    clean = IclGroundTruth(manifest_path).clean_depth_m(10)
    assert np.median(np.abs(frame.depth_m - clean)[frame.depth_m > 0]) < 0.01


def test_changed_manifest_requires_revalidation(tmp_path: Path) -> None:
    manifest_path = fetch(build_fake_icl(tmp_path), tmp_path)
    mark_validation(manifest_path, passed=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split"] = "test"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetError, match="different manifest.json"):
        IclInputSequence(manifest_path)


def test_download_script_rejects_unknown_sequence(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "download_dataset.py"),
            "--dataset",
            "icl-nuim",
            "--sequence",
            "kt9",
            "--output",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unknown dataset/sequence" in result.stderr
