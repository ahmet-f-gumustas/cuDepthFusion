from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cudepthfusion.data.camera import PinholeCamera
from cudepthfusion.data.icl_nuim import IclGroundTruth, IclInputSequence
from cudepthfusion.data.manifest import load_manifest
from cudepthfusion.data.validation import validate_sequence
from cudepthfusion.synthetic import SyntheticSequence
from cudepthfusion.synthetic.export import ORACLE_NAME, export_sequence

SMALL = PinholeCamera.vga().scaled(0.25)
REPO_ROOT = Path(__file__).resolve().parents[2]
QUANTISATION_M = 0.5 / 5000.0


def export(root: Path, name: str, frames: int = 8, seed: int = 0) -> tuple[Path, SyntheticSequence]:
    sequence = SyntheticSequence(name, camera=SMALL, frames=frames, seed=seed)
    return export_sequence(sequence, root / "out", command="test"), sequence


def test_layout_and_manifest(tmp_path: Path) -> None:
    manifest_path, _ = export(tmp_path, "front_surface_pass")
    root = manifest_path.parent
    manifest = load_manifest(manifest_path)

    assert len(list((root / "clean" / "depth").glob("*.png"))) == 8
    assert len(list((root / "noisy" / "depth").glob("*.png"))) == 8
    assert manifest["dataset"] == "synthetic"
    assert manifest["frames"]["matched_ids"] == list(range(8))
    assert manifest["frames"]["pairing"]["ok"] is True
    assert manifest["conventions"]["fy"] == SMALL.fy
    assert manifest["generator"]["dynamic_surface_ids"] == [2]
    assert manifest["unstorable_pixels"] == 0
    oracle = np.load(root / ORACLE_NAME)
    assert oracle["surface_id"].shape == (8, SMALL.height, SMALL.width)
    assert oracle["dynamic_mask"].dtype == bool


def test_exported_input_matches_the_generator(tmp_path: Path) -> None:
    manifest_path, sequence = export(tmp_path, "room_handheld")
    exported = IclInputSequence(manifest_path, require_validated=False)
    assert exported.frame_ids == tuple(sequence.frame_ids)
    for frame_id in (0, 5):
        mine, theirs = exported.frame(frame_id), sequence.frame(frame_id)
        np.testing.assert_allclose(mine.depth_m, theirs.input.depth_m, atol=QUANTISATION_M)
        np.testing.assert_array_equal(mine.depth_m == 0, theirs.input.depth_m == 0)
        np.testing.assert_allclose(mine.T_world_camera, theirs.input.T_world_camera, atol=1e-12)
        assert mine.intrinsics == theirs.input.intrinsics
        assert mine.timestamp_s == pytest.approx(theirs.input.timestamp_s)
        clean = IclGroundTruth(manifest_path).clean_depth_m(frame_id)
        np.testing.assert_allclose(clean, theirs.truth.depth_m, atol=QUANTISATION_M)


def test_export_is_deterministic(tmp_path: Path) -> None:
    first = load_manifest(export(tmp_path / "a", "step")[0])["files"]
    second = load_manifest(export(tmp_path / "b", "step")[0])["files"]
    assert {k: v["sha256"] for k, v in first.items()} == {k: v["sha256"] for k, v in second.items()}


def test_reexport_replaces_a_previous_export(tmp_path: Path) -> None:
    export(tmp_path, "step", seed=1)
    manifest_path, _ = export(tmp_path, "step", seed=2)
    assert load_manifest(manifest_path)["generator"]["seed"] == 2


def test_refuses_to_replace_a_foreign_directory(tmp_path: Path) -> None:
    foreign = tmp_path / "out" / "step"
    foreign.mkdir(parents=True)
    (foreign / "notes.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not a synthetic export"):
        export(tmp_path, "step")
    assert (foreign / "notes.txt").read_text(encoding="utf-8") == "keep me"


def test_validate_data_confirms_a_moving_camera_export(tmp_path: Path) -> None:
    manifest_path, _ = export(tmp_path, "room_handheld", frames=48)
    report = validate_sequence(manifest_path)
    assert report["passed"], report["reasons"]
    best = report["geometry"]["best"]["candidate"]
    assert (best["fy_sign"], best["pose_id_offset"], best["depth_kind"]) == (1, 0, "z")


def test_static_camera_export_cannot_identify_pose_conventions(tmp_path: Path) -> None:
    # With identical poses every pose offset and direction explains the data equally well,
    # so validate-data must refuse rather than pretend the conventions were confirmed.
    manifest_path, _ = export(tmp_path, "plane_static", frames=48)
    report = validate_sequence(manifest_path)
    assert not report["passed"]
    assert any("not identifiable" in reason for reason in report["reasons"])


def test_make_synthetic_script(tmp_path: Path) -> None:
    script = str(REPO_ROOT / "scripts" / "make_synthetic.py")
    common = ["--frames", "2", "--scale", "0.25", "--output", str(tmp_path)]
    ok = subprocess.run(
        [sys.executable, script, "--suite", "step", "plane_static", *common],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0, ok.stderr
    assert (tmp_path / "step" / "manifest.json").exists()
    assert (tmp_path / "plane_static" / "manifest.json").exists()

    bad = subprocess.run(
        [sys.executable, script, "--suite", "nope", *common],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode == 2 and "unknown scenario" in bad.stderr
