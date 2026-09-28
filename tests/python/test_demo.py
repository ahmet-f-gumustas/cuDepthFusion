from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cudepthfusion import cli
from cudepthfusion.data.camera import PinholeCamera
from cudepthfusion.data.manifest import VALIDATION_NAME, file_sha256, write_json
from cudepthfusion.eval.artifacts import LATENCY_NAME, PER_FRAME_NAME, SUMMARY_NAME
from cudepthfusion.synthetic import SyntheticSequence
from cudepthfusion.synthetic.export import export_sequence
from cudepthfusion.viz import demo as demo_module
from cudepthfusion.viz import render
from cudepthfusion.viz.demo import DemoError, DemoOptions, run_demo, write_demo_run

SMALL = PinholeCamera.vga().scaled(0.25)
SCALES = render.Scales(depth_min_m=1.0, depth_max_m=3.0, error_max_mm=100.0)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    sequence = SyntheticSequence("room_handheld", camera=SMALL, frames=6, seed=1)
    manifest = export_sequence(sequence, tmp_path / "data", command="test")
    write_json(
        manifest.parent / VALIDATION_NAME,
        {"passed": True, "reasons": [], "manifest_sha256": file_sha256(manifest)},
    )
    return manifest


def test_the_same_depth_gets_the_same_colour_in_every_panel_and_frame() -> None:
    first = np.full((4, 4), 2.0, dtype=np.float32)
    second = first.copy()
    second[0, 0] = 2.9  # a different frame with a different content range
    valid = np.ones((4, 4), dtype=bool)
    a = render.depth_panel("a", first, valid, SCALES).image
    b = render.depth_panel("b", second, valid, SCALES).image
    # Fixed scales: 2.0 m looks identical although the two frames differ.
    assert np.array_equal(a[1:, 1:], b[1:, 1:])
    assert not np.array_equal(a[0, 0], b[0, 0])


def test_invalid_pixels_are_black_everywhere() -> None:
    depth = np.full((3, 3), 2.0, dtype=np.float32)
    valid = np.ones((3, 3), dtype=bool)
    valid[1, 1] = False
    assert tuple(render.depth_panel("d", depth, valid, SCALES).image[1, 1]) == render.INVALID_COLOR
    error = render.error_panel("e", depth - 2.0, valid, SCALES).image
    assert tuple(error[1, 1]) == render.INVALID_COLOR
    source = render.source_panel("s", np.zeros((3, 3), dtype=np.uint8)).image
    assert tuple(source[0, 0]) == render.INVALID_COLOR


def test_the_depth_range_covers_the_whole_run_not_only_one_frame() -> None:
    near = np.full(1000, 1.0, dtype=np.float32)
    far = np.full(1000, 5.0, dtype=np.float32)
    low, high = render.auto_depth_range([near, far])
    assert low <= 1.0 and high >= 5.0
    assert render.auto_depth_range([]) == (0.0, 1.0)
    flat = render.auto_depth_range([np.full(10, 2.0, dtype=np.float32)])
    assert flat[1] > flat[0]


def test_compose_stacks_the_panels_the_colorbars_and_the_legend() -> None:
    panels = [
        render.depth_panel(f"p{i}", np.full((6, 8), 2.0, np.float32), np.ones((6, 8), bool), SCALES)
        for i in range(5)
    ]
    image = render.compose(panels, SCALES, "status", columns=3, panel_width=60)
    # Two rows of three panels, each with a title bar, plus the footer and the legend strip.
    panel_height = int(round(6 * 60 / 8))
    expected = 2 * (panel_height + render.TITLE_HEIGHT) + render.FOOTER_HEIGHT + 22
    assert image.shape == (expected, 3 * 60, 3)


def test_a_missing_panel_says_why_instead_of_showing_zeros() -> None:
    panel = render.missing_panel("clean ground truth", (20, 40), "no clean variant")
    assert panel.image.shape == (20, 40, 3)
    assert panel.image.any()  # not a black rectangle that reads as valid data


def test_a_sequence_without_clean_depth_reports_it_instead_of_faking_it(dataset: Path) -> None:
    stripped = json.loads(dataset.read_text())
    stripped["files"].pop("clean")
    manifest = dataset.parent / "no_clean.json"
    write_json(manifest, stripped)
    truth, reason = demo_module._ground_truth(manifest)
    assert truth is None
    assert "clean" in reason


def test_the_demo_runs_and_writes_every_artifact(dataset: Path, tmp_path: Path) -> None:
    config = cli.load_config(None)
    summary, rows = run_demo(dataset, config, options=DemoOptions(frames=4))
    output = tmp_path / "run"
    write_demo_run(output, summary, rows, config)

    assert summary["frames_rendered"] == 4
    assert len(rows) == 4
    written = json.loads((output / SUMMARY_NAME).read_text())
    assert written["manifest_sha256"] == file_sha256(dataset)
    assert written["scales"]["depth_max_m"] > written["scales"]["depth_min_m"]

    per_frame = list(csv.DictReader((output / PER_FRAME_NAME).open()))
    latency = list(csv.DictReader((output / LATENCY_NAME).open()))
    assert len(per_frame) == len(latency) == 4
    assert "rmse_b4_m" in per_frame[0] and "process_b4_ms" in latency[0]


def test_the_three_times_stay_separate_and_an_unmeasured_one_is_null(dataset: Path) -> None:
    summary, _ = run_demo(dataset, cli.load_config(None), options=DemoOptions(frames=3))
    latency = summary["latency"]
    # Demo throughput includes decoding, rendering and writing, so it is never the engine time.
    assert latency["demo_throughput"]["median_ms"] > latency["process_b4"]["median_ms"]
    assert latency["gpu_compute"]["median_ms"] is None
    assert "P7" in latency["gpu_compute"]["reason"]


def test_quality_is_measured_against_clean_depth_the_filters_never_saw(dataset: Path) -> None:
    summary, rows = run_demo(dataset, cli.load_config(None), options=DemoOptions(frames=4))
    assert set(summary["quality"]) == {"B1", "B4"}
    assert summary["quality"]["B4"]["rmse_m"] < summary["quality"]["B1"]["rmse_m"]
    assert all(row["rmse_b4_m"] is not None for row in rows)


def test_an_empty_run_is_an_error_not_an_empty_video(dataset: Path) -> None:
    with pytest.raises(DemoError, match="no frames"):
        run_demo(dataset, cli.load_config(None), options=DemoOptions(frames=0))


def test_interactive_mode_explains_itself_on_a_headless_build(dataset: Path, monkeypatch) -> None:
    def no_gui(*_args, **_kwargs):
        raise cv2.error("The function is not implemented")

    monkeypatch.setattr(cv2, "imshow", no_gui)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    with pytest.raises(DemoError, match="headless"):
        run_demo(dataset, cli.load_config(None), options=DemoOptions(frames=2, interactive=True))


def test_the_video_is_written_when_the_codec_exists(dataset: Path, tmp_path: Path) -> None:
    video = tmp_path / "demo.mp4"
    try:
        summary, _ = run_demo(
            dataset, cli.load_config(None), options=DemoOptions(frames=3, save_video=video)
        )
    except DemoError as error:
        pytest.skip(f"no video codec in this OpenCV build: {error}")
    assert video.exists() and video.stat().st_size > 0
    assert summary["outputs"]["video"] == str(video)


def test_png_frames_are_written_per_frame(dataset: Path, tmp_path: Path) -> None:
    frames_dir = tmp_path / "png"
    run_demo(dataset, cli.load_config(None), options=DemoOptions(frames=3, save_frames=frames_dir))
    written = sorted(frames_dir.glob("frame_*.png"))
    assert len(written) == 3
    assert cv2.imread(str(written[0])) is not None
