from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from cudepthfusion import cli
from cudepthfusion.data.camera import PinholeCamera
from cudepthfusion.data.manifest import VALIDATION_NAME, file_sha256, write_json
from cudepthfusion.eval import (
    BASELINE_NAMES,
    FrameMetrics,
    aggregate,
    evaluate_sequence,
    evaluation_mask,
    sanitize,
)
from cudepthfusion.eval.artifacts import CONFIG_NAME, ENVIRONMENT_NAME, PER_FRAME_NAME, SUMMARY_NAME
from cudepthfusion.synthetic import SyntheticSequence
from cudepthfusion.synthetic.export import export_sequence

SMALL = PinholeCamera.vga().scaled(0.25)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A small exported synthetic sequence, marked validated (validate-data has its own tests)."""
    sequence = SyntheticSequence("room_handheld", camera=SMALL, frames=8, seed=2)
    manifest = export_sequence(sequence, tmp_path / "data", command="test")
    write_json(
        manifest.parent / VALIDATION_NAME,
        {"passed": True, "reasons": [], "manifest_sha256": file_sha256(manifest)},
    )
    return manifest


def aggregates(dataset: Path, frames: int = 6) -> dict:
    result = evaluate_sequence(dataset, cli.load_config(None), frames=frames)
    return {
        name: aggregate([m for m in result.metrics if m.baseline == name])
        for name in BASELINE_NAMES
    }


def test_every_baseline_scores_every_frame(dataset: Path) -> None:
    result = evaluate_sequence(dataset, cli.load_config(None), frames=6)
    assert result.frames_evaluated == 6
    assert len(result.metrics) == 6 * len(BASELINE_NAMES)
    assert set(result.baselines) == set(BASELINE_NAMES)
    assert result.manifest_sha256 == file_sha256(dataset)
    for metric in result.metrics:
        assert metric.num_edge + metric.num_interior == metric.num_scored
        assert np.isfinite(metric.rmse_m)


def test_fusion_beats_the_raw_input_and_the_spatial_filter_alone(dataset: Path) -> None:
    scores = aggregates(dataset)
    assert scores["B1"]["rmse_m"] < scores["B0"]["rmse_m"]
    assert scores["B4"]["rmse_m"] < scores["B0"]["rmse_m"]
    assert scores["B4"]["rmse_m"] <= scores["B1"]["rmse_m"]


def test_raw_baseline_covers_the_whole_mask(dataset: Path) -> None:
    scores = aggregates(dataset)
    assert scores["B0"]["coverage"] == pytest.approx(1.0)
    for name in BASELINE_NAMES:
        assert scores[name]["coverage"] > 0.95  # nothing may quietly drop pixels


def test_evaluation_mask_uses_only_pixels_both_sources_have() -> None:
    clean = np.array([[1.0, 2.0], [0.0, 3.0]])
    raw = np.array([[1.0, 0.0], [2.0, 3.5]], dtype=np.float32)
    raw_valid = sanitize(raw, 0.2, 8.0)
    mask = evaluation_mask(clean, raw_valid)
    np.testing.assert_array_equal(mask, [[True, False], [False, True]])


def test_aggregate_pools_sums_not_averages() -> None:
    def record(scored: int, squared: float) -> FrameMetrics:
        return FrameMetrics(
            sequence="s",
            baseline="B4",
            frame_id=0,
            num_mask=scored,
            num_scored=scored,
            num_missing=0,
            sum_squared_error=squared,
            sum_abs_error=0.0,
            sum_error=0.0,
            num_bad=0,
            num_edge=0,
            edge_sum_squared=0.0,
            edge_sum_abs=0.0,
            num_interior=scored,
            interior_sum_squared=squared,
            interior_sum_abs=0.0,
            median_abs_m=0.0,
            p90_abs_m=0.0,
            num_hole_filled=0,
            hole_sum_squared=0.0,
        )

    pooled = aggregate([record(100, 1.0), record(900, 9.0)])
    assert pooled["rmse_m"] == pytest.approx(np.sqrt(10.0 / 1000.0))
    assert pooled["pixels_scored"] == 1000
    assert pooled["hole_rmse_m"] is None


def test_evaluate_cli_writes_a_run_folder(dataset: Path, tmp_path: Path) -> None:
    output = tmp_path / "run"
    code = cli.main(
        [
            "evaluate",
            "--manifest",
            str(dataset),
            "--output",
            str(output),
            "--frames",
            "3",
            "--baselines",
            "B0",
            "B4",
        ]
    )
    assert code == cli.EXIT_OK
    for name in (SUMMARY_NAME, PER_FRAME_NAME, CONFIG_NAME, ENVIRONMENT_NAME):
        assert (output / name).exists(), name

    def reject(token: str) -> None:
        raise ValueError(token)

    summary = json.loads((output / SUMMARY_NAME).read_text(), parse_constant=reject)
    assert summary["sequences"][0]["manifest_sha256"] == file_sha256(dataset)
    assert summary["thresholds"]["bad_pixel_abs_m"] == 0.02
    assert set(summary["overall"]["pooled"]) == {"B0", "B4"}
    assert "git_commit" in summary
    with (output / PER_FRAME_NAME).open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3 * 2
    assert {row["baseline"] for row in rows} == {"B0", "B4"}


def test_evaluate_cli_says_what_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        [
            "evaluate",
            "--split",
            "test",
            "--data-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "run"),
        ]
    )
    assert code == cli.EXIT_USAGE_ERROR
    assert "no downloaded sequence for split 'test'" in capsys.readouterr().err


def test_stability_tracks_fixed_world_points(dataset: Path) -> None:
    result = evaluate_sequence(
        dataset, cli.load_config(None), frames=8, baselines=("B0", "B4"), track_stability=True
    )
    stability = {entry.baseline: entry for entry in result.stability}
    assert set(stability) == {"B0", "B4"}
    for entry in stability.values():
        assert entry.tracked_points > 50
        assert entry.observations > entry.tracked_points
        assert np.isfinite(entry.median_point_std_m)
    # Fusion is supposed to make a fixed world point wobble less than the raw input does.
    assert stability["B4"].median_point_std_m < stability["B0"].median_point_std_m


def test_ablation_cli_runs_every_variant(dataset: Path, tmp_path: Path) -> None:
    output = tmp_path / "ablation"
    code = cli.main(
        ["ablate", "--manifest", str(dataset), "--output", str(output), "--frames", "4"]
    )
    assert code == cli.EXIT_OK
    summary = json.loads((output / SUMMARY_NAME).read_text())
    assert summary["kind"] == "ablation"
    assert set(summary["overall"]["pooled"]) == set(cli.ABLATIONS)
    assert summary["overall"]["pooled"]["full"]["frames"] == 4


def test_plots_are_written_when_asked(dataset: Path, tmp_path: Path) -> None:
    output = tmp_path / "run"
    code = cli.main(
        [
            "evaluate",
            "--manifest",
            str(dataset),
            "--output",
            str(output),
            "--frames",
            "3",
            "--baselines",
            "B0",
            "B4",
            "--plots",
        ]
    )
    assert code == cli.EXIT_OK
    for name in ("quality.png", "edge_vs_interior.png", "per_frame_rmse.png"):
        assert (output / name).stat().st_size > 1000, name
