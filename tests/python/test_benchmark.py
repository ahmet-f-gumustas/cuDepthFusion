from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

import cudepthfusion as cdf
from cudepthfusion import cli
from cudepthfusion.data.camera import PinholeCamera
from cudepthfusion.data.manifest import VALIDATION_NAME, file_sha256, write_json
from cudepthfusion.eval.artifacts import LATENCY_NAME, SUMMARY_NAME
from cudepthfusion.eval.benchmark import (
    BenchmarkError,
    BenchmarkPlan,
    load_frames,
    run_benchmark,
)
from cudepthfusion.synthetic import SyntheticSequence
from cudepthfusion.synthetic.export import export_sequence

SMALL = PinholeCamera.vga().scaled(0.25)
CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    sequence = SyntheticSequence("room_handheld", camera=SMALL, frames=10, seed=5)
    manifest = export_sequence(sequence, tmp_path / "data", command="test")
    write_json(
        manifest.parent / VALIDATION_NAME,
        {"passed": True, "reasons": [], "manifest_sha256": file_sha256(manifest)},
    )
    return manifest


def test_the_benchmark_workload_is_the_evaluated_one() -> None:
    icl = yaml.safe_load((CONFIGS / "icl.yaml").read_text())
    bench = yaml.safe_load((CONFIGS / "benchmark.yaml").read_text())
    assert {k: v for k, v in bench.items() if k != "benchmark"} == {
        k: v for k, v in icl.items() if k != "benchmark"
    }
    assert bench["benchmark"] == {"warmup_frames": 100, "measured_frames": 500, "repeats": 5}


def test_every_repeat_replays_the_same_slice_and_warmup_is_not_timed(dataset: Path) -> None:
    plan = BenchmarkPlan(warmup_frames=3, measured_frames=4, repeats=2)
    frames = load_frames(dataset, plan)
    summary, rows = run_benchmark(frames, cli.load_config(None), plan, backend="cpu")

    assert len(rows) == plan.measured_frames * plan.repeats
    per_repeat = [[row["frame_id"] for row in rows if row["repeat"] == r] for r in range(2)]
    assert per_repeat[0] == per_repeat[1] == [f.frame_id for f in frames[3:7]]
    assert len(summary["process_latency_repeat_medians_ms"]) == 2
    assert summary["process_latency"]["count"] == 8
    assert summary["io_scope"].startswith("frames decoded into memory")


def test_a_time_that_was_not_measured_is_null_with_a_reason(dataset: Path) -> None:
    plan = BenchmarkPlan(warmup_frames=1, measured_frames=2, repeats=1)
    summary, rows = run_benchmark(
        load_frames(dataset, plan), cli.load_config(None), plan, backend="cpu"
    )
    assert summary["gpu_compute"]["median_ms"] is None
    assert "CPU backend" in summary["gpu_compute"]["reason"]
    assert all(row["compute_ms"] is None for row in rows)


def test_a_slice_longer_than_the_sequence_is_refused(dataset: Path) -> None:
    with pytest.raises(BenchmarkError, match="needs 20"):
        load_frames(dataset, BenchmarkPlan(warmup_frames=10, measured_frames=10, repeats=1))


def test_the_cli_writes_summary_latency_config_and_environment(
    dataset: Path, tmp_path: Path
) -> None:
    output = tmp_path / "perf"
    code = cli.main(
        [
            "benchmark",
            "--manifest",
            str(dataset),
            "--output",
            str(output),
            "--backend",
            "cpu",
            "--warmup-frames",
            "2",
            "--measured-frames",
            "3",
            "--repeats",
            "2",
            "--label",
            "test",
        ]
    )
    assert code == 0
    summary = json.loads((output / SUMMARY_NAME).read_text())
    assert summary["label"] == "test"
    assert summary["source"]["manifest_sha256"] == file_sha256(dataset)
    assert summary["plan"]["repeats"] == 2
    rows = list(csv.DictReader((output / LATENCY_NAME).open()))
    assert len(rows) == 6
    assert (output / "environment.json").exists() and (output / "config_resolved.yaml").exists()


def test_cpu_results_carry_no_device_timings() -> None:
    engine = cdf.DepthFusion(backend="cpu")
    frame = next(iter(SyntheticSequence("room_handheld", camera=SMALL, frames=1, seed=0)))
    result = engine.process(
        depth_m=frame.input.depth_m,
        intrinsics=frame.input.intrinsics,
        T_world_camera=frame.input.T_world_camera,
        timestamp_s=frame.input.timestamp_s,
    )
    assert result.diagnostics.device_timings is None


@pytest.mark.gpu
def test_cuda_results_carry_device_timings_that_add_up() -> None:
    engine = cdf.DepthFusion(backend="cuda")
    for frame in SyntheticSequence("room_handheld", camera=SMALL, frames=3, seed=0):
        result = engine.process(
            depth_m=frame.input.depth_m,
            intrinsics=frame.input.intrinsics,
            T_world_camera=frame.input.T_world_camera,
            timestamp_s=frame.input.timestamp_s,
        )
    timings = result.diagnostics.device_timings
    assert timings is not None
    stages = (
        timings.sanitize_ms
        + timings.bilateral_ms
        + timings.variance_ms
        + timings.reproject_ms
        + timings.fuse_ms
        + timings.history_ms
    )
    assert stages == pytest.approx(timings.compute_ms, abs=0.01)
    assert timings.upload_ms + timings.compute_ms + timings.download_ms <= (
        result.diagnostics.host_process_ms
    )
