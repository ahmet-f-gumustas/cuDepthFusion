"""The side-by-side comparison demo (spec 11).

Runs the raw input, B1 (spatial filter only) and B4 (full fusion) over a validated sequence,
renders them into one frame with shared scales, and writes PNGs, an MP4 and the same run
artifacts the evaluator writes. Three different times are kept apart, because mixing them is
how a demo ends up quoting a frame rate that no camera would ever see:

* **process latency** — the engine call itself, NumPy in to owned NumPy out (spec 10.5.2);
* **demo throughput** — decode, process, render and video writing together (spec 10.5.3);
* **GPU compute** — CUDA events around the kernel chain, which this demo does not measure and
  therefore reports as ``null`` with the reason, never as a zero.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import cudepthfusion as cdf
from cudepthfusion.config import LoadedConfig
from cudepthfusion.data.icl_nuim import IclGroundTruth, IclInputSequence
from cudepthfusion.data.manifest import file_sha256, load_manifest
from cudepthfusion.eval.artifacts import (
    LATENCY_NAME,
    environment_info,
    git_commit,
    utc_now,
    write_run,
    write_table,
)
from cudepthfusion.eval.metrics import (
    aggregate,
    edge_mask,
    evaluation_mask,
    frame_metrics,
    sanitize,
)
from cudepthfusion.viz import render
from cudepthfusion.viz.render import Scales

Logger = Callable[[str], None]
DEFAULT_ERROR_MAX_MM = 100.0
DEFAULT_FPS = 15.0
VIDEO_FOURCC = "mp4v"
GPU_COMPUTE_REASON = "not measured here: CUDA-event kernel timing belongs to the P7 benchmark"


class DemoError(RuntimeError):
    """Something the demo cannot do, said plainly instead of producing an empty file."""


SIXTH_PANELS = ("source", "confidence", "error_b1")
RANGE_SAMPLE_FRAMES = 3


@dataclass(frozen=True)
class DemoOptions:
    frames: int | None = None
    panel_width: int = 420
    depth_range: tuple[float, float] | None = None
    error_max_mm: float = DEFAULT_ERROR_MAX_MM
    sixth_panel: str = "source"  # or "confidence", "error_b1"
    fps: float = DEFAULT_FPS
    save_video: Path | None = None
    save_frames: Path | None = None
    interactive: bool = False


@dataclass
class _Timings:
    process_b1_ms: list[float] = field(default_factory=list)
    process_b4_ms: list[float] = field(default_factory=list)
    render_ms: list[float] = field(default_factory=list)
    write_ms: list[float] = field(default_factory=list)
    total_ms: list[float] = field(default_factory=list)


def _stats(samples: list[float]) -> dict[str, Any]:
    """Median/p95/p99 of real consecutive frames, or nulls when nothing was measured."""
    if not samples:
        return {"frames": 0, "median_ms": None, "p95_ms": None, "p99_ms": None}
    array = np.asarray(samples, dtype=float)
    return {
        "frames": int(array.size),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


def _sampled_depth_range(
    sequence: IclInputSequence, frame_ids: tuple[int, ...], minimum: float, maximum: float
) -> tuple[float, float]:
    """Decode a few frames spread over the run and fix one range from all of them."""
    count = min(RANGE_SAMPLE_FRAMES, len(frame_ids))
    picks = np.unique(np.linspace(0, len(frame_ids) - 1, count).round().astype(int))
    samples = []
    for index in picks:
        depth = sequence.frame(frame_ids[int(index)]).depth_m
        samples.append(depth[sanitize(depth, minimum, maximum)])
    return render.auto_depth_range(samples)


def _ground_truth(manifest_path: Path) -> tuple[IclGroundTruth | None, str | None]:
    manifest = load_manifest(manifest_path)
    if "clean" not in manifest.get("files", {}):
        return None, "this sequence has no clean variant, so no error panel is possible"
    return IclGroundTruth(manifest_path), None


def _open_video(path: Path, size: tuple[int, int], fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*VIDEO_FOURCC), fps, size)
    if not writer.isOpened():
        raise DemoError(
            f"OpenCV could not open {path} for writing with the {VIDEO_FOURCC} codec; "
            "write PNG frames with --save-frames instead, or install an OpenCV build with "
            "video support"
        )
    return writer


def _show(image: np.ndarray, window: str) -> bool:
    """Interactive playback: space pauses, n steps, q or Esc quits. Returns False to stop."""
    try:
        cv2.imshow(window, image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            while True:
                key = cv2.waitKey(50) & 0xFF
                if key in (ord(" "), ord("n"), ord("q"), 27):
                    break
    except cv2.error as exc:  # headless OpenCV has no GUI at all
        raise DemoError(
            "this OpenCV build cannot open a window (opencv-python-headless); drop "
            f"--interactive and use --save-video/--save-frames, or install opencv-python: {exc}"
        ) from exc
    return key not in (ord("q"), 27)


def run_demo(
    manifest_path: Path,
    config: LoadedConfig,
    *,
    backend: str = "cpu",
    options: DemoOptions | None = None,
    log: Logger = lambda _: None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Render the comparison over one validated sequence; returns the summary and the rows."""
    options = options or DemoOptions()
    sequence = IclInputSequence(manifest_path)  # refuses unvalidated data
    truth, truth_missing = _ground_truth(manifest_path)
    name = manifest_path.parent.name
    spatial_only = cdf.DepthFusion(config, backend=backend)
    fusion = cdf.DepthFusion(config, backend=backend)
    minimum = config.core.depth.min_m
    maximum = config.core.depth.max_m

    frame_ids = sequence.frame_ids[: options.frames] if options.frames else sequence.frame_ids
    low, high = options.depth_range or _sampled_depth_range(sequence, frame_ids, minimum, maximum)
    scales = Scales(low, high, options.error_max_mm)
    log(
        f"panels share depth {low:.2f}-{high:.2f} m and error "
        f"0-{options.error_max_mm:.0f} mm for every frame"
    )
    writer: cv2.VideoWriter | None = None
    window = f"cuDepthFusion — {name}"
    timings = _Timings()
    rows: list[dict[str, Any]] = []
    metrics: dict[str, list] = {"B1": [], "B4": []}
    rendered = 0
    stopped_early = False

    try:
        for index, frame in enumerate(sequence):
            if options.frames is not None and index >= options.frames:
                break
            started = time.perf_counter()
            raw_valid = sanitize(frame.depth_m, minimum, maximum)
            raw_depth = np.where(raw_valid, frame.depth_m, 0.0).astype(np.float32)

            # B1 sees no pose, so its temporal stage stays off; B4 is the full method.
            b1_started = time.perf_counter()
            b1 = spatial_only.process(
                depth_m=frame.depth_m,
                intrinsics=frame.intrinsics,
                T_world_camera=None,
                timestamp_s=frame.timestamp_s,
            )
            b1_ms = (time.perf_counter() - b1_started) * 1000.0
            b4_started = time.perf_counter()
            b4 = fusion.process(
                depth_m=frame.depth_m,
                intrinsics=frame.intrinsics,
                T_world_camera=frame.T_world_camera,
                timestamp_s=frame.timestamp_s,
            )
            b4_ms = (time.perf_counter() - b4_started) * 1000.0

            clean = truth.clean_depth_m(frame.frame_id) if truth is not None else None
            render_started = time.perf_counter()
            b1_valid = b1.valid_mask == 1
            b4_valid = b4.valid_mask == 1
            panels = [
                render.depth_panel("B0 raw input", raw_depth, raw_valid, scales),
                render.depth_panel("B1 spatial filter", b1.depth_m, b1_valid, scales),
                render.depth_panel("B4 fusion", b4.depth_m, b4_valid, scales),
            ]
            if clean is None:
                shape = raw_depth.shape
                panels.append(render.missing_panel("clean ground truth", shape, truth_missing))
                panels.append(render.missing_panel("|B4 - GT|", shape, truth_missing))
            else:
                clean_valid = clean > 0
                mask = evaluation_mask(clean, raw_valid)
                panels.append(render.depth_panel("clean ground truth", clean, clean_valid, scales))
                panels.append(
                    render.error_panel("|B4 - GT|", b4.depth_m - clean, mask & b4_valid, scales)
                )
                edges = edge_mask(clean)
                for label, depth, valid in (
                    ("B1", b1.depth_m, b1_valid),
                    ("B4", b4.depth_m, b4_valid),
                ):
                    metrics[label].append(
                        frame_metrics(
                            name, label, frame.frame_id, depth, valid, clean, mask, edges, raw_valid
                        )
                    )
            if options.sixth_panel == "confidence":
                panels.append(
                    render.confidence_panel("B4 confidence", b4.confidence_score, b4_valid)
                )
            elif options.sixth_panel == "error_b1" and clean is not None:
                panels.append(
                    render.error_panel(
                        "|B1 - GT| (same scale)",
                        b1.depth_m - clean,
                        evaluation_mask(clean, raw_valid) & b1_valid,
                        scales,
                    )
                )
            elif options.sixth_panel == "error_b1":
                panels.append(render.missing_panel("|B1 - GT|", raw_depth.shape, truth_missing))
            else:
                panels.append(render.source_panel("B4 source", b4.source_mask))

            recent = timings.total_ms[-10:]
            demo_fps = 1000.0 / float(np.mean(recent)) if recent else float("nan")
            status = (
                f"{name} frame {frame.frame_id}  |  {backend} backend  |  "
                f"process latency B4 {b4_ms:.1f} ms (B1 {b1_ms:.1f} ms)  |  "
                f"demo {demo_fps:.1f} fps incl. decode and writing  |  "
                f"fused {b4.diagnostics.fusion.fused} px, "
                f"temporal {b4.diagnostics.temporal_status}"
            )
            image = render.compose(panels, scales, status, panel_width=options.panel_width)
            render_ms = (time.perf_counter() - render_started) * 1000.0

            write_started = time.perf_counter()
            if options.save_video is not None:
                if writer is None:
                    writer = _open_video(
                        options.save_video, (image.shape[1], image.shape[0]), options.fps
                    )
                writer.write(image)
            if options.save_frames is not None:
                options.save_frames.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(options.save_frames / f"frame_{frame.frame_id:05d}.png"), image)
            write_ms = (time.perf_counter() - write_started) * 1000.0
            total_ms = (time.perf_counter() - started) * 1000.0

            timings.process_b1_ms.append(b1_ms)
            timings.process_b4_ms.append(b4_ms)
            timings.render_ms.append(render_ms)
            timings.write_ms.append(write_ms)
            timings.total_ms.append(total_ms)
            rows.append(
                {
                    "frame_id": frame.frame_id,
                    "process_b1_ms": b1_ms,
                    "process_b4_ms": b4_ms,
                    "engine_host_ms_b4": b4.diagnostics.host_process_ms,
                    "render_ms": render_ms,
                    "write_ms": write_ms,
                    "total_ms": total_ms,
                    "fused_px": b4.diagnostics.fusion.fused,
                    "current_only_px": b4.diagnostics.fusion.current_only,
                    "history_only_px": b4.diagnostics.fusion.history_only,
                    "mean_prior_weight": b4.diagnostics.fusion.mean_prior_weight,
                    "rmse_b1_m": metrics["B1"][-1].rmse_m if metrics["B1"] else None,
                    "rmse_b4_m": metrics["B4"][-1].rmse_m if metrics["B4"] else None,
                }
            )
            rendered += 1
            if options.interactive and not _show(image, window):
                stopped_early = True
                log("stopped by the user")
                break
    finally:
        if writer is not None:
            writer.release()
        if options.interactive:
            cv2.destroyAllWindows()

    if rendered == 0:
        raise DemoError(f"{manifest_path} produced no frames to render")

    quality = {label: aggregate(records) for label, records in metrics.items() if records}
    summary = {
        "kind": "demo",
        "created_utc": utc_now(),
        "git_commit": git_commit(),
        "sequence": name,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "backend": backend,
        "frames_rendered": rendered,
        "stopped_early": stopped_early,
        "scales": scales.to_dict(),
        "panels": {"sixth": options.sixth_panel, "width_px": options.panel_width},
        "ground_truth": {"available": truth is not None, "reason": truth_missing},
        "latency": {
            "process_b4": _stats(timings.process_b4_ms),
            "process_b1": _stats(timings.process_b1_ms),
            "render": _stats(timings.render_ms),
            "write": _stats(timings.write_ms),
            "demo_throughput": _stats(timings.total_ms),
            "gpu_compute": {"median_ms": None, "reason": GPU_COMPUTE_REASON},
            "includes": "decode, process, render and file writing; not a camera latency",
        },
        "quality": quality or None,
        "quality_note": None if quality else truth_missing,
        "outputs": {
            "video": str(options.save_video) if options.save_video else None,
            "frames_dir": str(options.save_frames) if options.save_frames else None,
        },
        "environment": environment_info(),
    }
    return summary, rows


PER_FRAME_COLUMNS = (
    "frame_id",
    "fused_px",
    "current_only_px",
    "history_only_px",
    "mean_prior_weight",
    "rmse_b1_m",
    "rmse_b4_m",
)
LATENCY_COLUMNS = (
    "frame_id",
    "process_b1_ms",
    "process_b4_ms",
    "engine_host_ms_b4",
    "render_ms",
    "write_ms",
    "total_ms",
)


def write_demo_run(
    output: Path, summary: dict[str, Any], rows: list[dict[str, Any]], config: LoadedConfig
) -> None:
    """The same artifacts the evaluator writes, plus latency.csv (spec 11)."""
    write_run(
        output, summary, [{key: row[key] for key in PER_FRAME_COLUMNS} for row in rows], config
    )
    write_table(output / LATENCY_NAME, [{key: row[key] for key in LATENCY_COLUMNS} for row in rows])
