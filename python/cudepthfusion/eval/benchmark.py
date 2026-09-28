"""Performance protocol (spec 10.5): real consecutive frames, warm-up, repeats, context.

Three times are recorded separately and never mixed:

* **GPU compute** — CUDA events around the kernel chain on the engine's stream, host copies
  excluded (``DeviceTimings.compute_ms``). Not measured on the CPU backend, so ``null`` there.
* **Process latency** — NumPy in to owned NumPy out around ``DepthFusion.process()``, measured
  here with ``perf_counter``: staging, H2D, D2H and the synchronisation are all inside it.
* **Offline demo throughput** — decoding and rendering included; that is the demo's number
  (``examples/compare_depth.py``), not this one.

Frames are decoded into memory before timing, so disk and PNG decoding never enter a
measurement. Every repeat resets the engine and replays the same slice: ``warmup_frames``
unmeasured frames, then ``measured_frames`` measured ones. The same frame is never fused over
and over, because a history that converged on a static input is not the workload.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import cudepthfusion as cdf
from cudepthfusion.config import LoadedConfig
from cudepthfusion.data.frames import InputFrame
from cudepthfusion.data.icl_nuim import IclInputSequence
from cudepthfusion.data.manifest import file_sha256

Logger = Callable[[str], None]
NVIDIA_SMI_FIELDS = (
    "name",
    "driver_version",
    "pstate",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.sm",
    "clocks.max.sm",
    "clocks.mem",
    "memory.used",
    "memory.total",
    "utilization.gpu",
)
DEVICE_FIELDS = (
    "upload_ms",
    "compute_ms",
    "download_ms",
    "sanitize_ms",
    "bilateral_ms",
    "variance_ms",
    "reproject_ms",
    "fuse_ms",
    "history_ms",
)
NOT_MEASURED_CPU = "not measured: the CPU backend has no device timeline"


class BenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class BenchmarkPlan:
    warmup_frames: int
    measured_frames: int
    repeats: int
    first_frame: int = 0

    @property
    def frames_needed(self) -> int:
        return self.warmup_frames + self.measured_frames


def _run(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def gpu_state() -> dict[str, Any]:
    """Clock, power, temperature and load right now, or the reason they are unavailable."""
    output = _run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(NVIDIA_SMI_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return {"available": False, "reason": "nvidia-smi is missing or reported nothing"}
    values = [value.strip() for value in output.splitlines()[0].split(",")]
    state: dict[str, Any] = {"available": True}
    for field, value in zip(NVIDIA_SMI_FIELDS, values, strict=False):
        state[field] = None if value in ("[N/A]", "N/A", "[Not Supported]") else value
    apps = _run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
    )
    state["other_compute_processes"] = _other_processes(apps or "", os.getpid())
    return state


def _other_processes(listing: str, own_pid: int) -> int:
    """Compute processes on the GPU other than this one (which is on the list itself)."""
    count = 0
    for line in listing.splitlines():
        pid = line.split(",")[0].strip()
        if pid and pid != str(own_pid):
            count += 1
    return count


def power_mode() -> dict[str, Any]:
    """Jetson power mode (nvpmodel); laptops report their power limit through gpu_state()."""
    output = _run(["nvpmodel", "-q"])
    if output is None:
        return {"nvpmodel": None, "reason": "nvpmodel not present (not a Jetson)"}
    return {"nvpmodel": output}


def load_frames(manifest: Path, plan: BenchmarkPlan) -> list[InputFrame]:
    """Decode the slice once, before anything is timed."""
    sequence = IclInputSequence(manifest)
    ids = sequence.frame_ids[plan.first_frame : plan.first_frame + plan.frames_needed]
    if len(ids) < plan.frames_needed:
        raise BenchmarkError(
            f"{manifest} has {len(sequence.frame_ids) - plan.first_frame} frames from index "
            f"{plan.first_frame}, the plan needs {plan.frames_needed}"
        )
    return [sequence.frame(frame_id) for frame_id in ids]


def _percentiles(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {"count": 0, "median_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    array = np.asarray(samples, dtype=float)
    return {
        "count": int(array.size),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "max_ms": float(array.max()),
    }


def run_benchmark(
    frames: list[InputFrame],
    config: LoadedConfig,
    plan: BenchmarkPlan,
    *,
    backend: str,
    log: Logger = lambda _: None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Time ``plan.repeats`` identical replays; returns the summary and one row per frame."""
    if len(frames) < plan.frames_needed:
        raise BenchmarkError(f"{len(frames)} frames given, the plan needs {plan.frames_needed}")
    engine = cdf.DepthFusion(config, backend=backend)
    rows: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    repeat_medians: list[float] = []

    for repeat in range(plan.repeats):
        engine.reset()
        states.append({"repeat": repeat, "before": gpu_state()})
        latencies = []
        for position, frame in enumerate(frames[: plan.frames_needed]):
            started = time.perf_counter()
            result = engine.process(
                depth_m=frame.depth_m,
                intrinsics=frame.intrinsics,
                T_world_camera=frame.T_world_camera,
                timestamp_s=frame.timestamp_s,
            )
            process_ms = (time.perf_counter() - started) * 1000.0
            if position < plan.warmup_frames:
                continue
            latencies.append(process_ms)
            timings = result.diagnostics.device_timings
            row: dict[str, Any] = {
                "repeat": repeat,
                "frame_id": frame.frame_id,
                "process_ms": process_ms,
                "engine_host_ms": result.diagnostics.host_process_ms,
                "temporal_status": result.diagnostics.temporal_status,
                "fused_px": result.diagnostics.fusion.fused,
            }
            for field in DEVICE_FIELDS:
                row[field] = getattr(timings, field) if timings is not None else None
            rows.append(row)
        states[-1]["after"] = gpu_state()
        repeat_medians.append(float(np.median(latencies)))
        log(f"repeat {repeat + 1}/{plan.repeats}: process median {repeat_medians[-1]:.3f} ms")

    device = {}
    for field in DEVICE_FIELDS:
        values = [row[field] for row in rows if row[field] is not None]
        device[field] = (
            _percentiles(values) if values else {"median_ms": None, "reason": NOT_MEASURED_CPU}
        )
    height, width = frames[0].depth_m.shape
    contended = any(
        state[moment].get("other_compute_processes", 0) > 0
        for state in states
        for moment in ("before", "after")
        if state.get(moment, {}).get("available")
    )
    summary = {
        "kind": "benchmark",
        "backend": backend,
        "resolution": {"width": int(width), "height": int(height)},
        "plan": {
            "warmup_frames": plan.warmup_frames,
            "measured_frames": plan.measured_frames,
            "repeats": plan.repeats,
            "first_frame": plan.first_frame,
            "frame_ids": [frames[0].frame_id, frames[plan.frames_needed - 1].frame_id],
        },
        "io_scope": "frames decoded into memory before timing; no disk or PNG decode inside",
        "process_latency": _percentiles([row["process_ms"] for row in rows]),
        "process_latency_repeat_medians_ms": repeat_medians,
        "engine_host": _percentiles([row["engine_host_ms"] for row in rows]),
        "gpu": device,
        "gpu_compute": device["compute_ms"],
        "gpu_state": states,
        "gpu_shared_with_other_processes": contended,
        "power_mode": power_mode(),
    }
    return summary, rows


def describe_source(manifest: Path) -> dict[str, Any]:
    return {"manifest_path": str(manifest), "manifest_sha256": file_sha256(manifest)}
