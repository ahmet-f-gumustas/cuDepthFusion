"""Run the baselines over a validated sequence and collect per-frame metrics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from cudepthfusion.config import LoadedConfig
from cudepthfusion.data.icl_nuim import IclGroundTruth, IclInputSequence
from cudepthfusion.data.manifest import file_sha256, load_manifest
from cudepthfusion.eval.baselines import BASELINE_NAMES, Baseline, make_baselines
from cudepthfusion.eval.metrics import (
    FrameMetrics,
    edge_mask,
    evaluation_mask,
    frame_metrics,
    sanitize,
)
from cudepthfusion.eval.stability import StabilityResult, StabilityTracker

Logger = Callable[[str], None]


@dataclass(frozen=True)
class SequenceResult:
    sequence: str
    split: str
    manifest_path: str
    manifest_sha256: str
    frames_evaluated: int
    baselines: dict[str, str]  # name -> description
    metrics: list[FrameMetrics] = field(default_factory=list)
    stability: list[StabilityResult] = field(default_factory=list)


def evaluate_sequence(
    manifest_path: Path,
    config: LoadedConfig,
    *,
    backend: str = "cpu",
    baselines: tuple[str, ...] = BASELINE_NAMES,
    frames: int | None = None,
    log: Logger = lambda _: None,
    methods: Sequence[Baseline] | None = None,
    track_stability: bool = False,
    **baseline_options: float,
) -> SequenceResult:
    """Evaluate one sequence. Clean depth is read here and never handed to a baseline."""
    manifest = load_manifest(manifest_path)
    sequence = IclInputSequence(manifest_path)  # refuses unvalidated data
    truth = IclGroundTruth(manifest_path)
    methods = (
        list(methods)
        if methods is not None
        else make_baselines(config, backend, baselines, **baseline_options)
    )
    tracker = (
        StabilityTracker(sequence.frame(sequence.frame_ids[0]).intrinsics)
        if track_stability
        else None
    )
    minimum = config.core.depth.min_m
    maximum = config.core.depth.max_m

    records: list[FrameMetrics] = []
    evaluated = 0
    for index, frame in enumerate(sequence):
        if frames is not None and index >= frames:
            break
        clean = truth.clean_depth_m(frame.frame_id)
        raw_valid = sanitize(frame.depth_m, minimum, maximum)
        mask = evaluation_mask(clean, raw_valid)
        edges = edge_mask(clean)
        outputs = {}
        for method in methods:
            depth, valid = method.process(frame)
            outputs[method.name] = (depth, valid)
            records.append(
                frame_metrics(
                    manifest["sequence"],
                    method.name,
                    frame.frame_id,
                    depth,
                    valid,
                    clean,
                    mask,
                    edges,
                    raw_valid,
                )
            )
        if tracker is not None:
            tracker.observe(clean, frame.T_world_camera, outputs)
        evaluated += 1
        if evaluated % 100 == 0:
            log(f"  {manifest['sequence']}: {evaluated} frames")

    return SequenceResult(
        sequence=manifest["sequence"],
        split=manifest["split"],
        manifest_path=str(manifest_path),
        manifest_sha256=file_sha256(manifest_path),
        frames_evaluated=evaluated,
        baselines={method.name: method.description for method in methods},
        metrics=records,
        stability=tracker.results() if tracker is not None else [],
    )
