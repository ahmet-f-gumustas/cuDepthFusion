"""Evaluation protocol: masks, metrics, baselines and run artifacts (spec 10).

The evaluator is the only part that may touch clean ground truth. Baselines receive an
``InputFrame`` and nothing else.
"""

from cudepthfusion.eval.baselines import (
    BASELINE_NAMES,
    Baseline,
    EmaBaseline,
    EngineBaseline,
    RawBaseline,
    make_baselines,
)
from cudepthfusion.eval.metrics import (
    FrameMetrics,
    aggregate,
    edge_mask,
    evaluation_mask,
    frame_metrics,
    sanitize,
)
from cudepthfusion.eval.runner import SequenceResult, evaluate_sequence

__all__ = [
    "BASELINE_NAMES",
    "Baseline",
    "EmaBaseline",
    "EngineBaseline",
    "FrameMetrics",
    "RawBaseline",
    "SequenceResult",
    "aggregate",
    "edge_mask",
    "evaluate_sequence",
    "evaluation_mask",
    "frame_metrics",
    "make_baselines",
    "sanitize",
]
