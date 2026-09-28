"""Masks and per-frame metrics (spec 10.2).

Every metric is computed on the fixed mask M = clean-valid AND raw-input-valid, so a method
cannot improve its numbers by dropping pixels. Pixels it fails to produce inside M are
counted as missing coverage instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

EDGE_JUMP_M = 0.05
EDGE_DILATE_PX = 2
BAD_PIXEL_ABS_M = 0.02
BAD_PIXEL_REL = 0.01


def sanitize(depth: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    """The engine's validity rule: finite and inside the configured range."""
    return np.isfinite(depth) & (depth >= min_m) & (depth <= max_m)


def evaluation_mask(clean: np.ndarray, raw_valid: np.ndarray) -> np.ndarray:
    return (clean > 0) & raw_valid


def edge_mask(
    depth: np.ndarray, jump_m: float = EDGE_JUMP_M, dilate: int = EDGE_DILATE_PX
) -> np.ndarray:
    """Pixels next to a depth discontinuity in the ground truth, dilated (spec 10.2)."""
    edge = np.zeros(depth.shape, dtype=bool)
    for axis in (0, 1):
        step = np.abs(np.diff(depth, axis=axis)) > jump_m
        low = [slice(None), slice(None)]
        high = [slice(None), slice(None)]
        low[axis] = slice(0, -1)
        high[axis] = slice(1, None)
        edge[tuple(low)] |= step
        edge[tuple(high)] |= step
    for _ in range(dilate):
        grown = edge.copy()
        grown[1:, :] |= edge[:-1, :]
        grown[:-1, :] |= edge[1:, :]
        grown[:, 1:] |= edge[:, :-1]
        grown[:, :-1] |= edge[:, 1:]
        edge = grown
    return edge


@dataclass(frozen=True)
class FrameMetrics:
    """One baseline on one frame. The sums let sequences be pooled exactly."""

    sequence: str
    baseline: str
    frame_id: int
    num_mask: int
    num_scored: int  # pixels of M the method actually produced
    num_missing: int
    sum_squared_error: float
    sum_abs_error: float
    sum_error: float
    num_bad: int
    num_edge: int
    edge_sum_squared: float
    edge_sum_abs: float
    num_interior: int
    interior_sum_squared: float
    interior_sum_abs: float
    median_abs_m: float
    p90_abs_m: float
    num_hole_filled: int
    hole_sum_squared: float

    @property
    def coverage(self) -> float:
        return self.num_scored / self.num_mask if self.num_mask else 0.0

    @property
    def rmse_m(self) -> float:
        return (
            float(np.sqrt(self.sum_squared_error / self.num_scored))
            if self.num_scored
            else float("nan")
        )

    @property
    def mae_m(self) -> float:
        return self.sum_abs_error / self.num_scored if self.num_scored else float("nan")

    @property
    def bias_m(self) -> float:
        return self.sum_error / self.num_scored if self.num_scored else float("nan")

    @property
    def bad_pixel_rate(self) -> float:
        return self.num_bad / self.num_scored if self.num_scored else float("nan")

    def row(self) -> dict[str, Any]:
        """Flat record for per_frame.csv."""
        data = asdict(self)
        data.update(
            coverage=self.coverage,
            rmse_m=self.rmse_m,
            mae_m=self.mae_m,
            bias_m=self.bias_m,
            bad_pixel_rate=self.bad_pixel_rate,
        )
        return data


def frame_metrics(
    sequence: str,
    baseline: str,
    frame_id: int,
    depth: np.ndarray,
    valid: np.ndarray,
    clean: np.ndarray,
    mask: np.ndarray,
    edges: np.ndarray,
    raw_valid: np.ndarray,
) -> FrameMetrics:
    scored = mask & valid
    error = (depth - clean)[scored]
    absolute = np.abs(error)
    threshold = np.maximum(BAD_PIXEL_ABS_M, BAD_PIXEL_REL * clean[scored])
    holes = valid & ~raw_valid & (clean > 0)  # produced where the input had nothing

    def sums(selection: np.ndarray) -> tuple[float, float]:
        if not selection.any():
            return 0.0, 0.0
        difference = depth[selection] - clean[selection]
        return float(np.sum(difference**2)), float(np.sum(np.abs(difference)))

    edge_scored = scored & edges
    interior_scored = scored & ~edges
    edge_squared, edge_abs = sums(edge_scored)
    interior_squared, interior_abs = sums(interior_scored)
    hole_squared, _ = sums(holes)

    return FrameMetrics(
        sequence=sequence,
        baseline=baseline,
        frame_id=frame_id,
        num_mask=int(mask.sum()),
        num_scored=int(scored.sum()),
        num_missing=int((mask & ~valid).sum()),
        sum_squared_error=float(np.sum(error**2)),
        sum_abs_error=float(np.sum(absolute)),
        sum_error=float(np.sum(error)),
        num_bad=int(np.count_nonzero(absolute > threshold)),
        num_edge=int(edge_scored.sum()),
        edge_sum_squared=edge_squared,
        edge_sum_abs=edge_abs,
        num_interior=int(interior_scored.sum()),
        interior_sum_squared=interior_squared,
        interior_sum_abs=interior_abs,
        median_abs_m=float(np.median(absolute)) if absolute.size else float("nan"),
        p90_abs_m=float(np.percentile(absolute, 90)) if absolute.size else float("nan"),
        num_hole_filled=int(holes.sum()),
        hole_sum_squared=hole_squared,
    )


def aggregate(frames: list[FrameMetrics]) -> dict[str, Any]:
    """Pool a list of per-frame records exactly (sums, not averages of averages)."""
    if not frames:
        return {}
    scored = sum(f.num_scored for f in frames)
    mask_total = sum(f.num_mask for f in frames)
    edge = sum(f.num_edge for f in frames)
    interior = sum(f.num_interior for f in frames)
    squared = sum(f.sum_squared_error for f in frames)
    absolute = sum(f.sum_abs_error for f in frames)
    signed = sum(f.sum_error for f in frames)
    holes = sum(f.num_hole_filled for f in frames)
    result = {
        "frames": len(frames),
        "pixels_in_mask": mask_total,
        "pixels_scored": scored,
        "coverage": scored / mask_total if mask_total else 0.0,
        "rmse_m": float(np.sqrt(squared / scored)) if scored else float("nan"),
        "mae_m": absolute / scored if scored else float("nan"),
        "bias_m": signed / scored if scored else float("nan"),
        "bad_pixel_rate": sum(f.num_bad for f in frames) / scored if scored else float("nan"),
        "median_abs_m": float(np.median([f.median_abs_m for f in frames])),
        "p90_abs_m": float(np.median([f.p90_abs_m for f in frames])),
        "edge_rmse_m": float(np.sqrt(sum(f.edge_sum_squared for f in frames) / edge))
        if edge
        else float("nan"),
        "edge_mae_m": sum(f.edge_sum_abs for f in frames) / edge if edge else float("nan"),
        "interior_rmse_m": float(np.sqrt(sum(f.interior_sum_squared for f in frames) / interior))
        if interior
        else float("nan"),
        "interior_mae_m": sum(f.interior_sum_abs for f in frames) / interior
        if interior
        else float("nan"),
        "holes_filled": holes,
        "hole_rmse_m": float(np.sqrt(sum(f.hole_sum_squared for f in frames) / holes))
        if holes
        else None,
    }
    return result
