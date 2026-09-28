"""Temporal stability on fixed world points (spec 10.3).

The variance of one pixel over time is not stability when the camera moves: the pixel sees a
different surface each frame. So the evaluator picks world points from the clean depth, follows
them with the ground-truth poses, keeps only the frames where they are really visible, and
measures how much each method's error wobbles at those points.

The tracks are chosen from ground truth, never from a method's own output, so a filter cannot
improve the metric by keeping only the pixels it finds easy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cudepthfusion.data.poses import invert_rigid

VISIBILITY_TOLERANCE_M = 0.02
MIN_OBSERVATIONS = 5


@dataclass
class _Track:
    point_world: np.ndarray  # (3,)
    errors: dict[str, list[float]] = field(default_factory=dict)
    frames_seen: int = 0


@dataclass(frozen=True)
class StabilityResult:
    baseline: str
    tracked_points: int
    observations: int
    median_point_std_m: float
    p90_point_std_m: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "baseline": self.baseline,
            "tracked_points": self.tracked_points,
            "observations": self.observations,
            "median_point_std_m": self.median_point_std_m,
            "p90_point_std_m": self.p90_point_std_m,
        }


class StabilityTracker:
    """Streaming tracker: new points every ``reference_every`` frames, each followed for a while."""

    def __init__(
        self,
        intrinsics,
        *,
        stride: int = 8,
        reference_every: int = 30,
        track_length: int = 30,
        visibility_tolerance_m: float = VISIBILITY_TOLERANCE_M,
    ) -> None:
        self._fx = intrinsics.fx
        self._fy = intrinsics.fy
        self._cx = intrinsics.cx
        self._cy = intrinsics.cy
        self._stride = stride
        self._reference_every = reference_every
        self._track_length = track_length
        self._tolerance = visibility_tolerance_m
        self._active: list[tuple[_Track, int]] = []  # (track, frames left)
        self._finished: list[_Track] = []
        self._index = 0

    def observe(
        self,
        clean: np.ndarray,
        pose: np.ndarray,
        outputs: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        self._follow(clean, pose, outputs)
        if self._index % self._reference_every == 0:
            self._seed(clean, pose)
        self._index += 1

    def results(self) -> list[StabilityResult]:
        tracks = self._finished + [track for track, _ in self._active]
        names = sorted({name for track in tracks for name in track.errors})
        results = []
        for name in names:
            deviations = [
                float(np.std(track.errors[name]))
                for track in tracks
                if len(track.errors.get(name, ())) >= MIN_OBSERVATIONS
            ]
            observations = sum(len(track.errors.get(name, ())) for track in tracks)
            if not deviations:
                continue
            results.append(
                StabilityResult(
                    baseline=name,
                    tracked_points=len(deviations),
                    observations=observations,
                    median_point_std_m=float(np.median(deviations)),
                    p90_point_std_m=float(np.percentile(deviations, 90)),
                )
            )
        return results

    def _seed(self, clean: np.ndarray, pose: np.ndarray) -> None:
        height, width = clean.shape
        rows = np.arange(self._stride // 2, height, self._stride)
        columns = np.arange(self._stride // 2, width, self._stride)
        grid_y, grid_x = np.meshgrid(rows, columns, indexing="ij")
        depth = clean[grid_y, grid_x]
        usable = depth > 0
        if not usable.any():
            return
        x = (grid_x[usable] - self._cx) / self._fx * depth[usable]
        y = (grid_y[usable] - self._cy) / self._fy * depth[usable]
        z = depth[usable]
        camera_points = np.stack([x, y, z, np.ones_like(z)], axis=0)
        world = (pose @ camera_points)[:3].T
        for point in world:
            self._active.append((_Track(point_world=point), self._track_length))

    def _follow(
        self,
        clean: np.ndarray,
        pose: np.ndarray,
        outputs: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        if not self._active:
            return
        height, width = clean.shape
        world = np.stack([track.point_world for track, _ in self._active], axis=0)
        homogeneous = np.concatenate([world, np.ones((world.shape[0], 1))], axis=1).T
        camera = (invert_rigid(pose) @ homogeneous)[:3]
        z = camera[2]
        in_front = z > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.floor(self._fx * camera[0] / z + self._cx + 0.5)
            v = np.floor(self._fy * camera[1] / z + self._cy + 0.5)
        inside = in_front & np.isfinite(u) & np.isfinite(v)
        inside &= (u >= 0) & (u < width) & (v >= 0) & (v < height)

        columns = np.where(inside, u, 0).astype(np.int64)
        rows = np.where(inside, v, 0).astype(np.int64)
        truth = clean[rows, columns]
        # The point must actually be the visible surface there, not hidden behind something.
        visible = inside & (truth > 0) & (np.abs(truth - z) <= self._tolerance)

        still_active: list[tuple[_Track, int]] = []
        for index, (track, remaining) in enumerate(self._active):
            if visible[index]:
                track.frames_seen += 1
                for name, (depth, valid) in outputs.items():
                    if valid[rows[index], columns[index]]:
                        error = float(depth[rows[index], columns[index]] - truth[index])
                        track.errors.setdefault(name, []).append(error)
            if remaining > 1:
                still_active.append((track, remaining - 1))
            else:
                self._finished.append(track)
        self._active = still_active
