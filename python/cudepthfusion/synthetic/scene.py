"""Scenes made of planar rectangles, rendered by exact ray casting.

A rectangle with infinite half extents is an unbounded plane. Rectangles may move with a
constant velocity, which makes them dynamic objects in an otherwise static world.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cudepthfusion.data.camera import PinholeCamera

NO_SURFACE = 0
MIN_DISTANCE_M = 1e-6
AXIS_TOLERANCE = 1e-9

Vector = tuple[float, float, float]


@dataclass(frozen=True)
class Rectangle:
    """Planar patch spanned by orthonormal ``u_axis``/``v_axis`` around ``center`` (metres)."""

    surface_id: int
    center: Vector
    u_axis: Vector
    v_axis: Vector
    half_u: float = math.inf
    half_v: float = math.inf
    velocity: Vector = (0.0, 0.0, 0.0)  # m/s; the centre moves, the orientation does not

    def __post_init__(self) -> None:
        if self.surface_id <= NO_SURFACE:
            raise ValueError(f"surface_id must be >= 1 (0 means no surface), got {self.surface_id}")
        u, v = np.asarray(self.u_axis, float), np.asarray(self.v_axis, float)
        if (
            abs(np.linalg.norm(u) - 1) > AXIS_TOLERANCE
            or abs(np.linalg.norm(v) - 1) > AXIS_TOLERANCE
        ):
            raise ValueError(f"surface {self.surface_id}: u_axis and v_axis must be unit vectors")
        if abs(float(u @ v)) > AXIS_TOLERANCE:
            raise ValueError(f"surface {self.surface_id}: u_axis and v_axis must be orthogonal")
        if not (self.half_u > 0 and self.half_v > 0):
            raise ValueError(f"surface {self.surface_id}: half extents must be positive")

    @property
    def normal(self) -> np.ndarray:
        return np.cross(np.asarray(self.u_axis, float), np.asarray(self.v_axis, float))

    @property
    def is_dynamic(self) -> bool:
        return any(component != 0.0 for component in self.velocity)

    def center_at(self, time_s: float) -> np.ndarray:
        return np.asarray(self.center, float) + time_s * np.asarray(self.velocity, float)


@dataclass(frozen=True)
class Scene:
    surfaces: tuple[Rectangle, ...]

    def __post_init__(self) -> None:
        ids = [surface.surface_id for surface in self.surfaces]
        if len(ids) != len(set(ids)):
            raise ValueError(f"surface ids must be unique, got {ids}")

    @property
    def dynamic_surface_ids(self) -> frozenset[int]:
        return frozenset(s.surface_id for s in self.surfaces if s.is_dynamic)


def wall(
    surface_id: int,
    distance_m: float,
    *,
    half_u: float = math.inf,
    half_v: float = math.inf,
    center_xy: tuple[float, float] = (0.0, 0.0),
    velocity: Vector = (0.0, 0.0, 0.0),
) -> Rectangle:
    """A patch parallel to the world x/y plane at ``z = distance_m``."""
    return Rectangle(
        surface_id=surface_id,
        center=(center_xy[0], center_xy[1], distance_m),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.0),
        half_u=half_u,
        half_v=half_v,
        velocity=velocity,
    )


def box_room(minimum: Sequence[float], maximum: Sequence[float], first_id: int = 1) -> tuple:
    """Six unbounded planes enclosing an axis-aligned box; seen from inside, the first hit
    along any ray is the wall the ray exits through."""
    axes = {0: ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), 1: ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0))}
    axes[2] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    planes = []
    for axis in range(3):
        u_axis, v_axis = axes[axis]
        for bound in (minimum[axis], maximum[axis]):
            center = [0.0, 0.0, 0.0]
            center[axis] = float(bound)
            planes.append(
                Rectangle(first_id + len(planes), (center[0], center[1], center[2]), u_axis, v_axis)
            )
    return tuple(planes)


def render(
    scene: Scene, camera: PinholeCamera, T_world_camera: np.ndarray, time_s: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Exact camera Z (float64, 0 where no surface is hit) and surface id (uint16) per pixel."""
    rays_world = camera.rays() @ T_world_camera[:3, :3].T
    origin = T_world_camera[:3, 3]
    depth = np.full((camera.height, camera.width), np.inf)
    surface_id = np.zeros((camera.height, camera.width), dtype=np.uint16)
    with np.errstate(divide="ignore", invalid="ignore"):
        for surface in scene.surfaces:
            center = surface.center_at(time_s)
            normal = surface.normal
            distance = ((center - origin) @ normal) / (rays_world @ normal)
            offset = origin + distance[..., None] * rays_world - center
            inside = (np.abs(offset @ np.asarray(surface.u_axis)) <= surface.half_u) & (
                np.abs(offset @ np.asarray(surface.v_axis)) <= surface.half_v
            )
            closer = (
                np.isfinite(distance) & (distance > MIN_DISTANCE_M) & inside & (distance < depth)
            )
            depth[closer] = distance[closer]
            surface_id[closer] = surface.surface_id
    depth[~np.isfinite(depth)] = 0.0
    return depth, surface_id
