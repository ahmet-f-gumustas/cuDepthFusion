"""Analytic test scene: a camera moving inside an asymmetric box room.

Depth is ray-cast exactly, so any disagreement in a reprojection test comes from the
conventions under test and nearest-pixel rounding, not from rendering noise.
"""

from __future__ import annotations

import math

import numpy as np

# Room bounds in world metres (x, y, z); asymmetric so a mirrored axis cannot fit.
ROOM_MIN = np.array([-2.0, -1.0, -3.0])
ROOM_MAX = np.array([2.5, 2.0, 4.0])


def camera_pose(frame: int) -> np.ndarray:
    """T_world_camera of a smooth handheld-like trajectory (rotation about x and y)."""
    yaw = 0.012 * frame
    pitch = 0.3 * math.sin(0.04 * frame)
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    transform = np.eye(4)
    transform[:3, :3] = rot_y @ rot_x
    transform[:3, 3] = [
        0.3 + 0.01 * frame,
        0.2 + 0.4 * math.sin(0.05 * frame),
        -0.5 + 0.006 * frame,
    ]
    return transform


def render_z(
    T_world_camera: np.ndarray, width: int, height: int, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """Camera Z of the first room wall hit by each pixel ray (fy may be negative)."""
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    rays_c = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    rays_w = rays_c @ T_world_camera[:3, :3].T
    origin = T_world_camera[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        to_max = (ROOM_MAX - origin) / rays_w
        to_min = (ROOM_MIN - origin) / rays_w
    exit_t = np.where(rays_w > 0, to_max, np.where(rays_w < 0, to_min, np.inf))
    return exit_t.min(axis=-1)  # ray_c has unit Z, so the ray parameter is camera Z


def ray_distance(z: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    height, width = z.shape
    u = (np.arange(width) - cx) / fx
    v = (np.arange(height) - cy) / fy
    return z * np.sqrt(1.0 + u[None, :] ** 2 + v[:, None] ** 2)
