"""Camera trajectories: functions of time (seconds) returning T_world_camera (float64 4x4)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

Trajectory = Callable[[float], np.ndarray]


def rotation_about(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    """Rodrigues rotation of ``angle_rad`` about ``axis`` (normalised here)."""
    k = np.asarray(axis, dtype=np.float64)
    k = k / np.linalg.norm(k)
    skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle_rad) * skew + (1 - math.cos(angle_rad)) * skew @ skew


def rigid(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def static(position: Sequence[float] = (0.0, 0.0, 0.0)) -> Trajectory:
    pose = rigid(np.eye(3), position)
    return lambda _time_s: pose.copy()


def linear(velocity: Sequence[float], start: Sequence[float] = (0.0, 0.0, 0.0)) -> Trajectory:
    """Pure translation at constant velocity (m/s); orientation stays identity."""
    start_v, velocity_v = np.asarray(start, float), np.asarray(velocity, float)
    return lambda time_s: rigid(np.eye(3), start_v + time_s * velocity_v)


def spin(
    axis: Sequence[float], rate_rad_s: float, position: Sequence[float] = (0.0, 0.0, 0.0)
) -> Trajectory:
    """Pure rotation about the camera centre at a constant angular rate."""
    return lambda time_s: rigid(rotation_about(axis, rate_rad_s * time_s), position)


def handheld(frame_rate_hz: float = 30.0) -> Trajectory:
    """Smooth 6-DoF motion: yaw ramp, oscillating pitch, drifting translation."""

    def pose(time_s: float) -> np.ndarray:
        frame = time_s * frame_rate_hz
        yaw = 0.012 * frame
        pitch = 0.3 * math.sin(0.04 * frame)
        cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rot_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        translation = [0.3 + 0.01 * frame, 0.2 + 0.4 * math.sin(0.05 * frame), -0.5 + 0.006 * frame]
        return rigid(rot_y @ rot_x, translation)

    return pose
