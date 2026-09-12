"""Pinhole camera helpers shared by dataset adapters and the evaluator-side checks."""

from __future__ import annotations

import numpy as np


def ray_norms(width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """||K^-1 [u, v, 1]^T|| per pixel: converts ray distance r to camera Z = r / norm."""
    u = (np.arange(width, dtype=np.float64) - cx) / fx
    v = (np.arange(height, dtype=np.float64) - cy) / fy
    return np.sqrt(1.0 + u[None, :] ** 2 + v[:, None] ** 2)
