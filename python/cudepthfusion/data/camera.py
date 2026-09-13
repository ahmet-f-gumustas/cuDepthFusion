"""Pinhole camera helpers shared by dataset adapters, the synthetic generator and checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cudepthfusion import _core

VGA_FOCAL_PX = 525.0  # Kinect-like focal length at 640 x 480


def ray_norms(width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """||K^-1 [u, v, 1]^T|| per pixel: converts ray distance r to camera Z = r / norm."""
    u = (np.arange(width, dtype=np.float64) - cx) / fx
    v = (np.arange(height, dtype=np.float64) - cy) / fy
    return np.sqrt(1.0 + u[None, :] ** 2 + v[:, None] ** 2)


@dataclass(frozen=True)
class PinholeCamera:
    """Distortion-free pinhole camera. Pixel centres are integer coordinates; fy may be < 0."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError(f"camera size must be positive, got {self.width}x{self.height}")
        if not self.fx > 0 or self.fy == 0:
            raise ValueError(f"need fx > 0 and fy != 0, got fx={self.fx}, fy={self.fy}")

    @classmethod
    def vga(cls) -> PinholeCamera:
        """640 x 480, the spec's starting resolution."""
        return cls(640, 480, VGA_FOCAL_PX, VGA_FOCAL_PX, 319.5, 239.5)

    def scaled(self, factor: float) -> PinholeCamera:
        """Same field of view at ``factor`` times the resolution (for small test images)."""
        return PinholeCamera(
            width=round(self.width * factor),
            height=round(self.height * factor),
            fx=self.fx * factor,
            fy=self.fy * factor,
            cx=(self.cx + 0.5) * factor - 0.5,
            cy=(self.cy + 0.5) * factor - 0.5,
        )

    def rays(self) -> np.ndarray:
        """(H, W, 3) camera-frame rays with unit Z, so a ray parameter equals camera Z."""
        u, v = np.meshgrid(
            np.arange(self.width, dtype=np.float64), np.arange(self.height, dtype=np.float64)
        )
        return np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u)], -1)

    def ray_norms(self) -> np.ndarray:
        return ray_norms(self.width, self.height, self.fx, self.fy, self.cx, self.cy)

    def intrinsics(self) -> _core.Intrinsics:
        return _core.Intrinsics(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)
