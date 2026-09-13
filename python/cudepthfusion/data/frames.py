"""The per-frame filter input shared by every dataset source (real or synthetic)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cudepthfusion import _core


@dataclass(frozen=True, eq=False)
class InputFrame:
    """Everything the filter may see for one frame. There is deliberately no clean depth."""

    frame_id: int
    timestamp_s: float
    depth_m: np.ndarray  # float32 (H, W), metres, C-contiguous; 0 = invalid
    intrinsics: _core.Intrinsics
    T_world_camera: np.ndarray  # float64 (4, 4), camera -> world, C-contiguous
