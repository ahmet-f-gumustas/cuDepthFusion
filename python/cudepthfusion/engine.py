"""Python-facing engine: thin wrapper over the native core with typed results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cudepthfusion import _core
from cudepthfusion.config import ConfigSource, LoadedConfig, load_config


@dataclass(frozen=True)
class InputStats:
    num_pixels: int
    num_valid: int
    num_zero: int
    num_nonfinite: int
    num_below_min: int
    num_above_max: int


@dataclass(frozen=True)
class Diagnostics:
    frame_index: int
    reset_reason: str
    temporal_status: str
    spatial_applied: bool
    input: InputStats
    host_process_ms: float
    notes: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class FusionResult:
    """Owned H x W arrays; later process() calls never modify them.

    Invalid pixels are 0 in every array; ``valid_mask`` is authoritative.
    ``confidence_score`` is a display score in [0, 1), not a probability.
    """

    depth_m: np.ndarray  # float32, metres
    valid_mask: np.ndarray  # uint8
    variance_m2: np.ndarray  # float32, approximate error model
    confidence_score: np.ndarray  # float32
    source_mask: np.ndarray  # uint8: 0 invalid, 1 current, 2 fused, 3 history-only
    history_age: np.ndarray  # uint16, frames since the last supporting measurement
    diagnostics: Diagnostics


class DepthFusion:
    """Synchronous depth fusion engine.

    ``backend="cuda"`` raises ``BackendUnavailableError`` when CUDA cannot be used; it never
    falls back to the CPU. ``backend=None`` takes the backend from the configuration.
    """

    def __init__(self, config: ConfigSource = None, backend: str | None = None) -> None:
        loaded = load_config(config)
        # The native engine keeps its own validated copy; later edits to `loaded.core` by the
        # caller cannot reach it.
        self._engine = _core.DepthFusion(loaded.core, backend or loaded.backend)
        self._benchmark = loaded.benchmark
        self._source = loaded.source

    @property
    def backend(self) -> str:
        return self._engine.backend

    @property
    def config(self) -> LoadedConfig:
        """The configuration the engine actually runs with (a fresh copy on every access)."""
        return LoadedConfig(
            core=self._engine.config,
            backend=self.backend,
            benchmark=self._benchmark,
            source=self._source,
        )

    def process(
        self,
        *,
        depth_m: np.ndarray,
        intrinsics: _core.Intrinsics,
        T_world_camera: np.ndarray | None,
        timestamp_s: float,
    ) -> FusionResult:
        """Fuse one frame.

        ``depth_m`` must be a C-contiguous float32 (H, W) array in metres; ``T_world_camera`` a
        C-contiguous float64 (4, 4) camera-to-world transform, or ``None`` when no pose exists.
        """
        raw = self._engine.process(
            depth_m=depth_m,
            intrinsics=intrinsics,
            T_world_camera=T_world_camera,
            timestamp_s=timestamp_s,
        )
        return _to_result(raw)

    def reset(self) -> None:
        self._engine.reset()


def _to_result(raw: dict[str, Any]) -> FusionResult:
    diagnostics = raw["diagnostics"]
    return FusionResult(
        depth_m=raw["depth_m"],
        valid_mask=raw["valid_mask"],
        variance_m2=raw["variance_m2"],
        confidence_score=raw["confidence_score"],
        source_mask=raw["source_mask"],
        history_age=raw["history_age"],
        diagnostics=Diagnostics(
            frame_index=diagnostics["frame_index"],
            reset_reason=diagnostics["reset_reason"],
            temporal_status=diagnostics["temporal_status"],
            spatial_applied=diagnostics["spatial_applied"],
            input=InputStats(**diagnostics["input"]),
            host_process_ms=diagnostics["host_process_ms"],
            notes=tuple(diagnostics["notes"]),
        ),
    )
