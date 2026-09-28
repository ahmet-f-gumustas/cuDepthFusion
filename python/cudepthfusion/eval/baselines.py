"""The methods the protocol compares (spec 10.1).

| Name | Method | What it isolates |
|---|---|---|
| B0 | raw noisy depth, sanitised only | the starting point |
| B1 | mask-aware bilateral filter only | the spatial gain alone |
| B2 | per-pixel EMA without motion compensation | why motion compensation matters |
| B3 | reprojection + gate + a fixed temporal weight | the adaptive confidence, isolated |
| B4 | full cuDepthFusion | the proposed method |

B3 and B4 run through the same engine and therefore share the reprojection and visibility
code; only the merge weights differ. A baseline only ever receives an ``InputFrame``.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

import cudepthfusion as cdf
from cudepthfusion.config import LoadedConfig
from cudepthfusion.data.frames import InputFrame
from cudepthfusion.eval.metrics import sanitize

BASELINE_NAMES = ("B0", "B1", "B2", "B3", "B4")
DEFAULT_EMA_PRIOR_WEIGHT = 0.5
DEFAULT_FIXED_PRIOR_WEIGHT = 0.5


class Baseline(Protocol):
    name: str
    description: str

    def reset(self) -> None: ...

    def process(self, frame: InputFrame) -> tuple[np.ndarray, np.ndarray]: ...


class RawBaseline:
    """B0: the input itself, with the engine's validity rule applied so masks compare."""

    name = "B0"
    description = "raw noisy depth, sanitised only"

    def __init__(self, min_m: float, max_m: float) -> None:
        self._min_m = min_m
        self._max_m = max_m

    def reset(self) -> None:
        return None

    def process(self, frame: InputFrame) -> tuple[np.ndarray, np.ndarray]:
        valid = sanitize(frame.depth_m, self._min_m, self._max_m)
        return np.where(valid, frame.depth_m, 0.0).astype(np.float32), valid


class EmaBaseline:
    """B2: exponential moving average at a fixed pixel, ignoring camera motion."""

    name = "B2"
    description = "per-pixel EMA, no motion compensation"

    def __init__(
        self, min_m: float, max_m: float, prior_weight: float = DEFAULT_EMA_PRIOR_WEIGHT
    ) -> None:
        self._min_m = min_m
        self._max_m = max_m
        self._prior_weight = prior_weight
        self._depth: np.ndarray | None = None
        self._valid: np.ndarray | None = None

    def reset(self) -> None:
        self._depth = None
        self._valid = None

    def process(self, frame: InputFrame) -> tuple[np.ndarray, np.ndarray]:
        valid = sanitize(frame.depth_m, self._min_m, self._max_m)
        current = np.where(valid, frame.depth_m, 0.0).astype(np.float32)
        if self._depth is None or self._depth.shape != current.shape:
            output = current
        else:
            both = valid & self._valid
            output = current.copy()
            output[both] = (1.0 - self._prior_weight) * current[
                both
            ] + self._prior_weight * self._depth[both]
        # Like the main mode of the engine: a pixel without a current measurement stays invalid.
        self._depth = output
        self._valid = valid
        return output, valid


class EngineBaseline:
    """B1, B3 and B4: the engine itself, with the knobs that define each method."""

    def __init__(
        self,
        name: str,
        description: str,
        config: LoadedConfig,
        backend: str,
        *,
        use_pose: bool = True,
    ) -> None:
        self.name = name
        self.description = description
        self._engine = cdf.DepthFusion(config, backend=backend)
        self._use_pose = use_pose

    def reset(self) -> None:
        self._engine.reset()

    def process(self, frame: InputFrame) -> tuple[np.ndarray, np.ndarray]:
        result = self._engine.process(
            depth_m=frame.depth_m,
            intrinsics=frame.intrinsics,
            T_world_camera=frame.T_world_camera if self._use_pose else None,
            timestamp_s=frame.timestamp_s,
        )
        return result.depth_m, result.valid_mask == 1


def _with_fusion(config: LoadedConfig, **overrides: float) -> LoadedConfig:
    data = config.to_dict()
    data["fusion"].update(overrides)
    return cdf.load_config(data)


def make_baselines(
    config: LoadedConfig,
    backend: str,
    names: tuple[str, ...] = BASELINE_NAMES,
    *,
    ema_prior_weight: float = DEFAULT_EMA_PRIOR_WEIGHT,
    fixed_prior_weight: float = DEFAULT_FIXED_PRIOR_WEIGHT,
) -> list[Baseline]:
    minimum = config.core.depth.min_m
    maximum = config.core.depth.max_m
    builders = {
        "B0": lambda: RawBaseline(minimum, maximum),
        "B1": lambda: EngineBaseline(
            "B1",
            "mask-aware bilateral filter only (no pose, so the temporal stage stays off)",
            config,
            backend,
            use_pose=False,
        ),
        "B2": lambda: EmaBaseline(minimum, maximum, ema_prior_weight),
        "B3": lambda: EngineBaseline(
            "B3",
            f"reprojection + gate + fixed temporal weight {fixed_prior_weight}",
            _with_fusion(config, fixed_prior_weight=fixed_prior_weight),
            backend,
        ),
        "B4": lambda: EngineBaseline("B4", "full cuDepthFusion", config, backend),
    }
    unknown = [name for name in names if name not in builders]
    if unknown:
        raise ValueError(f"unknown baseline(s) {unknown}; known: {list(builders)}")
    return [builders[name]() for name in names]
