from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import cudepthfusion as cdf

REPO_ROOT = Path(__file__).resolve().parents[2]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """GPU tests are SKIPPED, never PASSED, without a CUDA build and device."""
    info = cdf.build_info()
    if not info["cuda_compiled"]:
        reason = "built without CUDA support (CUDEPTHFUSION_ENABLE_CUDA=OFF)"
    elif not info["cuda_devices"]:
        reason = f"no CUDA device: {info['cuda_error']}"
    else:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def default_config_path() -> Path:
    return REPO_ROOT / "configs" / "default.yaml"


@pytest.fixture
def intrinsics() -> cdf.Intrinsics:
    return cdf.Intrinsics(fx=481.2, fy=480.0, cx=319.5, cy=239.5)


@pytest.fixture
def identity_pose() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


@pytest.fixture
def cpu_engine() -> cdf.DepthFusion:
    return cdf.DepthFusion(backend="cpu")
