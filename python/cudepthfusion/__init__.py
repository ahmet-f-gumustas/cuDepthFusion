"""cuDepthFusion: pose-aware, edge-preserving temporal depth fusion."""

from cudepthfusion._core import (
    BackendUnavailableError,
    ConfigError,
    Intrinsics,
    InvalidInputError,
    build_info,
    cuda_compiled,
    library_version,
)
from cudepthfusion.config import BenchmarkConfig, LoadedConfig, load_config
from cudepthfusion.engine import DepthFusion, Diagnostics, FusionResult, InputStats

__version__ = library_version()

__all__ = [
    "BackendUnavailableError",
    "BenchmarkConfig",
    "ConfigError",
    "DepthFusion",
    "Diagnostics",
    "FusionResult",
    "InputStats",
    "Intrinsics",
    "InvalidInputError",
    "LoadedConfig",
    "__version__",
    "build_info",
    "cuda_compiled",
    "load_config",
]
