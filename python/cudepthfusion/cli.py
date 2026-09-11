"""Command line entry point: ``python -m cudepthfusion.cli <command>``."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cudepthfusion import _core
from cudepthfusion.config import BACKENDS, load_config
from cudepthfusion.engine import DepthFusion

EXIT_OK = 0
EXIT_FAILED_CHECK = 1
EXIT_USAGE_ERROR = 2

JETSON_RELEASE_FILE = Path("/etc/nv_tegra_release")
SMOKE_INTRINSICS = _core.Intrinsics(fx=481.2, fy=480.0, cx=319.5, cy=239.5)
SMOKE_FRAME_PERIOD_S = 1.0 / 30.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (_core.ConfigError, _core.InvalidInputError, _core.BackendUnavailableError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE_ERROR


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cudepthfusion", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    info = commands.add_parser("info", help="print build, CUDA and platform information as JSON")
    info.set_defaults(handler=_cmd_info)

    check = commands.add_parser("check-config", help="validate a YAML config and print it resolved")
    check.add_argument("path", type=Path)
    check.set_defaults(handler=_cmd_check_config)

    smoke = commands.add_parser(
        "smoke", help="run synthetic frames through the engine and check output invariants"
    )
    smoke.add_argument("--backend", choices=BACKENDS, default="cpu")
    smoke.add_argument("--config", type=Path, default=None)
    smoke.add_argument("--width", type=int, default=640)
    smoke.add_argument("--height", type=int, default=480)
    smoke.add_argument("--frames", type=int, default=5)
    smoke.add_argument("--seed", type=int, default=0)
    smoke.set_defaults(handler=_cmd_smoke)
    return parser


def environment_info() -> dict[str, Any]:
    jetson = (
        JETSON_RELEASE_FILE.read_text(encoding="utf-8").strip()
        if JETSON_RELEASE_FILE.exists()
        else None
    )
    return {
        "cudepthfusion": _core.build_info(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "jetson_l4t_release": jetson,
    }


def _cmd_info(_: argparse.Namespace) -> int:
    print(json.dumps(environment_info(), indent=2))
    return EXIT_OK


def _cmd_check_config(args: argparse.Namespace) -> int:
    loaded = load_config(args.path)
    print(yaml.safe_dump(loaded.to_dict(), sort_keys=False), end="")
    return EXIT_OK


def _cmd_smoke(args: argparse.Namespace) -> int:
    """Pipeline plumbing check on a noisy synthetic plane. Not a benchmark."""
    if args.width < 1 or args.height < 1 or args.frames < 1:
        print("error: --width, --height and --frames must be >= 1", file=sys.stderr)
        return EXIT_USAGE_ERROR

    engine = DepthFusion(args.config, backend=args.backend)
    rng = np.random.default_rng(args.seed)
    pose = np.eye(4, dtype=np.float64)
    failures: list[str] = []
    frames: list[dict[str, Any]] = []
    for index in range(args.frames):
        depth = (2.0 + rng.normal(0.0, 0.01, (args.height, args.width))).astype(np.float32)
        depth[0, :] = 0.0  # a row of sensor dropouts
        result = engine.process(
            depth_m=depth,
            intrinsics=SMOKE_INTRINSICS,
            T_world_camera=pose,
            timestamp_s=index * SMOKE_FRAME_PERIOD_S,
        )
        failures.extend(_smoke_invariant_failures(index, depth, result))
        frames.append(
            {
                "frame_index": result.diagnostics.frame_index,
                "reset_reason": result.diagnostics.reset_reason,
                "temporal_status": result.diagnostics.temporal_status,
                "num_valid": result.diagnostics.input.num_valid,
            }
        )

    summary = {
        "backend": engine.backend,
        "shape": [args.height, args.width],
        "frames": frames,
        "notes": list(result.diagnostics.notes),
        "failures": failures,
        "passed": not failures,
    }
    print(json.dumps(summary, indent=2))
    return EXIT_OK if not failures else EXIT_FAILED_CHECK


def _smoke_invariant_failures(index: int, depth: np.ndarray, result: Any) -> list[str]:
    failures: list[str] = []
    expected_dtypes = {
        "depth_m": np.float32,
        "valid_mask": np.uint8,
        "variance_m2": np.float32,
        "confidence_score": np.float32,
        "source_mask": np.uint8,
        "history_age": np.uint16,
    }
    for name, dtype in expected_dtypes.items():
        array = getattr(result, name)
        if array.shape != depth.shape or array.dtype != dtype:
            failures.append(f"frame {index}: {name} has {array.dtype}{array.shape}")
    expected_valid = int(np.count_nonzero(depth))
    actual_valid = int(np.count_nonzero(result.valid_mask))
    if actual_valid != expected_valid:
        failures.append(f"frame {index}: {actual_valid} valid pixels, expected {expected_valid}")
    if np.any(result.depth_m[result.valid_mask == 0] != 0):
        failures.append(f"frame {index}: invalid pixels carry non-zero depth")
    return failures


if __name__ == "__main__":
    sys.exit(main())
