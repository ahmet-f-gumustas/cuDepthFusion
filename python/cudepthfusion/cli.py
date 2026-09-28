"""Command line entry point: ``python -m cudepthfusion.cli <command>``."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cudepthfusion import _core
from cudepthfusion.config import BACKENDS, load_config
from cudepthfusion.engine import DepthFusion
from cudepthfusion.eval.artifacts import environment_info, git_commit, utc_now, write_run
from cudepthfusion.eval.baselines import (
    BASELINE_NAMES,
    DEFAULT_EMA_PRIOR_WEIGHT,
    DEFAULT_FIXED_PRIOR_WEIGHT,
    EngineBaseline,
)
from cudepthfusion.eval.metrics import (
    BAD_PIXEL_ABS_M,
    BAD_PIXEL_REL,
    EDGE_DILATE_PX,
    EDGE_JUMP_M,
    aggregate,
)
from cudepthfusion.eval.runner import evaluate_sequence

EXIT_OK = 0
EXIT_FAILED_CHECK = 1
EXIT_USAGE_ERROR = 2

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

    validate = commands.add_parser(
        "validate-data",
        help="check frame pairing and data conventions of a downloaded sequence "
        "(writes validation.json next to the manifest)",
    )
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--source-frames", type=int, default=8)
    validate.set_defaults(handler=_cmd_validate_data)

    evaluate = commands.add_parser(
        "evaluate", help="run the baselines over a split and write a run folder (spec 10)"
    )
    evaluate.add_argument(
        "--manifest", type=Path, action="append", help="sequence manifest; repeatable"
    )
    evaluate.add_argument("--split", choices=("development", "validation", "test"))
    evaluate.add_argument("--data-root", type=Path, default=Path("data/icl"))
    evaluate.add_argument("--config", type=Path, default=None)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--backend", choices=BACKENDS, default="cpu")
    evaluate.add_argument("--frames", type=int, default=None, help="limit frames per sequence")
    evaluate.add_argument("--baselines", nargs="+", default=list(BASELINE_NAMES))
    evaluate.add_argument("--fixed-prior-weight", type=float, default=DEFAULT_FIXED_PRIOR_WEIGHT)
    evaluate.add_argument("--ema-prior-weight", type=float, default=DEFAULT_EMA_PRIOR_WEIGHT)
    evaluate.add_argument("--stability", action="store_true", help="also track fixed world points")
    evaluate.add_argument("--plots", action="store_true", help="write figures into the run folder")
    evaluate.set_defaults(handler=_cmd_evaluate)

    ablate = commands.add_parser(
        "ablate", help="remove one part of the method at a time and measure the cost (spec 10.7)"
    )
    ablate.add_argument("--manifest", type=Path, action="append")
    ablate.add_argument("--split", choices=("development", "validation", "test"))
    ablate.add_argument("--data-root", type=Path, default=Path("data/icl"))
    ablate.add_argument("--config", type=Path, default=None)
    ablate.add_argument("--output", type=Path, required=True)
    ablate.add_argument("--backend", choices=BACKENDS, default="cpu")
    ablate.add_argument("--frames", type=int, default=None)
    ablate.add_argument("--stability", action="store_true")
    ablate.add_argument("--plots", action="store_true")
    ablate.set_defaults(handler=_cmd_ablate)

    benchmark = commands.add_parser(
        "benchmark",
        help="time the engine on real consecutive frames with warm-up and repeats (spec 10.5)",
    )
    benchmark.add_argument("--manifest", type=Path, default=Path("data/icl/kt0/manifest.json"))
    benchmark.add_argument("--config", type=Path, default=None)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--backend", choices=BACKENDS, default=None)
    benchmark.add_argument("--warmup-frames", type=int, default=None)
    benchmark.add_argument("--measured-frames", type=int, default=None)
    benchmark.add_argument("--repeats", type=int, default=None)
    benchmark.add_argument("--first-frame", type=int, default=0)
    benchmark.add_argument("--label", default=None, help="free text stored in summary.json")
    benchmark.set_defaults(handler=_cmd_benchmark)
    return parser


# Each variant removes exactly one part of the method (spec 10.7). The flag says whether the
# pose reaches the engine: dropping it is how "no pose compensation" is expressed, together
# with assume_static_camera so the temporal stage still runs, on an identity transport.
ABLATIONS: dict[str, tuple[dict, bool]] = {
    "full": ({}, True),
    "no_spatial_filter": ({"spatial": {"enabled": False}}, True),
    "no_pose_compensation": ({"fusion": {"assume_static_camera": True}}, False),
    "no_depth_gate": ({"fusion": {"tau_abs_m": 1000.0}}, True),
    "no_adaptive_weighting": ({"fusion": {"fixed_prior_weight": 0.5}}, True),
    "no_variance_caps": ({"fusion": {"history_decay": 1.0, "max_history_ratio": 1.0e9}}, True),
    "no_gradient_term": ({"fusion": {"q_gradient": 0.0}}, True),
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


def _cmd_validate_data(args: argparse.Namespace) -> int:
    try:
        from cudepthfusion.data.icl_nuim import DatasetError
        from cudepthfusion.data.manifest import ManifestError
        from cudepthfusion.data.poses import TrajectoryFormatError
        from cudepthfusion.data.validation import validate_sequence
    except ImportError as error:  # opencv is an optional dependency
        print(f"error: {error}; install with pip install 'cudepthfusion[data]'", file=sys.stderr)
        return EXIT_USAGE_ERROR

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    try:
        report = validate_sequence(args.manifest, source_frames=args.source_frames, log=log)
    except (DatasetError, ManifestError, TrajectoryFormatError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    geometry = report["geometry"]
    summary = {
        "passed": report["passed"],
        "reasons": report["reasons"],
        "report": str(args.manifest.parent / "validation.json"),
        "pairing": report["pairing"],
        "noisy_clean_consistency": report["noisy_clean_consistency"],
        "geometry": {key: geometry[key] for key in ("declared", "best", "runner_up")},
    }
    print(json.dumps(summary, indent=2))
    return EXIT_OK if report["passed"] else EXIT_FAILED_CHECK


def _resolve_manifests(args: argparse.Namespace) -> list[Path]:
    if args.manifest:
        return list(args.manifest)
    if not args.split:
        raise ValueError("pass --manifest or --split")
    from cudepthfusion.data.registry import SEQUENCES

    manifests = []
    missing = []
    for spec in SEQUENCES["icl-nuim"].values():
        if spec.split != args.split:
            continue
        candidate = args.data_root / spec.name / "manifest.json"
        (manifests if candidate.exists() else missing).append(candidate)
    if not manifests:
        raise ValueError(
            f"no downloaded sequence for split '{args.split}' under {args.data_root}; "
            f"expected one of {[str(path) for path in missing]}"
        )
    return manifests


def _summarise(results: list, names: tuple[str, ...]) -> dict[str, Any]:
    sequences = []
    for result in results:
        sequences.append(
            {
                "sequence": result.sequence,
                "split": result.split,
                "manifest": result.manifest_path,
                "manifest_sha256": result.manifest_sha256,
                "frames_evaluated": result.frames_evaluated,
                "per_baseline": {
                    name: aggregate([m for m in result.metrics if m.baseline == name])
                    for name in names
                },
                "stability": [entry.as_dict() for entry in result.stability],
            }
        )
    pooled = {
        name: aggregate([m for result in results for m in result.metrics if m.baseline == name])
        for name in names
    }
    # A long sequence must not drown the others, so report the equal-weight average too.
    sequence_mean = {
        name: {
            key: float(
                sum(entry["per_baseline"][name][key] for entry in sequences) / len(sequences)
            )
            for key in ("rmse_m", "mae_m", "bad_pixel_rate", "coverage", "edge_mae_m")
        }
        for name in names
    }
    return {"sequences": sequences, "overall": {"pooled": pooled, "sequence_mean": sequence_mean}}


def _finish_run(args: argparse.Namespace, summary: dict[str, Any], results: list, config) -> int:
    rows = [metric.row() for result in results for metric in result.metrics]
    write_run(args.output, summary, rows, config)
    if getattr(args, "plots", False):
        from cudepthfusion.eval.plots import write_plots

        for path in write_plots(args.output, summary, rows):
            print(f"wrote {path}", file=sys.stderr)
    print(
        json.dumps(
            {"output": str(args.output), "overall": summary["overall"]},
            indent=2,
            default=lambda value: None,
        )
    )
    return EXIT_OK


def _cmd_ablate(args: argparse.Namespace) -> int:
    from cudepthfusion.data.icl_nuim import DatasetError
    from cudepthfusion.data.manifest import ManifestError

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    try:
        manifests = _resolve_manifests(args)
        config = load_config(args.config)
        results = []
        for manifest in manifests:
            log(f"ablating on {manifest}")
            methods = []
            for name, (overrides, use_pose) in ABLATIONS.items():
                data = config.to_dict()
                for section, values in overrides.items():
                    data[section].update(values)
                methods.append(
                    EngineBaseline(
                        name,
                        f"ablation: {name}",
                        load_config(data),
                        args.backend,
                        use_pose=use_pose,
                    )
                )
            results.append(
                evaluate_sequence(
                    manifest,
                    config,
                    backend=args.backend,
                    frames=args.frames,
                    log=log,
                    methods=methods,
                    track_stability=args.stability,
                )
            )
    except (DatasetError, ManifestError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    names = tuple(ABLATIONS)
    summary = {
        "created_at": utc_now(),
        "command": shlex.join(["python", "-m", "cudepthfusion.cli", *sys.argv[1:]]),
        "git_commit": git_commit(),
        "backend": args.backend,
        "split": args.split,
        "frames_limit": args.frames,
        "kind": "ablation",
        "variants": {name: str(overrides) for name, (overrides, _) in ABLATIONS.items()},
        "thresholds": {
            "bad_pixel_abs_m": BAD_PIXEL_ABS_M,
            "bad_pixel_rel": BAD_PIXEL_REL,
            "edge_jump_m": EDGE_JUMP_M,
            "edge_dilate_px": EDGE_DILATE_PX,
        },
        **_summarise(results, names),
    }
    return _finish_run(args, summary, results, config)


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from cudepthfusion.data.icl_nuim import DatasetError
    from cudepthfusion.data.manifest import ManifestError
    from cudepthfusion.eval.artifacts import LATENCY_NAME, write_table
    from cudepthfusion.eval.benchmark import (
        BenchmarkError,
        BenchmarkPlan,
        describe_source,
        load_frames,
        run_benchmark,
    )

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    config = load_config(args.config)
    backend = args.backend or config.backend
    defaults = config.benchmark
    plan = BenchmarkPlan(
        warmup_frames=args.warmup_frames
        if args.warmup_frames is not None
        else defaults.warmup_frames,
        measured_frames=args.measured_frames
        if args.measured_frames is not None
        else defaults.measured_frames,
        repeats=args.repeats if args.repeats is not None else defaults.repeats,
        first_frame=args.first_frame,
    )
    try:
        log(f"decoding {plan.frames_needed} frames of {args.manifest} before timing")
        frames = load_frames(args.manifest, plan)
        summary, rows = run_benchmark(frames, config, plan, backend=backend, log=log)
    except (DatasetError, ManifestError, BenchmarkError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    summary = {
        "created_at": utc_now(),
        "command": shlex.join(["python", "-m", "cudepthfusion.cli", *sys.argv[1:]]),
        "git_commit": git_commit(),
        "label": args.label,
        "config_file": str(args.config) if args.config else None,
        "source": describe_source(args.manifest),
        **summary,
    }
    write_run(args.output, summary, [], config)
    write_table(args.output / LATENCY_NAME, rows)
    latency = summary["process_latency"]
    compute = summary["gpu_compute"]
    print(
        json.dumps(
            {
                "backend": backend,
                "process_latency_ms": {k: latency[k] for k in ("median_ms", "p95_ms", "p99_ms")},
                "gpu_compute_ms": compute.get("median_ms"),
                "gpu_shared_with_other_processes": summary["gpu_shared_with_other_processes"],
                "run": str(args.output),
            },
            indent=2,
        )
    )
    return EXIT_OK


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from cudepthfusion.data.icl_nuim import DatasetError
    from cudepthfusion.data.manifest import ManifestError

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    try:
        manifests = _resolve_manifests(args)
        config = load_config(args.config)
        results = []
        for manifest in manifests:
            log(f"evaluating {manifest}")
            results.append(
                evaluate_sequence(
                    manifest,
                    config,
                    backend=args.backend,
                    baselines=tuple(args.baselines),
                    frames=args.frames,
                    log=log,
                    fixed_prior_weight=args.fixed_prior_weight,
                    ema_prior_weight=args.ema_prior_weight,
                    track_stability=args.stability,
                )
            )
    except (DatasetError, ManifestError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    names = tuple(args.baselines)
    summary = {
        "created_at": utc_now(),
        "command": shlex.join(["python", "-m", "cudepthfusion.cli", *sys.argv[1:]]),
        "git_commit": git_commit(),
        "backend": args.backend,
        "split": args.split,
        "frames_limit": args.frames,
        "kind": "evaluation",
        "config_file": str(args.config) if args.config else None,
        "baselines": results[0].baselines if results else {},
        "thresholds": {
            "bad_pixel_abs_m": BAD_PIXEL_ABS_M,
            "bad_pixel_rel": BAD_PIXEL_REL,
            "edge_jump_m": EDGE_JUMP_M,
            "edge_dilate_px": EDGE_DILATE_PX,
            "fixed_prior_weight": args.fixed_prior_weight,
            "ema_prior_weight": args.ema_prior_weight,
        },
        **_summarise(results, names),
    }
    return _finish_run(args, summary, results, config)


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
