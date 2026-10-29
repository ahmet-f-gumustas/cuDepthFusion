#!/usr/bin/env python3
"""Reproduce the published results, the demo and the benchmark with one command.

Example:
    python scripts/reproduce.py --backend cuda --output runs/reproduce

Each step is a documented command, run as its own process, in this order: download any
missing ICL-NUIM sequence (kt0-kt3), validate-data on every sequence, evaluate the
validation split, ablate on it, evaluate the test split once with the frozen configuration,
render the kt0 demo and run the benchmark. Every step writes its own run folder under
--output, and reproduce.json records each command, its exit code and its duration. The
first step that fails stops the run.

--frames limits every step to a few frames. That checks that the pipeline runs end to end;
it does not reproduce the published numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cudepthfusion import _core
from cudepthfusion.config import BACKENDS
from cudepthfusion.data.registry import SEQUENCES, VARIANTS
from cudepthfusion.eval.artifacts import git_commit, utc_now

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = "icl-nuim"
DEMO_SEQUENCE = "kt0"
DEMO_FRAMES = 200
REPORT_NAME = "reproduce.json"


@dataclass(frozen=True)
class Step:
    name: str
    argv: tuple[str, ...]  # everything after the interpreter

    @property
    def command(self) -> str:
        return shlex.join(["python", *self.argv])


def _shown(path: Path) -> str:
    """A repository file as the user would type it: relative when below the working directory."""
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return os.fspath(path)


def plan_steps(data_root: Path, output: Path, backend: str, frames: int | None) -> list[Step]:
    cli = ("-m", "cudepthfusion.cli")
    icl_config = _shown(REPO_ROOT / "configs" / "icl.yaml")
    limit = ("--frames", str(frames)) if frames is not None else ()
    steps = []
    for name in SEQUENCES[DATASET]:
        if not (data_root / name / "manifest.json").exists():
            steps.append(
                Step(
                    f"download {name}",
                    (
                        _shown(REPO_ROOT / "scripts" / "download_dataset.py"),
                        *("--dataset", DATASET, "--sequence", name),
                        *("--variants", *VARIANTS, "--output", os.fspath(data_root)),
                    ),
                )
            )
    for name in SEQUENCES[DATASET]:
        manifest = os.fspath(data_root / name / "manifest.json")
        steps.append(Step(f"validate {name}", (*cli, "validate-data", "--manifest", manifest)))

    def split_run(command: str, split: str, folder: str, *extra: str) -> Step:
        return Step(
            f"{command} {split}",
            (
                *cli,
                command,
                *("--split", split, "--data-root", os.fspath(data_root)),
                *("--config", icl_config, "--output", os.fspath(output / folder)),
                *("--backend", backend, *limit, *extra),
            ),
        )

    # The order of docs/RESULTS.md: parameters were chosen on the validation split, and the
    # test split is run once, with the frozen configuration.
    steps.append(split_run("evaluate", "validation", "validation", "--stability", "--plots"))
    steps.append(split_run("ablate", "validation", "ablation", "--plots"))
    steps.append(split_run("evaluate", "test", "test", "--stability", "--plots"))

    demo_output = output / "demo" / DEMO_SEQUENCE
    steps.append(
        Step(
            f"demo {DEMO_SEQUENCE}",
            (
                _shown(REPO_ROOT / "examples" / "compare_depth.py"),
                *("--manifest", os.fspath(data_root / DEMO_SEQUENCE / "manifest.json")),
                *("--config", icl_config, "--backend", backend),
                *("--frames", str(frames if frames is not None else DEMO_FRAMES)),
                *("--sixth-panel", "error_b1", "--panel-width", "320"),
                *("--output", os.fspath(demo_output)),
                *("--save-video", os.fspath(demo_output / "demo.mp4")),
            ),
        )
    )

    quick = ()
    if frames is not None:
        quick = ("--warmup-frames", str(frames), "--measured-frames", str(frames), "--repeats", "1")
    steps.append(
        Step(
            "benchmark",
            (
                *cli,
                "benchmark",
                *("--manifest", os.fspath(data_root / DEMO_SEQUENCE / "manifest.json")),
                *("--config", _shown(REPO_ROOT / "configs" / "benchmark.yaml")),
                *("--backend", backend, *quick),
                *("--output", os.fspath(output / "performance")),
            ),
        )
    )
    return steps


def _cuda_unavailable() -> str | None:
    info = _core.build_info()
    if not info["cuda_compiled"]:
        return "this build has no CUDA support; pass --backend cpu or reinstall with CUDA"
    if not info["cuda_devices"]:
        return f"no CUDA device: {info['cuda_error']}; pass --backend cpu"
    return None


def _write_report(output: Path, report: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2) + "\n"
    (output / REPORT_NAME).write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path("data/icl"))
    parser.add_argument("--output", type=Path, default=Path("runs/reproduce"))
    parser.add_argument("--backend", choices=BACKENDS, default="cuda")
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="limit every step to this many frames (a pipeline check, not the published numbers)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    args = parser.parse_args(argv)
    if args.frames is not None and args.frames < 1:
        print("error: --frames must be >= 1", file=sys.stderr)
        return 2

    steps = plan_steps(args.data_root, args.output, args.backend, args.frames)
    if args.dry_run:
        for step in steps:
            print(step.command)
        return 0
    if args.backend == "cuda" and (reason := _cuda_unavailable()):
        print(f"error: {reason}", file=sys.stderr)
        return 2

    report: dict[str, Any] = {
        "created_at": utc_now(),
        "command": shlex.join(["python", Path(sys.argv[0]).as_posix(), *(argv or sys.argv[1:])]),
        "git_commit": git_commit(),
        "backend": args.backend,
        "frames_limit": args.frames,
        "passed": None,
        "steps": [],
    }
    for index, step in enumerate(steps, start=1):
        print(f"[{index}/{len(steps)}] {step.name}: {step.command}", file=sys.stderr, flush=True)
        start = time.perf_counter()
        code = subprocess.run([sys.executable, *step.argv], check=False).returncode
        report["steps"].append(
            {
                "name": step.name,
                "command": step.command,
                "exit_code": code,
                "seconds": round(time.perf_counter() - start, 1),
            }
        )
        if code != 0:
            report["passed"] = False
            _write_report(args.output, report)
            print(f"error: step '{step.name}' failed with exit code {code}", file=sys.stderr)
            return code
        _write_report(args.output, report)
    report["passed"] = True
    _write_report(args.output, report)
    print(args.output / REPORT_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
