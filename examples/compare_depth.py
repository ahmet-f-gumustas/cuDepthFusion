#!/usr/bin/env python3
"""Side-by-side depth comparison: raw input, spatial filter, fusion, ground truth and error.

    python examples/compare_depth.py --manifest data/icl/kt0/manifest.json \
        --config configs/icl.yaml --backend cpu --frames 100 --output runs/demo/kt0
    python examples/compare_depth.py --manifest data/icl/kt0/manifest.json \
        --config configs/icl.yaml --backend cuda --save-video runs/demo/kt0/demo.mp4

Every depth panel uses the same metre range and colormap and every error panel the same
millimetre range, fixed for the whole run and recorded in ``summary.json``. The status line
keeps process latency and demo throughput apart: the second includes decoding, rendering and
writing the video, so it is not a camera frame rate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from cudepthfusion.config import BACKENDS, load_config
from cudepthfusion.viz.demo import (
    SIXTH_PANELS,
    DemoError,
    DemoOptions,
    run_demo,
    write_demo_run,
)

EXIT_OK = 0
EXIT_USAGE_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare_depth.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, required=True, help="a validated sequence")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--backend", choices=BACKENDS, default="cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    parser.add_argument("--frames", type=int, default=None, help="limit the number of frames")
    parser.add_argument("--save-video", type=Path, default=None, help="write an MP4 here")
    parser.add_argument("--save-frames", type=Path, default=None, help="write PNG frames here")
    parser.add_argument("--fps", type=float, default=15.0, help="playback rate of the MP4")
    parser.add_argument("--panel-width", type=int, default=420)
    parser.add_argument(
        "--depth-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=None,
        help="fixed depth range for every panel; taken from the first frame when omitted",
    )
    parser.add_argument("--error-max-mm", type=float, default=100.0)
    parser.add_argument("--sixth-panel", choices=SIXTH_PANELS, default="source")
    parser.add_argument(
        "--interactive", action="store_true", help="open a window; space pauses, q quits"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    options = DemoOptions(
        frames=args.frames,
        panel_width=args.panel_width,
        depth_range=tuple(args.depth_range) if args.depth_range else None,
        error_max_mm=args.error_max_mm,
        sixth_panel=args.sixth_panel,
        fps=args.fps,
        save_video=args.save_video,
        save_frames=args.save_frames,
        interactive=args.interactive,
    )
    try:
        summary, rows = run_demo(
            args.manifest,
            config,
            backend=args.backend,
            options=options,
            log=lambda message: print(message, file=sys.stderr),
        )
    except DemoError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    write_demo_run(args.output, summary, rows, config)
    print(
        json.dumps(
            {
                "sequence": summary["sequence"],
                "frames_rendered": summary["frames_rendered"],
                "scales": summary["scales"],
                "process_latency_b4_ms": summary["latency"]["process_b4"],
                "demo_throughput_ms": summary["latency"]["demo_throughput"],
                "outputs": summary["outputs"],
                "run": str(args.output),
            },
            indent=2,
        )
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
