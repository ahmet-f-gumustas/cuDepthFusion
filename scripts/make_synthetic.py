#!/usr/bin/env python3
"""Generate the analytic synthetic scenarios and write them to disk.

Example:
    python scripts/make_synthetic.py --suite all --output data/synthetic

No network, camera or third-party asset is needed. Each scenario is written in the same
TUM-PNG layout as ICL-NUIM (plus oracle masks), with a manifest that records every
generator parameter, so the exact ground truth can be regenerated.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from cudepthfusion.data.camera import PinholeCamera
from cudepthfusion.synthetic import NOISELESS, NoiseModel, SyntheticSequence, scenario_names
from cudepthfusion.synthetic.export import export_sequence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--suite",
        nargs="+",
        default=["all"],
        help="'all' or scenario names: " + ", ".join(scenario_names()),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=None, help="default: each scenario's own")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0, help="resolution relative to 640x480")
    parser.add_argument("--noiseless", action="store_true", help="noisy depth equals clean depth")
    args = parser.parse_args(argv)

    names = list(scenario_names()) if "all" in args.suite else args.suite
    unknown = sorted(set(names) - set(scenario_names()))
    if unknown:
        print(
            f"error: unknown scenario(s) {unknown}; known: {list(scenario_names())}",
            file=sys.stderr,
        )
        return 2
    camera = PinholeCamera.vga() if args.scale == 1.0 else PinholeCamera.vga().scaled(args.scale)
    noise = NOISELESS if args.noiseless else NoiseModel()
    command = shlex.join([Path(sys.argv[0]).as_posix(), *(argv or sys.argv[1:])])

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    try:
        for name in names:
            sequence = SyntheticSequence(
                name, camera=camera, noise=noise, seed=args.seed, frames=args.frames
            )
            print(export_sequence(sequence, args.output, command=command, log=log))
    except (ValueError, OSError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
