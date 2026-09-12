#!/usr/bin/env python3
"""Download one dataset sequence and write its manifest.

Example:
    python scripts/download_dataset.py --dataset icl-nuim --sequence kt0 \
        --variants clean noisy poses --output data/icl

Only the selected sequence is fetched. Interrupted downloads resume; archives are
extracted with path-traversal and link checks. Nothing is fetched from mirrors.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from cudepthfusion.data.download import DownloadError
from cudepthfusion.data.fetch import fetch_sequence
from cudepthfusion.data.registry import SEQUENCES, VARIANTS, get_sequence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=sorted(SEQUENCES))
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        spec = get_sequence(args.dataset, args.sequence)
        command = shlex.join([Path(sys.argv[0]).as_posix(), *(argv or sys.argv[1:])])
        manifest = fetch_sequence(spec, args.variants, args.output, command=command)
    except (KeyError, ValueError, DownloadError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
