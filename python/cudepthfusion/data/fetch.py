"""Download one sequence, extract it safely and write its manifest."""

from __future__ import annotations

import datetime as dt
import shutil
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cudepthfusion.data import download as dl
from cudepthfusion.data.icl_nuim import index_depth_frames, pair_frames, pose_table
from cudepthfusion.data.manifest import (
    ADAPTER_VERSION,
    MANIFEST_NAME,
    MANIFEST_SCHEMA,
    write_json,
)
from cudepthfusion.data.poses import parse_tum_trajectory
from cudepthfusion.data.registry import VARIANTS, SequenceSpec

EXTRACTED_MARKER = ".extracted_from_sha256"
ARCHIVE_DIR = "archives"
POSES_DIR = "poses"

Downloader = Callable[..., dl.DownloadResult]


def fetch_sequence(
    spec: SequenceSpec,
    variants: Sequence[str],
    output_root: Path,
    *,
    command: str,
    downloader: Downloader = dl.download,
    log: dl.Logger = dl.log_to_stderr,
) -> Path:
    """Fetch ``variants`` of ``spec`` into ``output_root/<sequence>``; returns the manifest path.

    A manifest needs clean, noisy and poses together, because frame pairing is part of it.
    """
    missing = sorted(set(VARIANTS) - set(variants))
    if missing:
        raise ValueError(f"a manifest needs all of {list(VARIANTS)}; missing {missing}")
    if spec.conventions is None:
        raise ValueError(f"{spec.dataset}/{spec.name} has no declared conventions")

    sequence_dir = output_root / spec.name
    files: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        remote = spec.files[variant]
        target_dir = sequence_dir / (ARCHIVE_DIR if remote.is_archive else POSES_DIR)
        result = downloader(
            remote.url,
            target_dir / remote.filename,
            official_page=spec.official_page,
            reserve_for_extraction=remote.is_archive,
            log=log,
        )
        entry: dict[str, Any] = {
            "url": remote.url,
            "path": result.path.relative_to(sequence_dir).as_posix(),
            "size_bytes": result.size_bytes,
            "sha256": result.sha256,
            "publisher_sha256": None,
            "hash_note": "computed locally after download; the publisher provides no checksum",
            "retrieved_at": _utc_now() if not result.reused_existing else None,
            "reused_existing_file": result.reused_existing,
        }
        if remote.is_archive:
            entry["extracted_dir"] = variant
            entry["top_level_entries"] = _extract_once(result, sequence_dir / variant, log)
        files[variant] = entry

    conventions = spec.conventions
    indices = {name: index_depth_frames(sequence_dir / name) for name in ("clean", "noisy")}
    records = parse_tum_trajectory(sequence_dir / files["poses"]["path"])
    poses = pose_table(records, conventions.pose_id_offset)
    pairing = pair_frames(indices, set(poses))

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "dataset": spec.dataset,
        "distribution": "TUM RGB-D compatible PNGs (clean and with noise) + TUM-format poses",
        "sequence": spec.name,
        "scene": spec.scene,
        "split": spec.split,
        "official_page": spec.official_page,
        "license": {"name": spec.license, "url": spec.license_url, "citation": spec.citation},
        "files": files,
        "conventions": asdict(conventions),
        "conventions_status": "declared; confirmed only by a passing validation.json",
        "camera_model": "pinhole, no distortion",
        "units": "depth_m = raw / depth_units_per_meter (metres); raw 0 = missing",
        "timestamp_rule": (
            f"timestamp_s = image_id / {conventions.frame_rate_hz:g} Hz; the pose file's first "
            "column is an integer frame id, not seconds"
        ),
        "frames": {
            "frames_listed_on_official_page": spec.frames_listed_on_page,
            "matched_ids": list(pairing.matched),
            "pairing": pairing.to_dict(),
        },
        "command": command,
        "adapter_version": ADAPTER_VERSION,
        "created_at": _utc_now(),
    }
    manifest_path = sequence_dir / MANIFEST_NAME
    write_json(manifest_path, manifest)
    log(
        f"wrote {manifest_path}: {len(pairing.matched)} matched frames; "
        f"counts {pairing.counts}; pairing ok={pairing.ok}"
    )
    return manifest_path


def _extract_once(result: dl.DownloadResult, target: Path, log: dl.Logger) -> list[str]:
    """Extract unless ``target`` already holds this exact archive's contents."""
    marker = target / EXTRACTED_MARKER
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == result.sha256:
        log(f"reusing extracted {target}")
        return sorted(p.name for p in target.iterdir() if p.name != EXTRACTED_MARKER)
    staging = target.with_name(target.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)  # our own leftover from an interrupted extraction
    log(f"extracting {result.path} -> {target}")
    top_level = dl.safe_extract(result.path, staging)
    (staging / EXTRACTED_MARKER).write_text(result.sha256 + "\n", encoding="utf-8")
    if target.exists():
        shutil.rmtree(target)  # stale extraction of a different archive
    staging.rename(target)
    return top_level


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
