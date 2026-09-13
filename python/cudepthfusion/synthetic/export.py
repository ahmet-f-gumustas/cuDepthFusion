"""Write synthetic sequences in the TUM-PNG layout that the dataset adapter reads.

Per sequence (the same layout as an extracted ICL-NUIM sequence, so one code path loads
both):

    <name>/clean/depth/<id>.png    uint16, 5000 units per metre, 0 = no surface
    <name>/noisy/depth/<id>.png    uint16, the filter input
    <name>/poses/<name>.freiburg   "id tx ty tz qx qy qz qw" (id = image id, T_world_camera)
    <name>/oracle.npz              surface_id (uint16) and dynamic_mask (bool) per frame
    <name>/manifest.json           the dataset manifest schema plus a generator block

Exact float ground truth is not stored; the manifest's generator block regenerates it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from cudepthfusion._core import library_version
from cudepthfusion.data.icl_nuim import index_depth_frames, pair_frames
from cudepthfusion.data.manifest import MANIFEST_NAME, MANIFEST_SCHEMA, file_sha256, write_json
from cudepthfusion.data.poses import rotation_to_quaternion
from cudepthfusion.data.registry import Conventions
from cudepthfusion.synthetic.suite import SyntheticSequence

DEPTH_UNITS_PER_M = 5000.0
MAX_RAW = np.iinfo(np.uint16).max  # 13.107 m at 5000 units per metre
EXPORT_VERSION = "synthetic-export/1"
ORACLE_NAME = "oracle.npz"
DATASET_NAME = "synthetic"

Logger = Callable[[str], None]


def to_raw(depth_m: np.ndarray) -> tuple[np.ndarray, int]:
    """Metres -> uint16 at 5000 units/m. Depth that cannot be stored becomes 0 and is counted."""
    raw = np.round(depth_m.astype(np.float64) * DEPTH_UNITS_PER_M)
    too_far = raw > MAX_RAW
    raw[too_far] = 0
    return raw.astype(np.uint16), int(too_far.sum())


def tree_sha256(directory: Path) -> str:
    """SHA-256 over the sorted (relative path, file SHA-256) pairs of a directory tree."""
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(f"{path.relative_to(directory).as_posix()} {file_sha256(path)}\n".encode())
    return digest.hexdigest()


def export_sequence(
    sequence: SyntheticSequence,
    output_root: Path,
    *,
    command: str,
    log: Logger = lambda _: None,
) -> Path:
    """Write ``sequence`` to ``output_root/<name>`` and return its manifest path."""
    import cv2  # optional dependency: pip install 'cudepthfusion[data]'

    target = output_root / sequence.name
    _refuse_foreign_directory(target)
    staging = target.with_name(target.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)  # our own leftover from an interrupted export
    for sub in ("clean/depth", "noisy/depth", "poses"):
        (staging / sub).mkdir(parents=True)

    surface_ids, dynamic_masks, pose_lines = [], [], []
    unstorable = 0
    for frame in sequence:
        frame_id = frame.input.frame_id
        clean_raw, clean_lost = to_raw(frame.truth.depth_m)
        noisy_raw, noisy_lost = to_raw(frame.input.depth_m)
        unstorable += clean_lost + noisy_lost
        for variant, raw in (("clean", clean_raw), ("noisy", noisy_raw)):
            if not cv2.imwrite(str(staging / variant / "depth" / f"{frame_id}.png"), raw):
                raise OSError(f"could not write {variant} frame {frame_id} under {staging}")
        surface_ids.append(frame.truth.surface_id)
        dynamic_masks.append(frame.truth.dynamic_mask)
        pose = frame.input.T_world_camera
        values = (*pose[:3, 3], *rotation_to_quaternion(pose[:3, :3]))
        pose_lines.append(" ".join([str(frame_id), *(f"{value:.17g}" for value in values)]))

    poses_path = f"poses/{sequence.name}.freiburg"
    (staging / poses_path).write_text("\n".join(pose_lines) + "\n", encoding="utf-8")
    np.savez_compressed(
        staging / ORACLE_NAME,
        surface_id=np.stack(surface_ids),
        dynamic_mask=np.stack(dynamic_masks),
    )
    manifest = _manifest(sequence, staging, poses_path, unstorable, command)
    write_json(staging / MANIFEST_NAME, manifest)

    if target.exists():
        shutil.rmtree(target)  # a previous synthetic export of the same scenario
    staging.rename(target)
    log(f"wrote {target / MANIFEST_NAME}: {len(sequence)} frames of '{sequence.name}'")
    return target / MANIFEST_NAME


def export_suite(
    sequences: Sequence[SyntheticSequence],
    output_root: Path,
    *,
    command: str,
    log: Logger = lambda _: None,
) -> list[Path]:
    return [export_sequence(s, output_root, command=command, log=log) for s in sequences]


def _manifest(
    sequence: SyntheticSequence, root: Path, poses_path: str, unstorable: int, command: str
) -> dict[str, Any]:
    camera = sequence.camera
    conventions = Conventions(
        depth_kind="z",
        depth_units_per_meter=DEPTH_UNITS_PER_M,
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
        pose_direction="T_world_camera",
        pose_id_offset=0,
        frame_rate_hz=sequence.frame_rate_hz,
    )
    pairing = pair_frames(
        {"clean": index_depth_frames(root / "clean"), "noisy": index_depth_frames(root / "noisy")},
        set(sequence.frame_ids),
    )
    tree_note = "SHA-256 over the sorted per-file hashes of the directory"
    scenario = sequence.scenario
    return {
        "schema": MANIFEST_SCHEMA,
        "dataset": DATASET_NAME,
        "distribution": "cuDepthFusion analytic scene generator, TUM-PNG layout",
        "sequence": sequence.name,
        "scene": scenario.name,
        "split": "synthetic",
        "license": {
            "name": "Apache-2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0",
            "citation": "generated by cuDepthFusion's analytic scene generator",
        },
        "files": {
            "clean": {
                "extracted_dir": "clean",
                "sha256": tree_sha256(root / "clean"),
                "hash_note": tree_note,
            },
            "noisy": {
                "extracted_dir": "noisy",
                "sha256": tree_sha256(root / "noisy"),
                "hash_note": tree_note,
            },
            "poses": {"path": poses_path, "sha256": file_sha256(root / poses_path)},
            "oracle": {"path": ORACLE_NAME, "sha256": file_sha256(root / ORACLE_NAME)},
        },
        "conventions": asdict(conventions),
        "conventions_status": (
            "by construction (generator); validate-data can confirm it only for "
            "moving-camera scenarios"
        ),
        "camera_model": "pinhole, no distortion",
        "units": "depth_m = raw / depth_units_per_meter (metres); raw 0 = missing",
        "timestamp_rule": (
            f"timestamp_s = image_id / {sequence.frame_rate_hz:g} Hz; the pose file's first "
            "column is the integer image id"
        ),
        "frames": {"matched_ids": list(pairing.matched), "pairing": pairing.to_dict()},
        "unstorable_pixels": unstorable,
        "generator": {
            "scenario": scenario.name,
            "purpose": scenario.purpose,
            "motion": scenario.motion,
            "dynamic_surface_ids": sorted(scenario.scene.dynamic_surface_ids),
            "seed": sequence.seed,
            "frames": len(sequence),
            "frame_rate_hz": sequence.frame_rate_hz,
            "camera": asdict(camera),
            "noise": asdict(sequence.noise),
            "library_version": library_version(),
        },
        "command": command,
        "adapter_version": EXPORT_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    }


def _refuse_foreign_directory(target: Path) -> None:
    """Only ever replace a directory that holds a previous synthetic export."""
    if not target.exists():
        return
    manifest = target / MANIFEST_NAME
    try:
        is_synthetic = (
            json.loads(manifest.read_text(encoding="utf-8")).get("dataset") == DATASET_NAME
        )
    except (OSError, json.JSONDecodeError):
        is_synthetic = False
    if not is_synthetic:
        raise FileExistsError(
            f"{target} exists and is not a synthetic export; refusing to replace it"
        )
