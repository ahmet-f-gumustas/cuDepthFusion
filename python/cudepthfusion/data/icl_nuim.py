"""ICL-NUIM adapter for the TUM RGB-D compatible PNG distribution.

Archive layout (checked when the manifest is built, not assumed afterwards): each variant
extracts to ``depth/<id>.png`` and ``rgb/<id>.png`` with integer frame ids. The pose file's
first column is an integer frame id, not seconds. Clean depth is only reachable through
``IclGroundTruth``; ``IclInputSequence`` never touches it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cudepthfusion import _core
from cudepthfusion.data.camera import ray_norms
from cudepthfusion.data.frames import InputFrame
from cudepthfusion.data.manifest import MANIFEST_NAME, VALIDATION_NAME, file_sha256, load_manifest
from cudepthfusion.data.poses import PoseRecord, invert_rigid, parse_tum_trajectory
from cudepthfusion.data.registry import Conventions

DEPTH_DIR = "depth"
DEPTH_SUFFIX = ".png"


class DatasetError(ValueError):
    """The dataset on disk does not match its manifest or the adapter's expectations."""


@dataclass(frozen=True)
class FrameIndex:
    paths: Mapping[int, Path]
    duplicates: tuple[int, ...] = ()
    unparsable: tuple[str, ...] = ()

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.paths))


def index_depth_frames(variant_dir: Path) -> FrameIndex:
    """Map integer frame id -> depth PNG. ``7.png`` and ``007.png`` count as a duplicate."""
    depth_dir = variant_dir / DEPTH_DIR
    if not depth_dir.is_dir():
        raise DatasetError(f"{depth_dir} does not exist; the archive layout is not as expected")
    paths: dict[int, Path] = {}
    duplicates: set[int] = set()
    unparsable: list[str] = []
    for path in sorted(depth_dir.iterdir()):
        if path.suffix != DEPTH_SUFFIX or not path.is_file():
            unparsable.append(path.name)
            continue
        if not path.stem.isdigit():
            unparsable.append(path.name)
            continue
        frame_id = int(path.stem)
        if frame_id in paths:
            duplicates.add(frame_id)
        paths[frame_id] = path
    for frame_id in duplicates:
        paths.pop(frame_id, None)  # ambiguous: never pick one silently
    return FrameIndex(paths, tuple(sorted(duplicates)), tuple(unparsable))


def pose_table(records: list[PoseRecord], pose_id_offset: int) -> dict[int, np.ndarray]:
    """Image id -> stored transform. Keys must be integer frame ids; duplicates are errors."""
    table: dict[int, np.ndarray] = {}
    for record in records:
        if not record.key.isdigit():
            raise DatasetError(
                f"pose line {record.line_number}: first column {record.key!r} is not an integer "
                "frame id (ICL-NUIM pose keys are frame ids, not seconds)"
            )
        image_id = int(record.key) + pose_id_offset
        if image_id in table:
            raise DatasetError(f"pose line {record.line_number}: duplicate frame id {record.key}")
        table[image_id] = record.transform
    return table


@dataclass(frozen=True)
class PairingReport:
    matched: tuple[int, ...]
    only_in: dict[str, tuple[int, ...]] = field(default_factory=dict)
    duplicates: dict[str, tuple[int, ...]] = field(default_factory=dict)
    unparsable: dict[str, tuple[str, ...]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Ambiguous ids make pairing unusable; missing frames are reported, not fatal."""
        has_duplicates = any(self.duplicates.values())
        has_unparsable = any(self.unparsable.values())
        return bool(self.matched) and not has_duplicates and not has_unparsable

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "counts": self.counts,
            "matched_count": len(self.matched),
            "only_in": {key: list(ids) for key, ids in self.only_in.items()},
            "duplicates": {key: list(ids) for key, ids in self.duplicates.items()},
            "unparsable": {key: list(names) for key, names in self.unparsable.items()},
        }


def pair_frames(indices: Mapping[str, FrameIndex], pose_image_ids: set[int]) -> PairingReport:
    """Intersect frame ids across variants and poses; list everything left out."""
    id_sets = {name: set(index.paths) for name, index in indices.items()}
    id_sets["poses"] = set(pose_image_ids)
    matched = set.intersection(*id_sets.values())
    return PairingReport(
        matched=tuple(sorted(matched)),
        only_in={name: tuple(sorted(ids - matched)) for name, ids in id_sets.items()},
        duplicates={name: index.duplicates for name, index in indices.items()},
        unparsable={name: index.unparsable for name, index in indices.items()},
        counts={name: len(ids) for name, ids in id_sets.items()},
    )


def read_depth_png(path: Path) -> np.ndarray:
    """Raw 16-bit depth PNG as uint16 (H, W)."""
    import cv2  # optional dependency: pip install 'cudepthfusion[data]'

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise DatasetError(f"cannot read depth image {path}")
    if raw.dtype != np.uint16 or raw.ndim != 2:
        raise DatasetError(f"{path}: expected single-channel uint16, got {raw.dtype} {raw.shape}")
    return raw


def depth_to_metres(raw: np.ndarray, conventions: Conventions) -> np.ndarray:
    """Stored units -> camera Z in metres (float32); 0 stays 0 (invalid)."""
    depth = raw.astype(np.float64) / conventions.depth_units_per_meter
    if conventions.depth_kind == "ray_distance":
        height, width = raw.shape
        c = conventions
        depth = depth / ray_norms(width, height, c.fx, c.fy, c.cx, c.cy)
    elif conventions.depth_kind != "z":
        raise DatasetError(f"unknown depth_kind {conventions.depth_kind!r}")
    return depth.astype(np.float32)


def conventions_from_manifest(manifest: Mapping) -> Conventions:
    return Conventions(**manifest["conventions"])


class IclInputSequence:
    """Noisy depth, intrinsics and poses of a validated sequence, in frame-id order."""

    def __init__(self, manifest_path: Path, *, require_validated: bool = True) -> None:
        self._root = manifest_path.parent
        self._manifest = load_manifest(manifest_path)
        if require_validated:
            require_passed_validation(manifest_path)
        self._conventions = conventions_from_manifest(self._manifest)
        self._frame_ids = tuple(self._manifest["frames"]["matched_ids"])
        noisy_dir = self._root / self._manifest["files"]["noisy"]["extracted_dir"]
        self._noisy = index_depth_frames(noisy_dir)
        records = parse_tum_trajectory(self._root / self._manifest["files"]["poses"]["path"])
        self._poses = pose_table(records, self._conventions.pose_id_offset)
        c = self._conventions
        self._intrinsics = _core.Intrinsics(fx=c.fx, fy=c.fy, cx=c.cx, cy=c.cy)

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return self._frame_ids

    @property
    def conventions(self) -> Conventions:
        return self._conventions

    def __len__(self) -> int:
        return len(self._frame_ids)

    def __iter__(self) -> Iterator[InputFrame]:
        return (self.frame(frame_id) for frame_id in self._frame_ids)

    def frame(self, frame_id: int) -> InputFrame:
        stored = self._poses[frame_id]
        stores_world_camera = self._conventions.pose_direction == "T_world_camera"
        pose = stored if stores_world_camera else invert_rigid(stored)
        return InputFrame(
            frame_id=frame_id,
            timestamp_s=frame_id / self._conventions.frame_rate_hz,
            depth_m=depth_to_metres(read_depth_png(self._noisy.paths[frame_id]), self._conventions),
            intrinsics=self._intrinsics,
            T_world_camera=np.ascontiguousarray(pose),
        )


class IclGroundTruth:
    """Evaluator-only access to clean depth. Never pass this object to a filter."""

    def __init__(self, manifest_path: Path) -> None:
        manifest = load_manifest(manifest_path)
        self._conventions = conventions_from_manifest(manifest)
        clean_dir = manifest_path.parent / manifest["files"]["clean"]["extracted_dir"]
        self._clean = index_depth_frames(clean_dir)

    def clean_depth_m(self, frame_id: int) -> np.ndarray:
        return depth_to_metres(read_depth_png(self._clean.paths[frame_id]), self._conventions)

    def raw_clean_depth(self, frame_id: int) -> np.ndarray:
        return read_depth_png(self._clean.paths[frame_id])


def require_passed_validation(manifest_path: Path) -> None:
    """Refuse to hand out filter input until validate-data passed for this exact manifest."""
    report_path = manifest_path.parent / VALIDATION_NAME
    if not report_path.exists():
        raise DatasetError(
            f"{manifest_path.parent} has not been validated; run "
            f"`python -m cudepthfusion.cli validate-data --manifest {manifest_path}` first"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("manifest_sha256") != file_sha256(manifest_path):
        raise DatasetError(f"{report_path} belongs to a different {MANIFEST_NAME}; re-validate")
    if not report.get("passed"):
        raise DatasetError(f"{report_path} records a failed validation: {report.get('reasons')}")
