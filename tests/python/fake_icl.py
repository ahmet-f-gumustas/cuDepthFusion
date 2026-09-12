"""A small ICL-NUIM look-alike built from the analytic room, plus an offline downloader.

Lets fetch, pairing and validation run end to end without network access, with known
ground-truth conventions that tests can then declare correctly or incorrectly.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import tarfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from synthetic_scene import camera_pose, ray_distance, render_z

from cudepthfusion.data import download as dl
from cudepthfusion.data.registry import Conventions, RemoteFile, SequenceSpec

WIDTH, HEIGHT = 80, 60
FX, FY_ABS, CX, CY = 70.0, 70.0, 39.5, 29.5
UNITS_PER_M = 5000.0
URLS = {
    "clean": "http://fake.invalid/clean.tar.gz",
    "noisy": "http://fake.invalid/noisy.tar.gz",
    "poses": "http://fake.invalid/poses.freiburg",
}


@dataclass(frozen=True)
class FakeIcl:
    spec: SequenceSpec
    truth: Conventions
    downloader: Callable[..., dl.DownloadResult]


def rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """(qx, qy, qz, qw) of a proper rotation matrix."""
    trace = float(np.trace(rotation))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        return (
            (rotation[2, 1] - rotation[1, 2]) / s,
            (rotation[0, 2] - rotation[2, 0]) / s,
            (rotation[1, 0] - rotation[0, 1]) / s,
            0.25 * s,
        )
    i = int(np.argmax(np.diag(rotation)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + rotation[i, i] - rotation[j, j] - rotation[k, k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (rotation[j, i] + rotation[i, j]) / s
    q[k] = (rotation[k, i] + rotation[i, k]) / s
    q[3] = (rotation[k, j] - rotation[j, k]) / s
    return q[0], q[1], q[2], q[3]


def build_fake_icl(
    root: Path,
    *,
    frames: int = 48,
    depth_kind: str = "z",
    fy_sign: int = -1,
    pose_offset: int = -1,
    declared: dict | None = None,
    missing_noisy: Sequence[int] = (),
    duplicate_noisy: int | None = None,
    swap_noisy: tuple[int, int] | None = None,
    noise_m: float = 0.003,
    seed: int = 0,
) -> FakeIcl:
    """``pose_offset`` follows the adapter's rule: image id = pose id + offset."""
    truth = Conventions(
        depth_kind=depth_kind,
        depth_units_per_meter=UNITS_PER_M,
        fx=FX,
        fy=fy_sign * FY_ABS,
        cx=CX,
        cy=CY,
        pose_direction="T_world_camera",
        pose_id_offset=pose_offset,
        frame_rate_hz=30.0,
    )
    rng = np.random.default_rng(seed)
    stage = root / "stage"
    for variant in ("clean", "noisy"):
        (stage / variant / "depth").mkdir(parents=True, exist_ok=True)

    noisy_frames: dict[int, np.ndarray] = {}
    pose_lines = []
    for image_id in range(frames):
        pose = camera_pose(image_id)
        z = render_z(pose, WIDTH, HEIGHT, FX, truth.fy, CX, CY)
        stored = z if depth_kind == "z" else ray_distance(z, FX, truth.fy, CX, CY)
        clean = np.round(stored * UNITS_PER_M).astype(np.uint16)
        noisy_m = stored + rng.normal(0.0, noise_m, stored.shape)
        noisy = np.clip(np.round(noisy_m * UNITS_PER_M), 0, 65535).astype(np.uint16)
        noisy[rng.random(stored.shape) < 0.02] = 0  # sensor holes
        cv2.imwrite(str(stage / "clean" / "depth" / f"{image_id}.png"), clean)
        noisy_frames[image_id] = noisy
        qx, qy, qz, qw = rotation_to_quaternion(pose[:3, :3])
        tx, ty, tz = pose[:3, 3]
        pose_lines.append(f"{image_id - pose_offset} {tx} {ty} {tz} {qx} {qy} {qz} {qw}")

    if swap_noisy is not None:
        a, b = swap_noisy
        noisy_frames[a], noisy_frames[b] = noisy_frames[b], noisy_frames[a]
    for image_id, noisy in noisy_frames.items():
        if image_id not in missing_noisy:
            cv2.imwrite(str(stage / "noisy" / "depth" / f"{image_id}.png"), noisy)
    if duplicate_noisy is not None:
        cv2.imwrite(
            str(stage / "noisy" / "depth" / f"{duplicate_noisy:03d}.png"),
            noisy_frames[duplicate_noisy],
        )

    published = root / "published"
    published.mkdir(parents=True, exist_ok=True)
    for variant in ("clean", "noisy"):
        with tarfile.open(published / URLS[variant].rsplit("/", 1)[-1], "w:gz") as tar:
            tar.add(stage / variant / "depth", arcname="depth")
    (published / "poses.freiburg").write_text("\n".join(pose_lines) + "\n", encoding="utf-8")

    def downloader(url: str, dest: Path, **_: object) -> dl.DownloadResult:
        source = published / url.rsplit("/", 1)[-1]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        return dl.DownloadResult(url, dest, dest.stat().st_size, digest, reused_existing=False)

    spec = SequenceSpec(
        dataset="fake-icl",
        name="kt0",
        scene="analytic_room",
        split="development",
        frames_listed_on_page=frames,
        files={
            variant: RemoteFile(url, is_archive=variant != "poses") for variant, url in URLS.items()
        },
        conventions=replace(truth, **(declared or {})),
        official_page="http://fake.invalid/",
    )
    return FakeIcl(spec=spec, truth=truth, downloader=downloader)
