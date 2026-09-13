"""Named scenarios and the lazily evaluated, reproducible synthetic sequence.

Coordinates: x right, y down, z forward (camera frame at t = 0 equals the world frame
for the static-camera scenarios). Ground truth and filter input are separate objects,
mirroring the real-data adapters.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from cudepthfusion.data.camera import PinholeCamera
from cudepthfusion.data.frames import InputFrame
from cudepthfusion.synthetic import trajectory as traj
from cudepthfusion.synthetic.noise import NoiseModel
from cudepthfusion.synthetic.scene import Rectangle, Scene, box_room, render, wall

FRAME_RATE_HZ = 30.0
DEFAULT_FRAMES = 60
ROOM_MIN = (-2.0, -1.0, -3.0)  # asymmetric, so a mirrored axis cannot fit the geometry
ROOM_MAX = (2.5, 2.0, 4.0)


@dataclass(frozen=True)
class Scenario:
    name: str
    purpose: str
    scene: Scene
    trajectory: traj.Trajectory
    motion: str
    frames: int = DEFAULT_FRAMES


@dataclass(frozen=True, eq=False)
class GroundTruthFrame:
    """Evaluator-only truth for one frame."""

    frame_id: int
    depth_m: np.ndarray  # float64 exact camera Z; 0 where no surface is hit
    surface_id: np.ndarray  # uint16; 0 = no surface
    dynamic_mask: np.ndarray  # bool; pixels showing a moving surface


@dataclass(frozen=True, eq=False)
class SyntheticFrame:
    input: InputFrame  # what a filter may see
    truth: GroundTruthFrame  # never pass this to a filter


def _scenarios() -> dict[str, Scenario]:
    diagonal = math.sqrt(0.5)
    scenarios = [
        Scenario(
            "plane_static",
            "fronto-parallel plane, static camera: bias and temporal variance (spec 9.2)",
            Scene((wall(1, 2.0),)),
            traj.static(),
            "static",
        ),
        Scenario(
            "plane_lateral",
            "fronto-parallel plane, camera translating in x: pose-compensated fusion (spec 9.1)",
            Scene((wall(1, 2.0),)),
            traj.linear((0.3, 0.0, 0.0)),
            "translation +x 0.3 m/s",
        ),
        Scenario(
            "plane_dolly",
            "fronto-parallel plane, camera moving towards it: pure Z motion (spec 9.1)",
            Scene((wall(1, 3.0),)),
            traj.linear((0.0, 0.0, 0.3)),
            "translation +z 0.3 m/s",
        ),
        Scenario(
            "slanted_plane",
            "plane at 45 degrees about y, slow lateral motion: slanted-surface bias",
            Scene((Rectangle(1, (0.0, 0.0, 2.5), (diagonal, 0.0, diagonal), (0.0, 1.0, 0.0)),)),
            traj.linear((0.1, 0.0, 0.0)),
            "translation +x 0.1 m/s",
        ),
        Scenario(
            "step",
            "depth step: near half-plane (x < 0) at 1.5 m over a far plane at 2.5 m; "
            "front and back must not blend (spec 9.2)",
            Scene(
                (
                    Rectangle(1, (-50.0, 0.0, 1.5), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 50.0),
                    wall(2, 2.5),
                )
            ),
            traj.static(),
            "static",
        ),
        Scenario(
            "front_surface_pass",
            "a 0.6 m square at 1.5 m crosses a static view of a wall at 3 m, entering and "
            "leaving: no ghosting after it leaves (spec 9.2, 10.4)",
            Scene(
                (
                    wall(1, 3.0),
                    wall(
                        2,
                        1.5,
                        half_u=0.3,
                        half_v=0.3,
                        center_xy=(-1.3, 0.0),
                        velocity=(1.4, 0.0, 0.0),
                    ),
                )
            ),
            traj.static(),
            "static",
        ),
        Scenario(
            "revealed_background",
            "a 1.2 m occluder at 1.5 m moves out of view and reveals a wall at 3 m: stale "
            "foreground must not persist (spec 10.4)",
            Scene(
                (
                    wall(1, 3.0),
                    wall(2, 1.5, half_u=0.6, half_v=0.6, velocity=(0.0, 0.9, 0.0)),
                )
            ),
            traj.static(),
            "static",
        ),
        Scenario(
            "room_handheld",
            "closed box room, smooth 6-DoF handheld motion",
            Scene(box_room(ROOM_MIN, ROOM_MAX)),
            traj.handheld(FRAME_RATE_HZ),
            "handheld 6-DoF",
        ),
        Scenario(
            "room_spin",
            "closed box room, pure rotation about the camera centre (spec 9.1)",
            Scene(box_room(ROOM_MIN, ROOM_MAX)),
            traj.spin((0.0, 1.0, 0.0), 0.5, position=(0.3, 0.2, -0.5)),
            "yaw 0.5 rad/s",
        ),
    ]
    return {scenario.name: scenario for scenario in scenarios}


SCENARIOS = _scenarios()


def scenario_names() -> tuple[str, ...]:
    return tuple(SCENARIOS)


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        raise KeyError(f"unknown scenario {name!r}; known: {list(SCENARIOS)}") from None


class SyntheticSequence:
    """Frames are rendered on demand. Frame ``i`` uses the noise generator seeded with
    ``(seed, i)``, so any frame is reproducible in any access order."""

    def __init__(
        self,
        scenario: Scenario | str,
        *,
        camera: PinholeCamera | None = None,
        noise: NoiseModel | None = None,
        seed: int = 0,
        frames: int | None = None,
        frame_rate_hz: float = FRAME_RATE_HZ,
    ) -> None:
        self.scenario = get_scenario(scenario) if isinstance(scenario, str) else scenario
        self.camera = camera or PinholeCamera.vga()
        self.noise = noise or NoiseModel()
        self.seed = seed
        self.frame_rate_hz = frame_rate_hz
        self._length = self.scenario.frames if frames is None else frames
        if self._length < 1:
            raise ValueError(f"frames must be >= 1, got {self._length}")
        self._intrinsics = self.camera.intrinsics()

    @property
    def name(self) -> str:
        return self.scenario.name

    @property
    def frame_ids(self) -> range:
        return range(self._length)

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[SyntheticFrame]:
        return (self.frame(frame_id) for frame_id in self.frame_ids)

    def pose(self, frame_id: int) -> np.ndarray:
        return np.ascontiguousarray(self.scenario.trajectory(frame_id / self.frame_rate_hz))

    def frame(self, frame_id: int) -> SyntheticFrame:
        if frame_id not in self.frame_ids:
            raise IndexError(f"frame {frame_id} outside 0..{self._length - 1}")
        time_s = frame_id / self.frame_rate_hz
        pose = self.pose(frame_id)
        depth, surface_id = render(self.scenario.scene, self.camera, pose, time_s)
        rng = np.random.default_rng([self.seed, frame_id])
        noisy = np.ascontiguousarray(self.noise.apply(depth, rng))
        dynamic = np.isin(surface_id, list(self.scenario.scene.dynamic_surface_ids))
        return SyntheticFrame(
            input=InputFrame(frame_id, time_s, noisy, self._intrinsics, pose),
            truth=GroundTruthFrame(frame_id, depth, surface_id, dynamic),
        )
