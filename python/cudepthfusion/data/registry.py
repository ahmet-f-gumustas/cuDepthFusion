"""Known dataset sequences: official URLs, license and declared data conventions.

URLs are taken from the official ICL-NUIM page. Conventions are declared here and must
be confirmed per sequence by ``cudepthfusion.cli validate-data`` before any benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ICL_NUIM_PAGE = "https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html"
ICL_NUIM_READER_PAGE = "https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html"
ICL_NUIM_LICENSE = "CC BY 3.0"
ICL_NUIM_LICENSE_URL = "http://creativecommons.org/licenses/by/3.0/"
ICL_NUIM_CITATION = (
    "A. Handa, T. Whelan, J.B. McDonald and A.J. Davison, "
    "A Benchmark for RGB-D Visual Odometry, 3D Reconstruction and SLAM, ICRA 2014"
)
_ICL_BASE = "https://www.doc.ic.ac.uk/~ahanda"


@dataclass(frozen=True)
class RemoteFile:
    url: str
    is_archive: bool

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class Conventions:
    """How to turn the stored files into the core data contract.

    ``depth_kind`` is "z" (camera Z) or "ray_distance" (Euclidean distance along the ray).
    ``pose_direction`` names what the pose file stores: "T_world_camera" or
    "T_camera_world". ``pose_id_offset`` maps pose ids to image ids: image = pose + offset.
    """

    depth_kind: str
    depth_units_per_meter: float
    fx: float
    fy: float
    cx: float
    cy: float
    pose_direction: str
    pose_id_offset: int
    frame_rate_hz: float


@dataclass(frozen=True)
class SequenceSpec:
    dataset: str
    name: str
    scene: str
    split: str
    frames_listed_on_page: int
    files: dict[str, RemoteFile] = field(default_factory=dict)
    conventions: Conventions | None = None
    official_page: str = ICL_NUIM_PAGE
    license: str = ICL_NUIM_LICENSE
    license_url: str = ICL_NUIM_LICENSE_URL
    citation: str = ICL_NUIM_CITATION


# TUM RGB-D compatible PNG distribution. Declared values; see docs/DATA_VALIDATION.md for
# the evidence. The reader page gives K with fy = -480 for the native POV-Ray format.
ICL_TUM_PNG_CONVENTIONS = Conventions(
    depth_kind="z",
    depth_units_per_meter=5000.0,
    fx=481.20,
    fy=-480.00,
    cx=319.50,
    cy=239.50,
    pose_direction="T_world_camera",
    pose_id_offset=0,
    frame_rate_hz=30.0,
)

# Splits from the development spec: kt0 development, kt1 parameter selection,
# kt2/kt3 final evaluation. All four are trajectories through the same scene.
_ICL_LIVING_ROOM = {
    "kt0": ("development", 1510, "living_room_traj0", "livingRoom0"),
    "kt1": ("validation", 967, "living_room_traj1", "livingRoom1"),
    "kt2": ("test", 882, "living_room_traj2", "livingRoom2"),
    "kt3": ("test", 1242, "living_room_traj3", "livingRoom3"),
}

VARIANTS = ("clean", "noisy", "poses")

SEQUENCES: dict[str, dict[str, SequenceSpec]] = {
    "icl-nuim": {
        name: SequenceSpec(
            dataset="icl-nuim",
            name=name,
            scene="living_room",
            split=split,
            frames_listed_on_page=frames,
            files={
                "clean": RemoteFile(f"{_ICL_BASE}/{stem}_frei_png.tar.gz", is_archive=True),
                "noisy": RemoteFile(f"{_ICL_BASE}/{stem}n_frei_png.tar.gz", is_archive=True),
                "poses": RemoteFile(f"{_ICL_BASE}/VaFRIC/{pose}.gt.freiburg", is_archive=False),
            },
            conventions=ICL_TUM_PNG_CONVENTIONS,
        )
        for name, (split, frames, stem, pose) in _ICL_LIVING_ROOM.items()
    }
}


def get_sequence(dataset: str, sequence: str) -> SequenceSpec:
    try:
        return SEQUENCES[dataset][sequence]
    except KeyError:
        known = {name: sorted(seqs) for name, seqs in SEQUENCES.items()}
        message = f"unknown dataset/sequence {dataset!r}/{sequence!r}; known: {known}"
        raise KeyError(message) from None
