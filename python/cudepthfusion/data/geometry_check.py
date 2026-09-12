"""Two-frame reprojection check that decides a dataset's conventions from evidence.

Evaluator-side code: it reads clean depth and ground-truth poses, never filter input.
Every candidate convention (depth kind, fy sign, pose direction, pose/image id offset,
depth scale) warps clean frame i into frame j with the ground-truth relative pose and is
scored by how well the predicted Z matches the observed Z on interior surfaces.

The observed Z is sampled by bilinear interpolation of inverse depth, which is exact for
planes under perspective projection. Nearest-pixel sampling (the core's rule) would add
a rounding floor that hides small convention errors such as a one-frame pose offset.

The thresholds below were fixed on synthetic tests before running on any real data.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from cudepthfusion.data.camera import ray_norms
from cudepthfusion.data.poses import invert_rigid

DEPTH_KINDS = ("z", "ray_distance")
POSE_DIRECTIONS = ("T_world_camera", "T_camera_world")
DEFAULT_POSE_OFFSETS = (-1, 0, 1)
DEFAULT_SCALES = (5000.0, 1000.0)
DEFAULT_BASELINES = (1, 5, 15, 30)
DEFAULT_SOURCE_FRAMES = 8

EDGE_JUMP_M = 0.05  # same 4-neighbour jump as the evaluation protocol's edge mask
EDGE_DILATE_PX = 2
INLIER_REL = 0.01  # |dz| / z below which a warped pixel counts as consistent

GATE_MEDIAN_REL_MAX = 0.005  # declared convention: mean per-pair median |dz|/z <= 0.5 %
GATE_INLIER_MIN = 0.90  # declared convention: mean inlier fraction >= 90 %
# Identifiability: for every parameter, the declared convention must beat the best
# candidate with a different value of that parameter on at least this share of pairs.
GATE_PAIR_WIN_RATE = 0.90
PARAMETERS = ("depth_kind", "fy_sign", "pose_direction", "pose_id_offset", "depth_units_per_meter")


@dataclass(frozen=True)
class Candidate:
    depth_kind: str
    fy_sign: int
    pose_direction: str
    pose_id_offset: int
    depth_units_per_meter: float

    def label(self) -> str:
        return (
            f"{self.depth_kind}, fy{'+' if self.fy_sign > 0 else '-'}, {self.pose_direction}, "
            f"offset {self.pose_id_offset:+d}, {self.depth_units_per_meter:g}/m"
        )


@dataclass(frozen=True)
class PairResidual:
    source_image_id: int
    target_image_id: int
    num_compared: int
    median_abs_m: float
    median_rel: float
    inlier_fraction: float


@dataclass(frozen=True)
class CandidateScore:
    candidate: Candidate
    mean_median_rel: float
    mean_inlier_fraction: float
    num_pairs: int
    pairs: tuple[PairResidual, ...]

    def summary(self) -> dict:
        return {
            "candidate": asdict(self.candidate),
            "label": self.candidate.label(),
            "mean_median_rel": self.mean_median_rel,
            "mean_inlier_fraction": self.mean_inlier_fraction,
            "num_pairs": self.num_pairs,
        }


@dataclass(frozen=True)
class Rival:
    """Strongest candidate that differs from the declared one in ``parameter``."""

    parameter: str
    label: str
    mean_median_rel: float
    pairs_compared: int
    declared_win_rate: float


@dataclass(frozen=True)
class GeometryVerdict:
    passed: bool
    best: CandidateScore
    declared: CandidateScore | None  # None when the declared convention scored no pair
    runner_up: CandidateScore | None
    rivals: tuple[Rival, ...]
    reasons: tuple[str, ...]


def all_candidates(
    fy_magnitude_signs: Sequence[int] = (1, -1),
    offsets: Sequence[int] = DEFAULT_POSE_OFFSETS,
    scales: Sequence[float] = DEFAULT_SCALES,
) -> list[Candidate]:
    return [
        Candidate(kind, sign, direction, offset, scale)
        for kind, sign, direction, offset, scale in itertools.product(
            DEPTH_KINDS, fy_magnitude_signs, POSE_DIRECTIONS, offsets, scales
        )
    ]


def select_pairs(
    image_ids: Sequence[int],
    baselines: Sequence[int] = DEFAULT_BASELINES,
    source_frames: int = DEFAULT_SOURCE_FRAMES,
) -> list[tuple[int, int]]:
    """Evenly spaced source frames, each paired with the frame ``b`` images later."""
    available = set(image_ids)
    ordered = sorted(image_ids)
    longest = max(baselines)
    usable = [i for i in ordered if i + longest in available] or ordered
    step = max(len(usable) // source_frames, 1)
    sources = usable[::step][:source_frames]
    return [(s, s + b) for s in sources for b in baselines if s + b in available]


def interior_mask(z: np.ndarray) -> np.ndarray:
    """Valid pixels away from depth discontinuities (4-neighbour jump > EDGE_JUMP_M)."""
    valid = z > 0
    edge = np.zeros_like(valid)
    for axis in (0, 1):
        jump = np.abs(np.diff(z, axis=axis)) > EDGE_JUMP_M
        both = np.logical_and(
            np.take(valid, range(z.shape[axis] - 1), axis=axis),
            np.take(valid, range(1, z.shape[axis]), axis=axis),
        )
        mark = jump | ~both
        if axis == 0:
            edge[:-1, :] |= mark
            edge[1:, :] |= mark
        else:
            edge[:, :-1] |= mark
            edge[:, 1:] |= mark
    for _ in range(EDGE_DILATE_PX):
        grown = edge.copy()
        grown[1:, :] |= edge[:-1, :]
        grown[:-1, :] |= edge[1:, :]
        grown[:, 1:] |= edge[:, :-1]
        grown[:, :-1] |= edge[:, 1:]
        edge = grown
    return valid & ~edge


def warp_residual(
    z_source: np.ndarray,
    z_target: np.ndarray,
    T_target_source: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted and observed target Z for interior source pixels landing on interior targets.

    The observed Z comes from bilinear interpolation of 1/Z over the four surrounding
    target pixels, all of which must be interior.
    """
    height, width = z_source.shape
    rows, cols = np.nonzero(interior_mask(z_source))
    z = z_source[rows, cols]
    points = np.stack([(cols - cx) / fx * z, (rows - cy) / fy * z, z, np.ones_like(z)], axis=0)
    moved = T_target_source @ points
    zc = moved[2]
    in_front = zc > 0
    safe_z = np.where(in_front, zc, 1.0)
    u = fx * moved[0] / safe_z + cx
    v = fy * moved[1] / safe_z + cy
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    keep = in_front & (u0 >= 0) & (u0 + 1 < width) & (v0 >= 0) & (v0 + 1 < height)
    u0, v0, du, dv, zc = (
        u0[keep],
        v0[keep],
        (u - np.floor(u))[keep],
        (v - np.floor(v))[keep],
        zc[keep],
    )

    target_interior = interior_mask(z_target)
    corners = [(v0, u0), (v0, u0 + 1), (v0 + 1, u0), (v0 + 1, u0 + 1)]
    usable = np.logical_and.reduce([target_interior[r, c] for r, c in corners])
    inverse = np.zeros_like(z_target)
    inverse[target_interior] = 1.0 / z_target[target_interior]
    weights = [(1 - du) * (1 - dv), du * (1 - dv), (1 - du) * dv, du * dv]
    inv_observed = sum(w * inverse[r, c] for w, (r, c) in zip(weights, corners, strict=True))
    return zc[usable], 1.0 / inv_observed[usable]


def score_candidate(
    candidate: Candidate,
    pairs: Sequence[tuple[int, int]],
    load_raw_depth: Callable[[int], np.ndarray],
    poses_by_id: Mapping[int, np.ndarray],
    fx: float,
    fy_magnitude: float,
    cx: float,
    cy: float,
) -> CandidateScore | None:
    fy = candidate.fy_sign * fy_magnitude
    residuals: list[PairResidual] = []
    norms: np.ndarray | None = None
    for source, target in pairs:
        pose_s = poses_by_id.get(source - candidate.pose_id_offset)
        pose_t = poses_by_id.get(target - candidate.pose_id_offset)
        if pose_s is None or pose_t is None:
            continue
        raw_s, raw_t = load_raw_depth(source), load_raw_depth(target)
        z_s = raw_s.astype(np.float64) / candidate.depth_units_per_meter
        z_t = raw_t.astype(np.float64) / candidate.depth_units_per_meter
        if candidate.depth_kind == "ray_distance":
            if norms is None:
                norms = ray_norms(z_s.shape[1], z_s.shape[0], fx, fy, cx, cy)
            z_s, z_t = z_s / norms, z_t / norms
        if candidate.pose_direction == "T_camera_world":
            pose_s, pose_t = invert_rigid(pose_s), invert_rigid(pose_t)
        predicted, observed = warp_residual(z_s, z_t, invert_rigid(pose_t) @ pose_s, fx, fy, cx, cy)
        if predicted.size == 0:
            residuals.append(PairResidual(source, target, 0, np.inf, np.inf, 0.0))
            continue
        abs_err = np.abs(predicted - observed)
        rel_err = abs_err / observed
        residuals.append(
            PairResidual(
                source,
                target,
                int(predicted.size),
                float(np.median(abs_err)),
                float(np.median(rel_err)),
                float(np.mean(rel_err <= INLIER_REL)),
            )
        )
    if not residuals:
        return None
    return CandidateScore(
        candidate=candidate,
        mean_median_rel=float(np.mean([r.median_rel for r in residuals])),
        mean_inlier_fraction=float(np.mean([r.inlier_fraction for r in residuals])),
        num_pairs=len(residuals),
        pairs=tuple(residuals),
    )


def rank_candidates(scores: Sequence[CandidateScore]) -> list[CandidateScore]:
    """Best first. Candidates missing pairs rank after complete ones."""
    most_pairs = max(score.num_pairs for score in scores)
    return sorted(scores, key=lambda s: (s.num_pairs < most_pairs, s.mean_median_rel))


def paired_win_rate(declared: CandidateScore, rival: CandidateScore) -> tuple[float, int]:
    """Share of common pairs on which ``declared`` has the lower median |dz|/z."""
    rival_by_pair = {(r.source_image_id, r.target_image_id): r.median_rel for r in rival.pairs}
    outcomes = [
        pair.median_rel < rival_by_pair[key]
        for pair in declared.pairs
        if (key := (pair.source_image_id, pair.target_image_id)) in rival_by_pair
    ]
    return (sum(outcomes) / len(outcomes) if outcomes else 0.0), len(outcomes)


def find_rivals(ranked: Sequence[CandidateScore], declared: CandidateScore) -> list[Rival]:
    rivals = []
    for parameter in PARAMETERS:
        value = getattr(declared.candidate, parameter)
        rival = next((s for s in ranked if getattr(s.candidate, parameter) != value), None)
        if rival is None:
            continue
        win_rate, compared = paired_win_rate(declared, rival)
        rivals.append(
            Rival(parameter, rival.candidate.label(), rival.mean_median_rel, compared, win_rate)
        )
    return rivals


def judge(ranked: Sequence[CandidateScore], declared: Candidate) -> GeometryVerdict:
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    declared_score = next((s for s in ranked if s.candidate == declared), None)
    reasons: list[str] = []
    if declared_score is None:
        reason = f"declared convention ({declared.label()}) could not be scored on any pair"
        return GeometryVerdict(False, best, None, runner_up, (), (reason,))
    rivals = find_rivals(ranked, declared_score)
    if best.candidate != declared:
        reasons.append(
            f"best candidate ({best.candidate.label()}) differs from the declared convention"
        )
    if declared_score.mean_median_rel > GATE_MEDIAN_REL_MAX:
        reasons.append(
            f"declared mean median |dz|/z {declared_score.mean_median_rel:.4f} > "
            f"{GATE_MEDIAN_REL_MAX}"
        )
    if declared_score.mean_inlier_fraction < GATE_INLIER_MIN:
        reasons.append(
            f"declared inlier fraction {declared_score.mean_inlier_fraction:.3f} < "
            f"{GATE_INLIER_MIN}"
        )
    for rival in rivals:
        if rival.declared_win_rate < GATE_PAIR_WIN_RATE:
            reasons.append(
                f"{rival.parameter} is not identifiable: the declared convention beats "
                f"'{rival.label}' on only {rival.declared_win_rate:.0%} of "
                f"{rival.pairs_compared} pairs (need {GATE_PAIR_WIN_RATE:.0%})"
            )
    return GeometryVerdict(
        not reasons, best, declared_score, runner_up, tuple(rivals), tuple(reasons)
    )
