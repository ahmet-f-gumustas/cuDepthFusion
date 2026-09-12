"""validate-data: check pairing, noisy/clean consistency and conventions of one sequence.

The result is written to validation.json next to the manifest. Filter input is only
handed out (IclInputSequence) when that report passed for the exact manifest on disk.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from cudepthfusion.data import geometry_check as geo
from cudepthfusion.data.icl_nuim import (
    conventions_from_manifest,
    index_depth_frames,
    pair_frames,
    pose_table,
    read_depth_png,
)
from cudepthfusion.data.manifest import VALIDATION_NAME, file_sha256, load_manifest, write_json
from cudepthfusion.data.poses import parse_tum_trajectory

# Fixed before looking at real data.
NOISY_CLEAN_MAX_MEDIAN_M = 0.10  # any matched frame above this is a pairing/scale error
CONTRAST_SHIFT_FRAMES = 15  # "wrong pairing" reference: clean frame this many ids later
CONTRAST_EVERY = 20  # sample every n-th matched frame for the contrast check
CONTRAST_MIN_WIN_RATE = 0.95  # matched pair must beat the shifted pair this often

Logger = Callable[[str], None]


def validate_sequence(
    manifest_path: Path,
    *,
    source_frames: int = geo.DEFAULT_SOURCE_FRAMES,
    log: Logger = lambda _: None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    root = manifest_path.parent
    conventions = conventions_from_manifest(manifest)
    reasons: list[str] = []

    clean = index_depth_frames(root / manifest["files"]["clean"]["extracted_dir"])
    noisy = index_depth_frames(root / manifest["files"]["noisy"]["extracted_dir"])
    records = parse_tum_trajectory(root / manifest["files"]["poses"]["path"])
    pairing = pair_frames(
        {"clean": clean, "noisy": noisy}, set(pose_table(records, conventions.pose_id_offset))
    )
    if not pairing.ok:
        reasons.append("frame pairing has duplicate or unparsable ids (see pairing)")
    if list(pairing.matched) != manifest["frames"]["matched_ids"]:
        reasons.append("frames on disk no longer match the manifest's matched_ids")
    log(f"pairing: {len(pairing.matched)} matched, counts {pairing.counts}")

    cache: dict[int, np.ndarray] = {}

    def clean_raw(frame_id: int) -> np.ndarray:
        if frame_id not in cache:
            cache[frame_id] = read_depth_png(clean.paths[frame_id])
        return cache[frame_id]

    consistency = _noisy_clean_consistency(pairing.matched, clean, noisy, conventions, log)
    reasons.extend(consistency.pop("reasons"))

    poses_by_pose_id = pose_table(records, 0)
    # Only frames that have a pose under every candidate offset, so all candidates are
    # scored on identical pairs.
    # The declared offset and scale are always among the candidates.
    offsets = sorted({*geo.DEFAULT_POSE_OFFSETS, conventions.pose_id_offset})
    eligible = set(clean.ids)
    for offset in offsets:
        eligible &= {pose_id + offset for pose_id in poses_by_pose_id}
    pairs = geo.select_pairs(sorted(eligible), source_frames=source_frames)
    scales = sorted({conventions.depth_units_per_meter, *geo.DEFAULT_SCALES}, reverse=True)
    scores = []
    for candidate in geo.all_candidates(offsets=offsets, scales=scales):
        score = geo.score_candidate(
            candidate,
            pairs,
            clean_raw,
            poses_by_pose_id,
            conventions.fx,
            abs(conventions.fy),
            conventions.cx,
            conventions.cy,
        )
        if score is not None:
            scores.append(score)
    ranked = geo.rank_candidates(scores)
    declared = geo.Candidate(
        conventions.depth_kind,
        1 if conventions.fy > 0 else -1,
        conventions.pose_direction,
        conventions.pose_id_offset,
        conventions.depth_units_per_meter,
    )
    verdict = geo.judge(ranked, declared)
    reasons.extend(verdict.reasons)
    log(f"geometry: best = {verdict.best.candidate.label()} ({verdict.best.mean_median_rel:.5f})")

    report = {
        "passed": not reasons,
        "reasons": reasons,
        "manifest_sha256": file_sha256(manifest_path),
        "dataset": manifest["dataset"],
        "sequence": manifest["sequence"],
        "created_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "thresholds": {
            "geometry_mean_median_rel_max": geo.GATE_MEDIAN_REL_MAX,
            "geometry_inlier_fraction_min": geo.GATE_INLIER_MIN,
            "geometry_inlier_rel": geo.INLIER_REL,
            "geometry_pair_win_rate_min": geo.GATE_PAIR_WIN_RATE,
            "edge_jump_m": geo.EDGE_JUMP_M,
            "edge_dilate_px": geo.EDGE_DILATE_PX,
            "noisy_clean_max_median_m": NOISY_CLEAN_MAX_MEDIAN_M,
            "contrast_shift_frames": CONTRAST_SHIFT_FRAMES,
            "contrast_min_win_rate": CONTRAST_MIN_WIN_RATE,
        },
        "pairing": pairing.to_dict(),
        "noisy_clean_consistency": consistency,
        "geometry": {
            "pairs": [list(pair) for pair in pairs],
            "declared": verdict.declared.summary() if verdict.declared else None,
            "declared_pairs": [asdict(r) for r in verdict.declared.pairs]
            if verdict.declared
            else [],
            "best": verdict.best.summary(),
            "runner_up": verdict.runner_up.summary() if verdict.runner_up else None,
            "rivals": [asdict(rival) for rival in verdict.rivals],
            "ranking": [score.summary() for score in ranked],
        },
    }
    # Callers (e.g. the CLI) print this dict as JSON too, so it must be finite as well.
    report = _finite(report)
    write_json(manifest_path.parent / VALIDATION_NAME, report)
    return report


def _noisy_clean_consistency(
    matched: tuple[int, ...], clean, noisy, conventions, log: Logger
) -> dict[str, Any]:
    """Per-frame median |noisy - clean| on pixels valid in both, plus a pairing contrast."""
    units = conventions.depth_units_per_meter
    medians: dict[int, float] = {}
    valid_fraction: list[float] = []
    for index, frame_id in enumerate(matched):
        c = read_depth_png(clean.paths[frame_id]).astype(np.float64) / units
        n = read_depth_png(noisy.paths[frame_id]).astype(np.float64) / units
        both = (c > 0) & (n > 0)
        medians[frame_id] = float(np.median(np.abs(n[both] - c[both]))) if both.any() else math.inf
        valid_fraction.append(float(np.mean(n > 0)))
        if index % 200 == 0:
            log(f"  noisy/clean consistency {index}/{len(matched)}")

    matched_set = set(matched)
    wins, trials = 0, 0
    for frame_id in matched[::CONTRAST_EVERY]:
        shifted = frame_id + CONTRAST_SHIFT_FRAMES
        if shifted not in matched_set:
            continue
        c_shift = read_depth_png(clean.paths[shifted]).astype(np.float64) / units
        n = read_depth_png(noisy.paths[frame_id]).astype(np.float64) / units
        both = (c_shift > 0) & (n > 0)
        shifted_median = (
            float(np.median(np.abs(n[both] - c_shift[both]))) if both.any() else math.inf
        )
        wins += medians[frame_id] < shifted_median
        trials += 1

    worst = max(medians, key=medians.get) if medians else None
    values = np.array(list(medians.values())) if medians else np.array([math.inf])
    reasons = []
    bad = [fid for fid, value in medians.items() if value > NOISY_CLEAN_MAX_MEDIAN_M]
    if bad:
        reasons.append(
            f"{len(bad)} matched frame(s) differ from clean by more than "
            f"{NOISY_CLEAN_MAX_MEDIAN_M} m median (first: {bad[:5]})"
        )
    win_rate = wins / trials if trials else 0.0
    if trials == 0 or win_rate < CONTRAST_MIN_WIN_RATE:
        reasons.append(
            f"matched noisy/clean pairs beat a {CONTRAST_SHIFT_FRAMES}-frame shift only "
            f"{wins}/{trials} times"
        )
    return {
        "frames_checked": len(medians),
        "median_abs_diff_m": {
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
            "worst_frame_id": worst,
        },
        "noisy_valid_fraction_mean": float(np.mean(valid_fraction)) if valid_fraction else 0.0,
        "contrast": {"wins": wins, "trials": trials, "win_rate": win_rate},
        "reasons": reasons,
    }


def _finite(value: Any) -> Any:
    """JSON has no Infinity/NaN: store them as null (with the reason recorded elsewhere)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(item) for item in value]
    return value
