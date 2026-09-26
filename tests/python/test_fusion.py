"""Fusion behaviour on the analytic scenes: the spec's section 9.2 cases."""

from __future__ import annotations

import numpy as np
import pytest

import cudepthfusion as cdf
from cudepthfusion.data.camera import PinholeCamera
from cudepthfusion.synthetic import NOISELESS, SyntheticSequence

SMALL = PinholeCamera.vga().scaled(0.25)


def run(sequence: SyntheticSequence, config: dict | None = None) -> list:
    engine = cdf.DepthFusion(config or {}, backend="cpu")
    results = []
    for frame in sequence:
        results.append(
            engine.process(
                depth_m=frame.input.depth_m,
                intrinsics=frame.input.intrinsics,
                T_world_camera=frame.input.T_world_camera,
                timestamp_s=frame.input.timestamp_s,
            )
        )
    return results


def rmse(estimate: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate[mask] - truth[mask]) ** 2)))


def test_noise_free_plane_gains_no_bias() -> None:
    sequence = SyntheticSequence("plane_static", camera=SMALL, noise=NOISELESS, frames=10)
    results = run(sequence)

    assert results[0].diagnostics.temporal_status == "no_history"
    assert results[1].diagnostics.temporal_status == "fused"
    for result in results[1:]:
        assert result.diagnostics.fusion.fused == result.diagnostics.input.num_valid
        np.testing.assert_allclose(result.depth_m[result.valid_mask == 1], 2.0, atol=1e-5)


def test_temporal_fusion_reduces_noise_on_a_static_plane() -> None:
    sequence = SyntheticSequence("plane_static", camera=SMALL, frames=30, seed=3)
    raw = np.stack([frame.input.depth_m for frame in sequence])
    fused = np.stack([result.depth_m for result in run(sequence)])

    steady = slice(10, 30)  # let the history build up first
    usable = np.all(raw[steady] > 0, axis=0) & np.all(fused[steady] > 0, axis=0)
    raw_std = np.median(raw[steady][:, usable].std(axis=0))
    fused_std = np.median(fused[steady][:, usable].std(axis=0))
    assert fused_std < 0.5 * raw_std, (fused_std, raw_std)


def test_moving_camera_fusion_beats_the_raw_input() -> None:
    sequence = SyntheticSequence("plane_lateral", camera=SMALL, frames=20, seed=5)
    results = run(sequence)
    frames = list(sequence)

    last = 19
    truth = frames[last].truth.depth_m
    mask = (results[last].valid_mask == 1) & (frames[last].input.depth_m > 0) & (truth > 0)
    assert rmse(results[last].depth_m, truth, mask) < rmse(frames[last].input.depth_m, truth, mask)
    assert results[last].diagnostics.fusion.fused > 0.8 * int(mask.sum())


def test_depth_step_does_not_blend_front_and_back() -> None:
    sequence = SyntheticSequence("step", camera=SMALL, noise=NOISELESS, frames=8)
    result = run(sequence)[-1]
    valid = result.valid_mask == 1
    distance_to_a_surface = np.minimum(
        np.abs(result.depth_m[valid] - 1.5), np.abs(result.depth_m[valid] - 2.5)
    )
    assert distance_to_a_surface.max() < 0.02  # nothing lands between the two surfaces


def test_a_new_front_surface_replaces_the_history_immediately() -> None:
    sequence = SyntheticSequence("front_surface_pass", camera=SMALL, noise=NOISELESS, frames=40)
    results = run(sequence)
    frames = list(sequence)

    arrival = next(i for i, f in enumerate(frames) if f.truth.dynamic_mask.any())
    covered = frames[arrival].truth.dynamic_mask
    result = results[arrival]
    assert results[arrival - 1].diagnostics.fusion.fused > 0  # history existed before it arrived
    np.testing.assert_allclose(result.depth_m[covered], 1.5, atol=0.02)
    assert result.diagnostics.fusion.rejected_current_nearer >= int(covered.sum())


def test_the_background_recovers_right_after_the_occluder_leaves() -> None:
    sequence = SyntheticSequence("revealed_background", camera=SMALL, noise=NOISELESS, frames=60)
    results = run(sequence)
    frames = list(sequence)

    gone = next(i for i, f in enumerate(frames) if i > 0 and not f.truth.dynamic_mask.any())
    revealed = frames[gone - 1].truth.dynamic_mask  # pixels the occluder covered one frame earlier
    result = results[gone]
    np.testing.assert_allclose(result.depth_m[revealed], 3.0, atol=0.02)
    assert result.diagnostics.fusion.rejected_current_farther >= int(revealed.sum())


def test_missing_measurements_stay_invalid_unless_hole_filling_is_enabled() -> None:
    sequence = SyntheticSequence("plane_static", camera=SMALL, frames=6, seed=11)
    frames = list(sequence)
    holes = frames[-1].input.depth_m == 0
    assert holes.any()

    default_result = run(sequence)[-1]
    assert not default_result.valid_mask[holes].any()
    assert default_result.diagnostics.fusion.history_only == 0

    filled = run(sequence, {"fusion": {"fill_holes": True, "max_history_age_frames": 2}})[-1]
    assert filled.diagnostics.fusion.history_only > 0
    filled_pixels = (filled.source_mask == 3) & holes
    assert filled_pixels.any()
    assert (filled.history_age[filled_pixels] >= 1).all()


def test_fusion_counters_add_up() -> None:
    sequence = SyntheticSequence("room_handheld", camera=SMALL, frames=6, seed=2)
    result = run(sequence)[-1]
    stats = result.diagnostics.fusion
    accounted = (
        stats.fused
        + stats.current_only
        + stats.rejected_current_nearer
        + stats.rejected_current_farther
        + stats.history_only
        + stats.invalid
    )
    assert accounted == result.depth_m.size
    assert stats.prior_visible <= stats.prior_candidates
    assert 0.0 <= stats.mean_prior_weight <= 1.0


def test_losing_the_pose_drops_the_history() -> None:
    sequence = SyntheticSequence("plane_static", camera=SMALL, noise=NOISELESS, frames=4)
    engine = cdf.DepthFusion(backend="cpu")
    statuses = []
    for index, frame in enumerate(sequence):
        pose = None if index == 2 else frame.input.T_world_camera
        result = engine.process(
            depth_m=frame.input.depth_m,
            intrinsics=frame.input.intrinsics,
            T_world_camera=pose,
            timestamp_s=frame.input.timestamp_s,
        )
        statuses.append(result.diagnostics.temporal_status)
    assert statuses == ["no_history", "fused", "temporal_disabled_no_pose", "no_history"]


@pytest.mark.parametrize("name", ["plane_dolly", "slanted_plane", "room_spin"])
def test_fusion_does_not_degrade_other_scenarios(name: str) -> None:
    sequence = SyntheticSequence(name, camera=SMALL, frames=12, seed=4)
    results = run(sequence)
    frames = list(sequence)
    truth = frames[-1].truth.depth_m
    mask = (results[-1].valid_mask == 1) & (frames[-1].input.depth_m > 0) & (truth > 0)
    assert rmse(results[-1].depth_m, truth, mask) <= rmse(frames[-1].input.depth_m, truth, mask)
