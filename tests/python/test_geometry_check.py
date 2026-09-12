from __future__ import annotations

import numpy as np
import pytest
from synthetic_scene import camera_pose, render_z

from cudepthfusion.data import geometry_check as geo
from cudepthfusion.data.poses import invert_rigid

W, H, FX, FY, CX, CY = 64, 48, 60.0, -60.0, 31.5, 23.5


def test_ray_norms_are_one_at_the_principal_point_and_grow_outwards() -> None:
    norms = geo.ray_norms(W, H, FX, FY, CX, CY)
    assert norms.shape == (H, W)
    assert norms.min() >= 1.0
    corner = np.sqrt(1 + (CX / FX) ** 2 + (CY / FY) ** 2)
    assert norms[0, 0] == pytest.approx(corner)


def test_interior_mask_drops_invalid_pixels_and_depth_jumps() -> None:
    z = np.full((20, 20), 2.0)
    z[:, 10:] = 3.0  # a 1 m step between columns 9 and 10
    z[5, 5] = 0.0  # a hole
    mask = geo.interior_mask(z)
    assert not mask[:, 7:13].any()  # the step, dilated by EDGE_DILATE_PX
    assert not mask[5, 5] and not mask[5, 3]  # the hole and its dilated neighbourhood
    assert mask[15, 2] and mask[15, 17]


def test_warp_with_true_relative_pose_is_exact_on_planes() -> None:
    pose_a, pose_b = camera_pose(3), camera_pose(9)
    z_a = render_z(pose_a, W, H, FX, FY, CX, CY)
    z_b = render_z(pose_b, W, H, FX, FY, CX, CY)
    predicted, observed = geo.warp_residual(z_a, z_b, invert_rigid(pose_b) @ pose_a, FX, FY, CX, CY)
    # Edge dilation and the 6-frame motion leave about half of this small image comparable.
    assert predicted.size > 0.3 * W * H
    np.testing.assert_allclose(predicted, observed, rtol=1e-9)


def test_warp_with_identity_pose_returns_the_same_depth() -> None:
    z = render_z(camera_pose(0), W, H, FX, FY, CX, CY)
    predicted, observed = geo.warp_residual(z, z, np.eye(4), FX, FY, CX, CY)
    np.testing.assert_allclose(predicted, observed, rtol=1e-12)


def test_select_pairs_is_deterministic_and_respects_baselines() -> None:
    ids = list(range(100))
    pairs = geo.select_pairs(ids, baselines=(1, 30), source_frames=4)
    assert pairs == geo.select_pairs(ids, baselines=(1, 30), source_frames=4)
    assert {t - s for s, t in pairs} == {1, 30}
    assert all(t in ids for _, t in pairs)


def test_all_candidates_cover_every_combination() -> None:
    candidates = geo.all_candidates()
    assert len(candidates) == len(set(candidates)) == 2 * 2 * 2 * 3 * 2


def _score(candidate: geo.Candidate, medians: list[float]) -> geo.CandidateScore:
    pairs = tuple(geo.PairResidual(i, i + 1, 100, m, m, 1.0) for i, m in enumerate(medians))
    return geo.CandidateScore(candidate, float(np.mean(medians)), 1.0, len(pairs), pairs)


def test_paired_win_rate_counts_pairs_not_averages() -> None:
    base = geo.Candidate("z", -1, "T_world_camera", 0, 5000.0)
    other = geo.Candidate("z", -1, "T_world_camera", 1, 5000.0)
    declared = _score(base, [0.001, 0.001, 0.001, 0.1])  # worse on average,
    rival = _score(other, [0.002, 0.002, 0.002, 0.002])  # but better on 3 of 4 pairs
    rate, compared = geo.paired_win_rate(declared, rival)
    assert (rate, compared) == (0.75, 4)


def test_judge_fails_cleanly_when_declared_was_not_scored() -> None:
    declared = geo.Candidate("z", -1, "T_world_camera", 5, 5000.0)
    scored = geo.Candidate("z", -1, "T_world_camera", 0, 5000.0)
    verdict = geo.judge([_score(scored, [0.001, 0.001])], declared)
    assert not verdict.passed and verdict.declared is None
    assert "could not be scored" in verdict.reasons[0]


def test_judge_flags_unidentifiable_parameter() -> None:
    declared = geo.Candidate("z", -1, "T_world_camera", 0, 5000.0)
    rival = geo.Candidate("z", -1, "T_world_camera", 1, 5000.0)
    ranked = [
        _score(declared, [0.0010, 0.0010, 0.0030, 0.0030]),
        _score(rival, [0.0011, 0.0011, 0.0020, 0.0020]),
    ]
    verdict = geo.judge(ranked, declared)
    assert not verdict.passed
    assert any("pose_id_offset is not identifiable" in r for r in verdict.reasons)
