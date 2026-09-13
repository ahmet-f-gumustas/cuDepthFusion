from __future__ import annotations

import math

import numpy as np
import pytest

import cudepthfusion as cdf
from cudepthfusion.data import geometry_check as geo
from cudepthfusion.data.camera import PinholeCamera
from cudepthfusion.data.frames import InputFrame
from cudepthfusion.data.poses import invert_rigid
from cudepthfusion.synthetic import (
    NOISELESS,
    NoiseModel,
    Rectangle,
    Scene,
    SyntheticSequence,
    get_scenario,
    render,
    scenario_names,
)
from cudepthfusion.synthetic import trajectory as traj
from cudepthfusion.synthetic.scene import wall

SMALL = PinholeCamera.vga().scaled(0.25)  # 160 x 120, same field of view as VGA


def pixel_rays(camera: PinholeCamera) -> tuple[np.ndarray, np.ndarray]:
    u = (np.arange(camera.width) - camera.cx) / camera.fx
    v = (np.arange(camera.height) - camera.cy) / camera.fy
    return np.broadcast_to(u, (camera.height, camera.width)), np.broadcast_to(
        v[:, None], (camera.height, camera.width)
    )


def assert_rigid(transform: np.ndarray) -> None:
    rotation = transform[:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_array_equal(transform[3], [0, 0, 0, 1])


# --- camera and scene: closed-form depth -------------------------------------------------


def test_scaled_camera_keeps_the_field_of_view() -> None:
    assert (SMALL.width, SMALL.height) == (160, 120)
    assert SMALL.cx == pytest.approx(79.5) and SMALL.cy == pytest.approx(59.5)
    assert SMALL.width / (2 * SMALL.fx) == pytest.approx(640 / (2 * 525.0))


def test_fronto_parallel_plane_depth_is_exact() -> None:
    depth, surface = render(Scene((wall(1, 2.0),)), SMALL, np.eye(4))
    np.testing.assert_array_equal(depth, 2.0)
    np.testing.assert_array_equal(surface, 1)


def test_moving_towards_the_plane_reduces_depth_exactly() -> None:
    pose = traj.rigid(np.eye(3), (0.0, 0.0, 0.75))
    depth, _ = render(Scene((wall(1, 3.0),)), SMALL, pose)
    np.testing.assert_allclose(depth, 2.25, rtol=0, atol=1e-15)


def test_slanted_plane_matches_closed_form() -> None:
    theta, distance = math.radians(45), 2.5
    depth, _ = render(get_scenario("slanted_plane").scene, SMALL, np.eye(4))
    x, _ = pixel_rays(SMALL)
    expected = distance * math.cos(theta) / (math.cos(theta) - x * math.sin(theta))
    np.testing.assert_allclose(depth, expected, rtol=1e-12)


def test_step_edge_is_at_the_principal_column() -> None:
    depth, surface = render(get_scenario("step").scene, SMALL, np.eye(4))
    x, _ = pixel_rays(SMALL)
    np.testing.assert_array_equal(depth[x < 0], 1.5)
    np.testing.assert_array_equal(depth[x > 0], 2.5)
    np.testing.assert_array_equal(surface[x < 0], 1)


def test_bounded_rectangle_footprint_matches_projection() -> None:
    scene = Scene((wall(1, 1.5, half_u=0.3, half_v=0.2),))
    depth, surface = render(scene, SMALL, np.eye(4))
    x, y = pixel_rays(SMALL)
    footprint = (np.abs(x * 1.5) <= 0.3) & (np.abs(y * 1.5) <= 0.2)
    np.testing.assert_array_equal(surface == 1, footprint)
    np.testing.assert_array_equal(depth[~footprint], 0.0)  # nothing behind it: no surface


def test_negative_fy_mirrors_the_image_vertically() -> None:
    flipped = PinholeCamera(SMALL.width, SMALL.height, SMALL.fx, -SMALL.fy, SMALL.cx, SMALL.cy)
    scene = get_scenario("revealed_background").scene
    upright, _ = render(scene, SMALL, np.eye(4), time_s=0.5)
    mirrored, _ = render(scene, flipped, np.eye(4), time_s=0.5)
    np.testing.assert_array_equal(mirrored, upright[::-1])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"surface_id": 0}, "surface_id"),
        ({"u_axis": (1.0, 1.0, 0.0)}, "unit vectors"),
        ({"v_axis": (math.sqrt(0.5), math.sqrt(0.5), 0.0)}, "orthogonal"),
        ({"half_u": 0.0}, "half extents"),
    ],
)
def test_rectangle_rejects_invalid_geometry(kwargs: dict, message: str) -> None:
    base = {
        "surface_id": 1,
        "center": (0.0, 0.0, 1.0),
        "u_axis": (1.0, 0.0, 0.0),
        "v_axis": (0.0, 1.0, 0.0),
    }
    with pytest.raises(ValueError, match=message):
        Rectangle(**{**base, **kwargs})


def test_scene_rejects_duplicate_surface_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        Scene((wall(1, 1.0), wall(1, 2.0)))


# --- trajectories -------------------------------------------------------------------------


@pytest.mark.parametrize("name", scenario_names())
def test_every_trajectory_yields_rigid_poses(name: str) -> None:
    sequence = SyntheticSequence(name, camera=SMALL)
    for frame_id in (0, len(sequence) // 2, len(sequence) - 1):
        assert_rigid(sequence.pose(frame_id))


def test_linear_motion_is_pure_translation() -> None:
    pose = traj.linear((0.3, -0.1, 0.2), start=(1.0, 0.0, 0.0))(2.0)
    np.testing.assert_array_equal(pose[:3, :3], np.eye(3))
    np.testing.assert_allclose(pose[:3, 3], [1.6, -0.2, 0.4])


def test_spin_is_pure_rotation_by_rate_times_time() -> None:
    pose = traj.spin((0.0, 1.0, 0.0), 0.5, position=(0.3, 0.2, -0.5))(1.2)
    np.testing.assert_allclose(pose[:3, 3], [0.3, 0.2, -0.5])
    angle = math.acos((np.trace(pose[:3, :3]) - 1) / 2)
    assert angle == pytest.approx(0.6)


def test_rotation_about_follows_the_right_hand_rule() -> None:
    np.testing.assert_allclose(
        traj.rotation_about((0, 0, 1), math.pi / 2) @ [1, 0, 0], [0, 1, 0], atol=1e-15
    )


# --- scenarios with moving surfaces -------------------------------------------------------


def test_front_surface_enters_and_leaves_the_view() -> None:
    sequence = SyntheticSequence("front_surface_pass", camera=SMALL, noise=NOISELESS)
    first, middle, last = (sequence.frame(i).truth for i in (0, 30, 59))
    assert not first.dynamic_mask.any() and not last.dynamic_mask.any()
    assert middle.dynamic_mask.any()
    np.testing.assert_array_equal(middle.depth_m[middle.dynamic_mask], 1.5)
    np.testing.assert_array_equal(last.depth_m, 3.0)


def test_front_surface_footprint_matches_its_analytic_position() -> None:
    sequence = SyntheticSequence("front_surface_pass", camera=SMALL, noise=NOISELESS)
    truth = sequence.frame(24).truth
    x, y = pixel_rays(SMALL)
    center_x = -1.3 + 1.4 * (24 / 30.0)
    footprint = (np.abs(x * 1.5 - center_x) <= 0.3) & (np.abs(y * 1.5) <= 0.3)
    np.testing.assert_array_equal(truth.dynamic_mask, footprint)


def test_revealed_background_ends_fully_visible() -> None:
    sequence = SyntheticSequence("revealed_background", camera=SMALL, noise=NOISELESS)
    first, last = sequence.frame(0).truth, sequence.frame(59).truth
    center = (SMALL.height // 2, SMALL.width // 2)
    assert first.depth_m[center] == 1.5 and first.dynamic_mask[center]
    assert not last.dynamic_mask.any()
    np.testing.assert_array_equal(last.depth_m, 3.0)


def test_static_scenarios_have_no_dynamic_pixels() -> None:
    for name in ("plane_static", "step", "room_handheld"):
        truth = SyntheticSequence(name, camera=SMALL, noise=NOISELESS).frame(10).truth
        assert not truth.dynamic_mask.any(), name


# --- ground truth is consistent with the poses --------------------------------------------


# room_spin turns 0.5 rad/s: 20 frames (19 degrees) keep most of the ~63 degree view overlapping.
@pytest.mark.parametrize(
    ("name", "i", "j"), [("room_handheld", 3, 33), ("room_spin", 0, 20), ("plane_lateral", 5, 50)]
)
def test_warping_ground_truth_with_true_poses_is_exact(name: str, i: int, j: int) -> None:
    sequence = SyntheticSequence(name, camera=SMALL, noise=NOISELESS)
    a, b = sequence.frame(i), sequence.frame(j)
    relative = invert_rigid(b.input.T_world_camera) @ a.input.T_world_camera
    predicted, observed = geo.warp_residual(
        a.truth.depth_m, b.truth.depth_m, relative, SMALL.fx, SMALL.fy, SMALL.cx, SMALL.cy
    )
    assert predicted.size > 0.2 * SMALL.width * SMALL.height
    np.testing.assert_allclose(predicted, observed, rtol=1e-9)


# --- noise --------------------------------------------------------------------------------


def test_noise_statistics_follow_the_model() -> None:
    noise = NoiseModel(a_m=0.002, b_per_m=0.001, dropout_fraction=0.01)
    noisy = SyntheticSequence("plane_static", noise=noise, seed=7).frame(0).input.depth_m
    kept = noisy > 0
    residual = noisy[kept].astype(np.float64) - 2.0
    sigma = 0.002 + 0.001 * 4.0
    assert residual.std() == pytest.approx(sigma, rel=0.02)
    assert abs(residual.mean()) < 5 * sigma / math.sqrt(residual.size)
    assert 1 - kept.mean() == pytest.approx(0.01, abs=0.002)


def test_noise_is_reproducible_per_frame_and_seed() -> None:
    first = SyntheticSequence("room_handheld", camera=SMALL, seed=3)
    second = SyntheticSequence("room_handheld", camera=SMALL, seed=3)
    other = SyntheticSequence("room_handheld", camera=SMALL, seed=4)
    late_first = first.frame(9).input.depth_m  # accessed before frame 0 on purpose
    second.frame(0)
    np.testing.assert_array_equal(late_first, second.frame(9).input.depth_m)
    assert not np.array_equal(late_first, other.frame(9).input.depth_m)


def test_no_surface_stays_invalid_after_noise() -> None:
    scene_frame = SyntheticSequence("front_surface_pass", camera=SMALL).frame(30)
    rng = np.random.default_rng(0)
    empty = NoiseModel(0.5, 0.5, 0.0).apply(np.zeros((4, 4)), rng)
    np.testing.assert_array_equal(empty, 0.0)
    assert scene_frame.input.depth_m.dtype == np.float32


def test_noise_model_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="coefficients"):
        NoiseModel(a_m=-1.0)
    with pytest.raises(ValueError, match="dropout"):
        NoiseModel(dropout_fraction=1.0)


# --- filter input contract and engine integration ----------------------------------------


def test_input_frames_follow_the_contract_and_carry_no_truth() -> None:
    frame = SyntheticSequence("room_handheld", camera=SMALL).frame(5)
    assert isinstance(frame.input, InputFrame)
    assert frame.input.depth_m.dtype == np.float32 and frame.input.depth_m.flags.c_contiguous
    assert frame.input.T_world_camera.dtype == np.float64
    assert frame.input.T_world_camera.flags.c_contiguous
    assert frame.input.timestamp_s == pytest.approx(5 / 30.0)
    assert frame.truth.depth_m.dtype == np.float64


def test_sequence_bounds_and_names() -> None:
    sequence = SyntheticSequence("step", camera=SMALL, frames=4)
    assert len(sequence) == 4 and sequence.name == "step"
    with pytest.raises(IndexError):
        sequence.frame(4)
    with pytest.raises(KeyError, match="unknown scenario"):
        get_scenario("nope")
    assert len(set(scenario_names())) == len(scenario_names()) >= 9


@pytest.mark.parametrize("name", scenario_names())
def test_every_scenario_runs_through_the_cpu_engine(name: str) -> None:
    engine = cdf.DepthFusion(backend="cpu")
    for frame in SyntheticSequence(name, camera=SMALL, frames=3):
        result = engine.process(
            depth_m=frame.input.depth_m,
            intrinsics=frame.input.intrinsics,
            T_world_camera=frame.input.T_world_camera,
            timestamp_s=frame.input.timestamp_s,
        )
        in_range = (frame.input.depth_m >= 0.2) & (frame.input.depth_m <= 8.0)
        assert result.diagnostics.input.num_valid == int(in_range.sum())
