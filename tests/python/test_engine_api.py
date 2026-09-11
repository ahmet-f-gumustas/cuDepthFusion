from __future__ import annotations

import threading

import numpy as np
import pytest

import cudepthfusion as cdf


def plane(height: int, width: int, depth_m: float = 2.0) -> np.ndarray:
    return np.full((height, width), depth_m, dtype=np.float32)


def run(engine: cdf.DepthFusion, depth: np.ndarray, intrinsics, pose, t: float = 0.0):
    return engine.process(depth_m=depth, intrinsics=intrinsics, T_world_camera=pose, timestamp_s=t)


def test_first_frame_returns_current_measurement(cpu_engine, intrinsics, identity_pose) -> None:
    result = run(cpu_engine, plane(48, 64), intrinsics, identity_pose)

    config = cpu_engine.config.core
    sigma = config.noise.a_m + config.noise.b_per_m * 4.0
    expected_variance = np.float32(sigma * sigma)
    assert result.depth_m.dtype == np.float32 and result.depth_m.shape == (48, 64)
    assert result.valid_mask.dtype == np.uint8 and result.valid_mask.all()
    assert result.source_mask.dtype == np.uint8 and (result.source_mask == 1).all()
    assert result.history_age.dtype == np.uint16 and not result.history_age.any()
    np.testing.assert_array_equal(result.depth_m, 2.0)
    np.testing.assert_allclose(result.variance_m2, expected_variance, rtol=1e-6)
    np.testing.assert_allclose(
        result.confidence_score,
        1.0 / (1.0 + expected_variance / config.fusion.variance_reference_m2),
        rtol=1e-6,
    )
    assert result.diagnostics.reset_reason == "first_frame"
    assert result.diagnostics.temporal_status == "no_history"
    assert result.diagnostics.input.num_valid == 48 * 64


def test_invalid_pixels_are_zero_everywhere(cpu_engine, intrinsics, identity_pose) -> None:
    depth = plane(2, 4)
    depth[0, :] = [np.nan, np.inf, -np.inf, 0.0]
    depth[1, :2] = [0.1, 9.0]  # below min / above max
    result = run(cpu_engine, depth, intrinsics, identity_pose)

    expected_valid = np.array([[0, 0, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    np.testing.assert_array_equal(result.valid_mask, expected_valid)
    invalid = expected_valid == 0
    for name in ("depth_m", "variance_m2", "confidence_score", "source_mask"):
        assert not getattr(result, name)[invalid].any(), name
    stats = result.diagnostics.input
    assert (stats.num_nonfinite, stats.num_zero, stats.num_below_min, stats.num_above_max) == (
        3,
        1,
        1,
        1,
    )


@pytest.mark.parametrize(("height", "width"), [(1, 1), (1, 17), (9, 1), (7, 13)])
def test_shapes_not_multiple_of_block_size(cpu_engine, intrinsics, identity_pose, height, width):
    result = run(cpu_engine, plane(height, width), intrinsics, identity_pose)
    assert result.depth_m.shape == (height, width)
    assert result.diagnostics.input.num_valid == height * width


def test_results_are_owned_copies(cpu_engine, intrinsics, identity_pose) -> None:
    depth = plane(4, 5)
    first = run(cpu_engine, depth, intrinsics, identity_pose, t=0.0)
    depth[:] = 3.0
    second = run(cpu_engine, depth, intrinsics, identity_pose, t=0.033)

    np.testing.assert_array_equal(first.depth_m, 2.0)
    np.testing.assert_array_equal(second.depth_m, 3.0)
    assert not np.shares_memory(first.depth_m, depth)


def test_non_contiguous_depth_is_rejected_without_copy(cpu_engine, intrinsics, identity_pose):
    strided = plane(8, 10)[:, ::2]
    with pytest.raises(cdf.InvalidInputError, match="C-contiguous"):
        run(cpu_engine, strided, intrinsics, identity_pose)


@pytest.mark.parametrize(
    ("depth", "message"),
    [
        (np.full((4, 4), 2.0, dtype=np.float64), "float32"),
        (np.full((4, 4), 2000, dtype=np.uint16), "float32"),
        (np.full((2, 4, 4), 2.0, dtype=np.float32), "2-D"),
        (np.zeros((0, 4), dtype=np.float32), "outside"),
        ([[2.0, 2.0]], "numpy.ndarray"),
    ],
)
def test_malformed_depth_is_rejected(cpu_engine, intrinsics, identity_pose, depth, message):
    with pytest.raises(cdf.InvalidInputError, match=message):
        run(cpu_engine, depth, intrinsics, identity_pose)


@pytest.mark.parametrize(
    ("pose", "message"),
    [
        (np.eye(4, dtype=np.float32), "float64"),
        (np.eye(3), r"shape \(4, 4\)"),
        (np.asfortranarray(np.arange(16, dtype=np.float64).reshape(4, 4)), "C-contiguous"),
    ],
)
def test_malformed_pose_is_rejected(cpu_engine, intrinsics, pose, message) -> None:
    with pytest.raises(cdf.InvalidInputError, match=message):
        run(cpu_engine, plane(4, 4), intrinsics, pose)


def test_numerically_invalid_pose_disables_temporal(cpu_engine, intrinsics) -> None:
    pose = np.eye(4)
    pose[0, 0] = 2.0
    result = run(cpu_engine, plane(4, 4), intrinsics, pose)
    assert result.diagnostics.temporal_status == "temporal_disabled_invalid_pose"
    assert any("pose rejected" in note for note in result.diagnostics.notes)


def test_missing_pose_disables_temporal(cpu_engine, intrinsics) -> None:
    result = run(cpu_engine, plane(4, 4), intrinsics, None)
    assert result.diagnostics.temporal_status == "temporal_disabled_no_pose"


def test_invalid_intrinsics_raise(cpu_engine, identity_pose) -> None:
    with pytest.raises(cdf.InvalidInputError, match="intrinsics"):
        run(cpu_engine, plane(4, 4), cdf.Intrinsics(fx=0.0, fy=1.0, cx=0.0, cy=0.0), identity_pose)


def test_reset_and_sequence_rules(cpu_engine, intrinsics, identity_pose) -> None:
    reasons = [
        run(cpu_engine, plane(4, 4), intrinsics, identity_pose, t).diagnostics.reset_reason
        for t in (0.0, 0.033, 0.033, 1.0)
    ]
    assert reasons == ["first_frame", "none", "timestamp_not_increasing", "frame_gap"]
    cpu_engine.reset()
    assert run(cpu_engine, plane(4, 4), intrinsics, identity_pose).diagnostics.reset_reason == (
        "explicit"
    )


def test_process_arguments_are_keyword_only_and_pose_is_required(cpu_engine, intrinsics) -> None:
    with pytest.raises(TypeError):
        cpu_engine.process(plane(4, 4), intrinsics, None, 0.0)
    with pytest.raises(TypeError):
        cpu_engine.process(depth_m=plane(4, 4), intrinsics=intrinsics, timestamp_s=0.0)


def test_cuda_backend_never_falls_back_to_cpu() -> None:
    # Without CUDA support, or before the P4 kernels exist, the CUDA backend must refuse.
    with pytest.raises(cdf.BackendUnavailableError):
        cdf.DepthFusion(backend="cuda")


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="backend"):
        cdf.DepthFusion(backend="opencl")


def test_concurrent_calls_on_one_engine_are_serialised(cpu_engine, intrinsics, identity_pose):
    indices: list[int] = []
    lock = threading.Lock()

    def worker(offset: int) -> None:
        for step in range(20):
            result = run(cpu_engine, plane(32, 32), intrinsics, identity_pose, offset + step * 0.01)
            with lock:
                indices.append(result.diagnostics.frame_index)

    threads = [threading.Thread(target=worker, args=(i * 100.0,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(indices) == list(range(80))
