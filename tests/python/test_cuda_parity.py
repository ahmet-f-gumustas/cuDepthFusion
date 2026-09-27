"""CPU/GPU parity through the Python API, on the analytic scenes (spec 9.3)."""

from __future__ import annotations

import numpy as np
import pytest

import cudepthfusion as cdf
from cudepthfusion.data.camera import PinholeCamera
from cudepthfusion.synthetic import NOISELESS, SyntheticSequence

SMALL = PinholeCamera.vga().scaled(0.25)
DEPTH_ATOL = 1e-5  # metres, the spec's parity tolerance
pytestmark = pytest.mark.gpu


def run(sequence: SyntheticSequence, backend: str) -> list:
    engine = cdf.DepthFusion(backend=backend)
    return [
        engine.process(
            depth_m=frame.input.depth_m,
            intrinsics=frame.input.intrinsics,
            T_world_camera=frame.input.T_world_camera,
            timestamp_s=frame.input.timestamp_s,
        )
        for frame in sequence
    ]


@pytest.mark.parametrize("name", ["room_handheld", "front_surface_pass", "step"])
def test_backends_agree_without_noise(name: str) -> None:
    sequence = SyntheticSequence(name, camera=SMALL, noise=NOISELESS, frames=8)
    for cpu, gpu in zip(run(sequence, "cpu"), run(sequence, "cuda"), strict=True):
        np.testing.assert_array_equal(cpu.valid_mask, gpu.valid_mask)
        np.testing.assert_array_equal(cpu.source_mask, gpu.source_mask)
        np.testing.assert_array_equal(cpu.history_age, gpu.history_age)
        valid = cpu.valid_mask == 1
        np.testing.assert_allclose(cpu.depth_m[valid], gpu.depth_m[valid], atol=DEPTH_ATOL, rtol=0)
        np.testing.assert_allclose(
            cpu.variance_m2[valid], gpu.variance_m2[valid], rtol=1e-4, atol=1e-12
        )
        assert cpu.diagnostics.temporal_status == gpu.diagnostics.temporal_status
        assert cpu.diagnostics.fusion.fused == gpu.diagnostics.fusion.fused
        assert cpu.diagnostics.input == gpu.diagnostics.input


def test_noisy_static_camera_agrees_strictly() -> None:
    # A static camera means an identity relative pose: every history pixel transports to its own
    # pixel, so there is no z-buffer contention and no decision can go either way.
    sequence = SyntheticSequence("plane_static", camera=SMALL, frames=8, seed=5)
    for cpu, gpu in zip(run(sequence, "cpu"), run(sequence, "cuda"), strict=True):
        np.testing.assert_array_equal(cpu.valid_mask, gpu.valid_mask)
        np.testing.assert_array_equal(cpu.source_mask, gpu.source_mask)
        valid = cpu.valid_mask == 1
        np.testing.assert_allclose(cpu.depth_m[valid], gpu.depth_m[valid], atol=DEPTH_ATOL, rtol=0)


def test_noisy_moving_camera_diverges_only_on_ties() -> None:
    """Where two decisions are a coin flip, float rounding can pick differently.

    Two transported pixels can land on the same target with depths equal to within a float
    ULP, and a pixel's depth difference can sit exactly on tau. The backends then choose
    different winners or different branches, and because the filter is recursive, that
    difference is carried in the history for a few frames. The point of this test is to bound
    how often it happens and how large it gets, not to pretend it does not.
    """
    sequence = SyntheticSequence("room_handheld", camera=SMALL, frames=8, seed=5)
    diverging = 0
    pixels = 0
    worst = 0.0
    for cpu, gpu in zip(run(sequence, "cpu"), run(sequence, "cuda"), strict=True):
        np.testing.assert_array_equal(cpu.valid_mask, gpu.valid_mask)  # validity never ties
        difference = np.abs(cpu.depth_m - gpu.depth_m)
        over_tolerance = (cpu.valid_mask == 1) & (difference > DEPTH_ATOL)
        diverging += int(np.count_nonzero(over_tolerance)) + int(
            np.count_nonzero(cpu.source_mask != gpu.source_mask)
        )
        pixels += cpu.depth_m.size
        worst = max(worst, float(difference[cpu.valid_mask == 1].max()))
    assert diverging / pixels < 0.001, f"{diverging} of {pixels} pixels diverged"
    assert worst < 0.02, f"worst divergence {worst * 1000:.2f} mm"


def test_cuda_engine_survives_reset_and_resolution_changes() -> None:
    engine = cdf.DepthFusion(backend="cuda")
    statuses = []
    for scale, frames in ((0.25, 3), (0.5, 3), (0.25, 3)):
        camera = PinholeCamera.vga().scaled(scale)
        for index, frame in enumerate(
            SyntheticSequence("plane_lateral", camera=camera, frames=frames)
        ):
            if index == 1 and scale == 0.5:
                engine.reset()
            result = engine.process(
                depth_m=frame.input.depth_m,
                intrinsics=frame.input.intrinsics,
                T_world_camera=frame.input.T_world_camera,
                timestamp_s=frame.input.timestamp_s,
            )
            statuses.append((result.diagnostics.reset_reason, result.diagnostics.temporal_status))
    assert statuses[0] == ("first_frame", "no_history")
    assert ("resolution_change", "no_history") in statuses
    assert ("explicit", "no_history") in statuses
