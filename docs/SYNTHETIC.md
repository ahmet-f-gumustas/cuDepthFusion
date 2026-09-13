# Synthetic scenes (analytic oracle)

`cudepthfusion.synthetic` generates depth sequences whose ground truth is exact:
- scenes are planar rectangles, and depth comes from closed-form ray casting;
- camera trajectories are analytic rigid motions;
- sensor noise is seeded per frame.

No third-party asset, network or camera is involved. These sequences are the oracle for
the geometry, filter and fusion tests of the later phases.

## Conventions

| Item | Value |
|---|---|
| Axes | x right, y down, z forward. For static-camera scenarios the world frame equals the camera frame at t = 0. |
| Default camera | 640 × 480, fx = fy = 525 px, cx = 319.5, cy = 239.5 (`PinholeCamera.vga()`); `scaled(f)` keeps the field of view |
| Frames | 60 per scenario at 30 Hz; `timestamp_s = frame_id / 30` |
| Poses | `T_world_camera`, float64 |
| Noise | Gaussian, σ(z) = a + b·z² with a = 0.002 m and b = 0.001 m⁻¹, plus 1 % random dropouts |
| Reproducibility | frame *i* draws its noise from `default_rng([seed, i])`, so every frame is identical for any access order |

The noise has the same functional form the filter assumes (spec 5.2), so these scenes
test the filter against its own model. They make no claim about any real sensor. Use
`NOISELESS` for zero-noise tests.

Ground truth and filter input are separate objects, as for real data:
- `frame.input` is an `InputFrame` holding noisy float32 depth, intrinsics, the pose and
  the timestamp.
- `frame.truth` is evaluator-only. It holds exact float64 Z (0 where no surface is hit),
  the surface id per pixel and the mask of pixels that show a moving surface.

```python
from cudepthfusion.synthetic import SyntheticSequence

for frame in SyntheticSequence("front_surface_pass", seed=0):
    filter_input = frame.input      # what a filter may see
    truth = frame.truth             # evaluator only
```

## Scenarios

| Name | Scene | Camera motion | Purpose |
|---|---|---|---|
| `plane_static` | fronto-parallel plane at 2.0 m | static | bias and temporal variance on a plane (spec 9.2) |
| `plane_lateral` | fronto-parallel plane at 2.0 m | +x at 0.3 m/s | pose-compensated fusion under pure x motion (spec 9.1) |
| `plane_dolly` | fronto-parallel plane at 3.0 m | +z at 0.3 m/s | pure Z motion (spec 9.1) |
| `slanted_plane` | plane at 45° about y through z = 2.5 m | +x at 0.1 m/s | bias on slanted surfaces |
| `step` | near half-plane (x < 0) at 1.5 m over a plane at 2.5 m | static | front and back must not blend at a depth edge (spec 9.2) |
| `front_surface_pass` | 0.6 m square at 1.5 m moving +x at 1.4 m/s in front of a wall at 3.0 m | static | the square enters and leaves the view; no ghosting afterwards (spec 9.2, 10.4) |
| `revealed_background` | 1.2 m occluder at 1.5 m moving +y at 0.9 m/s in front of a wall at 3.0 m | static | the occluder leaves and reveals the wall; stale foreground must not persist (spec 10.4) |
| `room_handheld` | closed asymmetric box room | smooth 6-DoF handheld motion | general motion |
| `room_spin` | closed asymmetric box room | yaw at 0.5 rad/s about the camera centre | pure rotation (spec 9.1) |

## Export

```bash
python scripts/make_synthetic.py --suite all --output data/synthetic
# options: --suite <names...>, --frames N, --seed S, --scale F (resolution relative to VGA), --noiseless
```

Each scenario is written in the same TUM-PNG layout as an extracted ICL-NUIM sequence,
so the same adapter reads it:

```
<name>/clean/depth/<id>.png     uint16, 5000 units per metre (0.2 mm step), 0 = no surface
<name>/noisy/depth/<id>.png     uint16, the filter input
<name>/poses/<name>.freiburg    "id tx ty tz qx qy qz qw" (id = image id, T_world_camera)
<name>/oracle.npz               surface_id (uint16) and dynamic_mask (bool) per frame
<name>/manifest.json            dataset manifest schema plus a "generator" block
```

- The manifest records every generator parameter: scenario, seed, frame count, frame
  rate, camera and noise. The exact float ground truth is therefore not stored; it is
  regenerated from the manifest.
- Exports are deterministic: two runs produce identical per-directory hashes.
- An export only ever replaces a directory that holds a previous synthetic export.
- Measured on the RTX 4090 Laptop host (CPU): the full suite at VGA with 60 frames per
  scenario takes 16.7 s and 264 MB.

## Validation of the exports

`validate-data` works on exported scenarios like on real data. It can only confirm pose
conventions when the camera moves.

| Scenario (VGA, 60 frames, seed 0) | Result |
|---|---|
| `room_handheld` | **PASSED.** Declared = best candidate (Z, fy +, `T_world_camera`, offset 0, 5000/m); mean median \|ΔZ\|/Z 1.40e-05; every rival loses on 32/32 pairs (the offset rival is at 6.55e-04); noisy/clean median difference 8.3 mm |
| `plane_static` | **Fails, as expected.** With a static camera and a static scene every candidate explains the data exactly (residual 0), so no parameter is identifiable. The shifted-frame contrast check cannot separate identical frames either. |

For static-camera exports, the evidence that the conventions are right is the generator's
own test suite (below), not `validate-data`. Loading such an export as filter input
therefore needs an explicit `IclInputSequence(manifest, require_validated=False)`.

## What the generator tests prove (`tests/python/test_synthetic*.py`)

- **Closed-form depth:**
  - a fronto-parallel plane gives exactly Z = d;
  - moving the camera towards it gives exactly d − Δ;
  - the slanted plane matches Z = d·cos θ / (cos θ − x̂·sin θ) to 1e-12;
  - the step edge falls exactly at the principal column;
  - a bounded rectangle's footprint equals its analytic projection;
  - a negative fy mirrors the image.
- **Trajectories:** every pose of every scenario is rigid. Linear motion is pure
  translation; spin is pure rotation by rate × time.
- **Moving surfaces:** the square's footprint matches its analytic position. It is absent
  in the first and last frames and present in the middle. The occluder is gone by the
  last frame.
- **Consistency:** warping exact ground truth with the true relative pose reproduces the
  target depth to 1e-9 (room_handheld, room_spin, plane_lateral).
- **Noise:** the empirical σ is within 2 % of the model, the mean is within 5 standard
  errors of zero, and dropouts are 1 % ± 0.2 %. Noise is reproducible by seed and frame,
  and pixels with no surface stay invalid.
- **Integration:** the first frames of every scenario run through the CPU engine, with
  valid counts as expected.
- **Export:**
  - the layout and manifest are complete;
  - the adapter reads back the generator's input within the 0.1 mm quantisation and its
    poses to 1e-12;
  - exports are deterministic;
  - a foreign directory is refused;
  - `validate-data` passes on a moving-camera export and correctly fails on a static
    one.
