# The CPU pipeline

What `DepthFusion::process()` does with one frame, in order. The CUDA backend (P4) must
reproduce this exactly; the CPU code is the reference that defines correctness.

## 1. Validate and sanitise

Shape, intrinsics (fx > 0, fy ≠ 0, both finite) and a finite timestamp are checked before
any state changes, so a rejected frame leaves the engine untouched. A pixel is valid only
if its depth is finite and inside `depth.min_m … depth.max_m`; everything else becomes 0
and is counted by category (missing, non-finite, below, above).

## 2. Mask-aware bilateral filter (spec 5.2)

```
w(p, q) = exp(-||p - q||² / 2σ_xy²) · exp(-(D(p) - D(q))² / 2σ_depth²)
D_spatial(p) = Σ_q w(p, q) D(q) / Σ_q w(p, q)
```

Only valid neighbours contribute and the centre always does. Out-of-image neighbours are
skipped, never clamped: clamping would count a border pixel twice. An invalid centre stays
invalid. The measurement variance is **not** divided by the neighbour count — that would
invent confidence the data does not have.

Measurement variance uses the spec's model, `σ(z) = a + b·z²`, evaluated on the filtered
depth.

## 3. Transport the previous result (spec 5.3, 5.4)

`T_c_p = inverse(T_world_camera_current) · T_world_camera_previous` is built in float64 and
narrowed to float32, because the per-pixel arithmetic has to match the future CUDA kernel.
Every valid history pixel is projected forward:

```
ray_p = [(u - cx_p)/fx_p, (v - cy_p)/fy_p, 1]
P_c   = R_c_p · (D_prev · ray_p) + t_c_p
u_c   = fx_c · P_c.x / P_c.z + cx_c        (nearest pixel: floor(coord + 0.5))
```

Points with `z_c ≤ 0`, non-finite values and off-screen projections are counted, not used.
Where several sources land on one pixel, the nearest wins. That is decided by a 64-bit key:

```
key = [float32 bits of z_c | source pixel index]
```

For positive finite floats the IEEE bit pattern grows with the value, so the smallest key
is the nearest surface, and at equal depth the lowest source index wins. On the GPU a
64-bit `atomicMin` gives the same answer without a race. Depth, variance and age of a pixel
always come from the winning source — they are read back through its index, never mixed.

Nearest-pixel transport leaves holes. They are measured and left empty; no bilinear
blending, which would fuse a front and a back surface into a value that exists nowhere.

The prior's uncertainty grows on the way:

```
j_z     = third row of R_c_p · ray_p
V_prior = j_z² · V_previous + Q
Q       = q0 + q_translation·||t||² + q_rotation·angle²
```

## 4. Compatibility gate (spec 5.5)

```
τ = tau_abs + k_sigma · sqrt(V_cur + V_prior)
compatible ⇔ |D_cur - D_prior| ≤ τ
```

| Situation | What happens |
|---|---|
| current valid, no prior | current only |
| current and prior valid, compatible | confidence-weighted merge |
| current nearer and incompatible | occlusion or a new front surface: prior dropped |
| current farther and incompatible | a surface moved away: prior dropped |
| current invalid | invalid by default; `fill_holes` may carry the history for a while |
| both invalid | invalid |

The prior is never blended into an incompatible measurement. The gate is deliberately
narrow: it uses only the transported uncertainty (see below).

## 5. Confidence-weighted merge (spec 5.6)

```
P_cur   = 1 / max(V_cur, variance_floor)
P_prior = min(history_decay / V_prior_merge, max_history_ratio · P_cur)
D_out   = (P_cur·D_cur + P_prior·D_prior) / (P_cur + P_prior)
V_out   = max(1 / (P_cur + P_prior), variance_floor)
```

The caps stop re-used and spatially filtered observations from piling up into
overconfidence. `V_out` is not a full Bayesian posterior, and `confidence_score =
1/(1 + V_out/variance_reference)` is a display score, not a probability.

**`V_prior_merge` is not the gate's variance.** Nearest-pixel transport can land half a
pixel off, which on a surface with slope `g` metres per pixel is a depth error of `0.5·g`.
So the merge charges that to the prior:

```
V_prior_merge = V_prior + q_gradient · g²        (q_gradient = 0.25 = (0.5 px)²)
```

where `g` is the largest depth step to a valid 4-neighbour of the current measurement. The
gate keeps using `V_prior` alone. Both halves matter:

- without the term in the merge, fusion made a 45° slanted plane *worse* than its raw
  input (RMSE 17.7 mm vs 15.5 mm), because the mis-sampled prior is a systematic error;
- with the term in the gate as well, τ grew to metres at depth edges and occlusions stopped
  being rejected — the exact behaviour the gate exists to prevent.

## 6. History-only mode (spec 5.7, off by default)

With `fusion.fill_holes` a pixel without a current measurement keeps the transported
history, `source_mask = 3`, and its age grows by one. It expires at
`max_history_age_frames`. Age is only reset by a current measurement.

## 7. State and reset

The history (depth, variance, validity, age, intrinsics, pose) is read-only while a frame
is processed and replaced afterwards. It is dropped on any sequence discontinuity —
resolution or calibration change, a timestamp that does not increase, a frame gap, an
explicit `reset()` — and whenever a frame has no usable pose, because such a frame could
never be reprojected later. A missing pose disables the temporal stage; it is only treated
as identity when `fusion.assume_static_camera` says so.

## Diagnostics

Every frame reports the reset reason, the temporal status, whether the spatial filter ran,
input pixel counts by category, and the fusion counters: how many prior pixels were
projected, how many were behind the camera or off screen, how many survived the z-buffer,
how many pixels were fused, kept current-only, rejected (nearer/farther), filled from
history or expired, plus the mean prior weight.

## The CUDA backend (P4)

The GPU path runs the same stages in the same order, one thread per pixel, six kernels
sequenced in a single stream: sanitize → bilateral → variance → project → gather → fuse.
`DepthFusion::process()` stays synchronous: it stages the frame in pinned memory and uploads
it, enqueues the kernels, downloads the six output arrays and the counters into one pinned
block, and copies each array into its owned result as soon as its own transfer has finished.
The output buffers then become the history by swapping places with it; nothing is copied on
the device. Every stage is timed with CUDA events and reported in `Diagnostics.device`.

- **The z-buffer is a 64-bit `atomicMin`** on the same key the CPU builds, so "nearest wins,
  ties fall to the lowest source index" needs no lock and no second pass.
- **Buffers only ever grow.** They are allocated on the first frame and on a resolution
  change; the steady frame loop performs no allocation. Verified by watching free device
  memory over 200 frames with resets and resize cycles: it does not move.
- **Precision is split on purpose.** Input classification, the noise model and the variance
  transport run in double on both backends; they are cheap. The bilateral filter, the gate
  and the merge run in float on the GPU (spec 7.2) and in double on the CPU reference. Until
  P7 the gate and merge were double on the GPU too, which put the fuse kernel on the FP64
  pipe (1/64 of the FP32 rate on consumer GPUs) and made it 85 % of the kernel time. Moving
  it to float cut it from 271 to 15 µs, and re-measured parity and quality did not move
  (see [PERFORMANCE.md](PERFORMANCE.md)). Every CUDA call is checked and reported as
  `CudaError`.
- Measured and rejected in P7: a shared-memory tile for the filter (bit-identical output, no
  speed-up: the kernel is bound by its 25 `expf` per pixel, not by memory). Not used: CUDA
  Graphs, async APIs, FP16, fast-math and approximate `exp`.

### What parity means in practice

| Case | Result |
|---|---|
| Noise-free sequence with camera motion | `valid_mask`, `source_mask` and `history_age` identical; depth within 1e-5 m |
| Noisy sequence, static camera (identity relative pose, so no z-buffer contention) | masks identical; depth within 1e-5 m |
| Noisy sequence with camera motion | validity always identical; about 1 pixel in 10 000 diverges, worst case 9 mm |

The last row is not a defect to be hidden. Two transported pixels can land on the same
target with depths equal to within a float ULP, and a pixel's depth difference can sit
exactly on τ. The two backends then pick different winners or different branches, and
because the filter is recursive, the difference is carried in the history for a few frames.
The tests bound how often that happens and how large it gets rather than averaging it away.
