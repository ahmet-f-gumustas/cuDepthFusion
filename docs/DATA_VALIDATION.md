# Data validation

This page records the evidence that each dataset adapter reads its data correctly.
No fusion parameter is tuned until the relevant sequence passes here. Every number
on this page comes from a real `validate-data` run; the full machine-readable
report is `data/<dataset>/<sequence>/validation.json`, written next to the manifest.

## What is checked

`python -m cudepthfusion.cli validate-data --manifest <manifest.json>` runs three checks.
All three must pass.

1. **Frame pairing.** Clean depth, noisy depth and pose records are matched by frame
   id, never by position in a list. Duplicate ids (for example `7.png` and `007.png`)
   and file names without an integer id make pairing fail. Frames missing from one
   source are listed by id and left out of the matched set. The frame ids on disk must
   still equal the manifest's `matched_ids`.
2. **Noisy/clean consistency.**
   - For every matched frame, the median |noisy − clean| over pixels valid in both
     must stay ≤ 0.10 m. A larger value means a pairing or scale error, not sensor noise.
   - Contrast check: on every 20th matched frame, the matched pair must be closer
     than the pair (noisy *i*, clean *i* + 15) in at least 95 % of samples.
3. **Two-frame geometry (conventions).**
   - Clean frame *i* is warped into frame *j* with the ground-truth relative pose
     `inv(T_world_camera_j) @ T_world_camera_i`.
   - This is done for every candidate convention: depth kind (Z or ray distance) ×
     `fy` sign × pose direction (`T_world_camera` or `T_camera_world`) × pose-to-image
     id offset (−1, 0, +1, plus the declared one) × depth scale (5000 or 1000 per
     metre). That is 48 candidates.
   - Pairs: 8 evenly spaced source frames × baselines of 1, 5, 15 and 30 frames. Only
     frames that have a pose under every candidate offset are used, so all candidates
     are scored on identical pairs.
   - Only interior pixels are compared: no 4-neighbour depth jump > 0.05 m, dilated by
     2 px. The observed Z is sampled by bilinear interpolation of 1/Z, which is exact
     for planes under perspective projection.
   - Score: mean over pairs of the per-pair median |ΔZ|/Z. A pixel counts as an inlier
     when |ΔZ|/Z ≤ 1 %.
   - The **declared** convention (from the dataset registry) passes only if:
     - it ranks first;
     - its mean median |ΔZ|/Z ≤ 0.5 %;
     - its mean inlier fraction ≥ 90 %;
     - for every parameter, it beats the best candidate with a different value of that
       parameter on ≥ 90 % of pairs (identifiability).

The thresholds were fixed on synthetic scenes (`tests/python/fake_icl.py`) before any
real data was evaluated. The synthetic tests show that each single wrong parameter
(offset, depth kind, fy sign, pose direction, scale) is caught, and that swapped
frames and duplicate ids fail validation.

Two choices were revised during that synthetic phase:
- The first version sampled the target by nearest pixel and required the runner-up
  to be 3× worse on average. The rounding floor hid a one-frame pose offset, because a
  consistent one-frame shift barely changes relative poses on a smooth trajectory.
  That version was replaced by bilinear inverse-depth sampling and the per-pair
  win-rate rule above.
- The evaluation protocol's nearest-pixel rule still applies to the fusion core; it is
  only the evaluator's convention check that samples bilinearly.

## ICL-NUIM living room kt0 (TUM RGB-D compatible PNGs)

### Archive facts observed while downloading

| Item | Value |
|---|---|
| Official page | https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html (lists 1510 images for kt0) |
| Pose file `livingRoom0.gt.freiburg` | 108 786 B, SHA-256 `658bfae1e3118c9f97ad7c99721649e3de65b16e209c1f5271dbc2a84cb67d61`; 1508 lines, first column = integer frame id 1…1508 (not seconds) |
| Noisy archive `living_room_traj0n_frei_png.tar.gz` | 940 525 575 B, SHA-256 `886a9c56c97746fbeee4fd4214fd267821a8caa3daa47e364bd05febba144502`; 1509 depth PNGs with ids 0…1508, plus `rgb/`, `associations.txt` and an embedded `livingRoom0n.gt.freiburg` |
| Clean archive `living_room_traj0_frei_png.tar.gz` | 711 444 709 B, SHA-256 `4eca8c2e9f77c1bd7436c746d22ea6144b8c01fe9bc29a84e734186823f1f1ad`; 1509 depth PNGs with ids 0…1508, plus `rgb/`, `associations.txt` and an embedded `livingRoom0.gt.freiburg` that is byte-identical to the published pose file |

- The noisy archive's embedded pose file has the same 1508 frame ids as the
  separately published pose file. It differs numerically in 1503 rows, but only at
  print precision: at most 1.0e-5 m in translation and 3.0e-6 per quaternion
  component. The adapter uses the separately published file.
- The page's 1510 does not match the 1509 depth images or the 1508 poses. The
  pairing report below lists which ids are missing where.

### Results — `validate-data`, 2026-09-12 (55 s on the RTX 4090 Laptop host)

**Verdict: PASSED.** The declared conventions are confirmed:

| Convention | Confirmed value |
|---|---|
| Depth kind | camera Z, not ray distance |
| Units | 5000 raw units per metre; raw 0 = missing |
| Intrinsics | fx 481.20, **fy −480.00**, cx 319.50, cy 239.50. The negative fy from the reader page also holds for the TUM-compatible PNGs. |
| Pose file | stores `T_world_camera` (camera to world) |
| Pose ↔ image | image id = pose id (offset 0). Image 0 has no pose and is excluded. |
| Timestamps | `image_id / 30 Hz` |

#### Frame pairing

| Source | Count | Ids |
|---|---|---|
| clean depth | 1509 | 0…1508 |
| noisy depth | 1509 | 0…1508 |
| poses | 1508 | 1…1508 |
| **matched** | **1508** | **1…1508** |

Left out: id 0 (in clean and noisy, no pose). There are no duplicate or unparsable ids.
The official page lists 1510 images, but the archives contain 1509.

#### Noisy/clean consistency

- Frames checked: 1508 (all matched frames).
- Per-frame median |noisy − clean|: median 7.0 mm, p95 16.0 mm, max 18.0 mm (frame
  1499). All are far below the 0.10 m pairing-error threshold.
- Noisy pixels that are non-zero: 99.47 % on average.
- Contrast check: the matched pair beat the 15-frame-shifted pair in 75 of 75 samples.

#### Two-frame geometry

- 32 pairs: 8 source frames × baselines of 1, 5, 15 and 30. The first pair is 2 → 3,
  the last 1290 → 1320.
- Declared convention: mean median |ΔZ|/Z = **0.000184**, about 0.33 mm median
  absolute error; mean inlier fraction 99.97 %.

| Baseline (frames) | Mean median \|ΔZ\|/Z | Mean median \|ΔZ\| | Min inlier fraction | Mean pixels compared |
|---|---|---|---|---|
| 1 | 0.000184 | 0.326 mm | 0.998 | 289 393 |
| 5 | 0.000185 | 0.329 mm | 0.999 | 276 084 |
| 15 | 0.000184 | 0.331 mm | 0.999 | 246 386 |
| 30 | 0.000183 | 0.334 mm | 0.999 | 205 060 |

The strongest rival for each parameter:

| Parameter | Strongest rival | Its mean median \|ΔZ\|/Z | Declared wins on |
|---|---|---|---|
| pose/image id offset | offset −1 (otherwise as declared) | 0.001406 (7.6× worse) | 32 / 32 pairs |
| depth scale | 1000 per metre, offset −1 | 0.008049 | 32 / 32 |
| fy sign | fy +480, offset +1 | 0.021698 | 32 / 32 |
| depth kind | ray distance, 1000 per metre, offset +1 | 0.024010 | 32 / 32 |
| pose direction | `T_camera_world`, fy +, 1000 per metre, offset +1 | 0.099716 | 32 / 32 |

Top of the ranking, all 48 candidates scored on the same 32 pairs:

| Mean median \|ΔZ\|/Z | Inliers | Candidate |
|---|---|---|
| 0.000184 | 1.000 | Z, fy −, `T_world_camera`, offset 0, 5000/m (**declared**) |
| 0.001406 | 0.958 | Z, fy −, `T_world_camera`, offset −1, 5000/m |
| 0.001471 | 0.953 | Z, fy −, `T_world_camera`, offset +1, 5000/m |
| 0.008049 | 0.736 | Z, fy −, `T_world_camera`, offset −1, 1000/m |
| 0.008169 | 0.734 | Z, fy −, `T_world_camera`, offset 0, 1000/m |

#### Observations for later phases

- The residual floor stays at about 0.33 mm across all baselines. That is consistent
  with the 0.2 mm quantisation step of 16-bit depth at 5000 units per metre.
- The noisy depth contains a few implausibly small values (e.g. 0.009 m in the first
  frames). With `depth.min_m = 0.2` the sanitizer marks them invalid and counts them
  in `num_below_min`. The P5 coverage metric must use the same sanitized definition
  of "raw-valid" (see the open item in PROGRESS.md).
- kt1–kt3 have not been downloaded yet. Each must pass `validate-data` on its own
  before it is used. kt1 is the parameter-selection sequence.
