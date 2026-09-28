# Results

Every number here comes from a real run in this repository. The run folders
(`summary.json`, `per_frame.csv`, `config_resolved.yaml`, `environment.json`, figures) are
written by the commands below and carry the git commit, the dataset manifest hashes and the
thresholds used.

```bash
# parameter selection happened on the validation split first (kt1)
python -m cudepthfusion.cli evaluate --split validation --data-root data/icl \
    --config configs/icl.yaml --output runs/validation --backend cuda --stability --plots
python -m cudepthfusion.cli ablate   --split validation --data-root data/icl \
    --config configs/icl.yaml --output runs/ablation   --backend cuda --plots
# the test split was then run once, with the frozen configuration
python -m cudepthfusion.cli evaluate --split test --data-root data/icl \
    --config configs/icl.yaml --output runs/test --backend cuda --stability --plots
```

Hardware: RTX 4090 Laptop, CUDA backend. Splits follow the spec: kt0 development, kt1
validation (parameter selection), kt2 and kt3 test. Metrics are computed on the fixed mask
`clean-valid AND raw-input-valid`, so nothing improves by dropping pixels; coverage says what
each method failed to produce inside that mask.

## Methods compared

| Name | Method |
|---|---|
| B0 | raw noisy depth, sanitised only |
| B1 | mask-aware bilateral filter only |
| B2 | per-pixel EMA, no motion compensation |
| B3 | reprojection + gate + fixed temporal weight (0.5) |
| B4 | full cuDepthFusion |

## Test split (kt2 + kt3, 2120 frames, frozen configuration)

| Method | RMSE | median \|e\| | p90 \|e\| | bad pixels | edge MAE | coverage |
|---|---|---|---|---|---|---|
| B0 | 208.1 mm | 9.00 mm | 26.00 mm | 9.55 % | 436.6 mm | 100 % |
| B1 | 207.8 mm | 8.34 mm | 18.39 mm | 4.11 % | 435.1 mm | 100 % |
| B2 | 164.5 mm | 9.98 mm | 41.82 mm | 19.61 % | 371.4 mm | 100 % |
| B3 | 207.7 mm | 8.41 mm | 17.35 mm | 3.72 % | 435.1 mm | 100 % |
| **B4** | **207.7 mm** | **8.43 mm** | **17.02 mm** | **3.83 %** | **435.1 mm** | **100 %** |

Per sequence, B0 → B4: kt2 p90 35.0 → 22.9 mm and bad pixels 12.04 → 4.86 %; kt3 p90
23.0 → 15.1 mm and bad pixels 7.74 → 3.09 %. No method loses a single raw-valid pixel.

B2 has the *lowest* RMSE and the *worst* robust metrics. Averaging without motion compensation
smears the large errors instead of removing them. That alone is a warning about reading RMSE
on this dataset as a quality ranking.

## Temporal stability (spec 10.3)

Fixed world points are picked from the clean depth, followed with the ground-truth poses and
kept only for frames where they are genuinely visible. The number is the standard deviation
of each point's depth error over time, median over points (lower is steadier).

| Sequence | B0 | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|
| kt2 | 10.38 mm | 3.44 mm | 9.30 mm | 2.28 mm | **1.88 mm** |
| kt3 | 7.25 mm | 2.28 mm | 5.87 mm | 1.49 mm | **1.40 mm** |

This is where the method does what it was built for: a fixed piece of the world stops
flickering. B4 is 5.2× steadier than the raw input on kt3 and 5.5× on kt2, and steadier than
the spatial filter alone by a further 1.6×.

## Ablation on the validation split (kt1, 965 frames)

| Variant | p90 \|e\| | bad pixels | interior RMSE | edge MAE |
|---|---|---|---|---|
| full | 17.50 mm | 3.59 % | 120.76 mm | 457.5 mm |
| no spatial filter | 23.00 mm | 5.96 % | 121.04 mm | 458.7 mm |
| no pose compensation | 29.24 mm | 13.46 % | 121.40 mm | 457.7 mm |
| no depth gate | 18.45 mm | 5.36 % | 139.76 mm | 480.4 mm |
| no adaptive weighting | 17.47 mm | 3.55 % | 120.76 mm | 457.4 mm |
| no variance caps | 17.53 mm | 3.59 % | 120.76 mm | 457.5 mm |
| no gradient term | 17.58 mm | 3.59 % | 120.76 mm | 457.5 mm |

- **Pose compensation carries the method.** Removing it costs 67 % on p90 and nearly four
  times the bad pixels.
- **The spatial filter is the second pillar** (+31 % p90 without it).
- **The gate is what protects the tail**: without it the interior RMSE rises 16 % and the edge
  MAE 5 %, exactly the stale-surface damage it exists to prevent.
- **Adaptive weighting shows no accuracy gain over a fixed weight here** (17.47 vs 17.50 mm,
  inside the noise). Per the spec, no advantage is claimed for it on this evidence. It is
  slightly steadier in time (1.40 vs 1.49 mm on kt3, 1.88 vs 2.28 mm on kt2), which is the only
  measurable difference.
- The variance caps and the gradient term change almost nothing on this dataset; the gradient
  term is kept because it prevents a real failure elsewhere (see the parameter section).

## Parameter selection (validation split only)

One knob at a time from `configs/default.yaml`, scored by p90 error with the bad-pixel rate as
tie-break, before the test sequences were touched:

| Knob | Values tried | Chosen | Effect on p90 |
|---|---|---|---|
| `spatial.sigma_depth_m` | 0.02, 0.03, 0.05 | **0.05** | 17.74 → 17.50 mm |
| `fusion.k_sigma` | 2, 3, 4 | 3 (default) | < 0.01 mm |
| `fusion.history_decay` | 0.90, 0.95, 0.99 | 0.95 (default) | 0.01 mm |
| `fusion.max_history_ratio` | 4, 8, 16 | 8 (default) | < 0.01 mm |
| `fusion.q_gradient` | 0, 0.25, 1.0 | 0.25 (default) | 0.0 would give 0.07 mm |
| `spatial.radius` | 1, 2, 3 | 2 (default) | 0.04 mm |

Only the range sigma moved anything measurable. `q_gradient = 0` is marginally better on ICL,
but the term is what keeps fusion from being *worse than its own input* on a 45° slanted plane
(P3: 17.7 vs 15.5 mm RMSE). Trading that for 0.07 mm on one dataset would be fitting the
metric rather than the problem, so it stays at 0.25. The frozen configuration is
`configs/icl.yaml`.

## Targets from the spec (10.6)

| Target | Result |
|---|---|
| RMSE at least 15 % below raw on held-out sequences | **NOT met**: 0.2 % (207.7 vs 208.1 mm) |
| RMSE at least 5 % below the bilateral filter | **NOT met**: 0.01 % |
| Edge MAE no more than 5 % worse than B1 | **met**: 435.13 vs 435.14 mm |
| No loss of raw-valid measurements | **met**: coverage 100 % for every method |
| Dynamic behaviour: a clearly incompatible current measurement drops the history in the same frame | **met** (P3 synthetic tests) |
| Process p95 ≤ 16.7 ms at 640×480 on an RTX 4070 Laptop | **not tested**: that GPU is not available here |

The two RMSE targets are not met, and the reason is measured, not guessed.

## Why RMSE does not move

On kt3 (50 frames, raw input against clean depth):

| \|error\| above | share of pixels | share of the total squared error |
|---|---|---|
| 0.05 m | 3.28 % | 99.5 % |
| 0.20 m | 2.70 % | 99.4 % |
| 1.00 m | 1.36 % | 87.6 % |

A 2.7 % population of gross outliers owns 99.4 % of the squared error, so RMSE is a
measurement of that population and of almost nothing else. Those outliers are not confined to
depth edges: widening the edge band from 2 to 16 pixels leaves the interior RMSE essentially
unchanged (117 → 110 mm), and even the flattest third of the image (local gradient below
2 mm/px) has a raw RMSE of 103 mm against a median error of about 9 mm.

**After fusion, 99.7 % of those gross outliers are still gross outliers.** That is the design
working as specified, not a bug: when the current measurement disagrees with the history, the
gate keeps the current measurement (spec 5.5), because the alternative is dragging stale
surfaces across occlusions. The MVP has no mechanism that overrules an implausible current
sample, so an input outlier passes straight through.

This is worth a decision before P7: rejecting gross current outliers (for example by trusting a
strongly supported history when the current sample is wildly inconsistent with both it and its
neighbours) would change the RMSE picture, but it directly conflicts with the spec's rule that
a valid current measurement always wins. It should be chosen deliberately, not slipped in.

## Latency

Informational, from the P4 run (640×480, 60 real kt0 frames, medians after warm-up, including
host-to-device and device-to-host copies and the synchronisation): **CPU 53.0 ms, CUDA
1.45 ms**, p95 2.56 ms. This is not the benchmark the spec asks for: GPU compute is not
separated from the copies, there are no repeats, and no power mode is recorded. The RTX 4070
Laptop target and the Jetson Orin Nano Super remain untested.
