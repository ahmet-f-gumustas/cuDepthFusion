# Performance

All numbers come from real runs of the commands below. The run folders (`summary.json`,
`latency.csv`, `config_resolved.yaml`, `environment.json`) record the plan, the per-frame
times, nvidia-smi state before and after every repeat, and whether other processes were
using the GPU.

```bash
python -m cudepthfusion.cli benchmark --config configs/benchmark.yaml --output runs/performance
```

## Protocol (spec 10.5)

- **Real consecutive frames.** 600 frames of ICL-NUIM kt0 at 640×480, decoded into memory
  before anything is timed, so no disk access or PNG decoding is inside a measurement.
- **Warm-up and repeats.** Every repeat resets the engine and replays the same slice:
  100 unmeasured frames, then 500 measured ones; 5 repeats. The same frame is never fused
  over and over.
- **Three times, kept apart.**
  - *GPU compute*: CUDA events around the kernel chain, host copies excluded.
  - *Process latency*: NumPy in to owned NumPy out around `DepthFusion.process()`, with
    staging, H2D, D2H and the synchronisation inside it.
  - *Demo throughput*: decoding and rendering included; that belongs to the demo
    ([DEMO.md](DEMO.md)), not here.

  A time that was not measured is `null` with its reason, never 0.
- **The workload is the evaluated one.** `configs/benchmark.yaml` carries exactly the
  parameters of `configs/icl.yaml`, and a test keeps the two equal.

## Conditions of these runs

RTX 4090 Laptop GPU (sm_89, 76 SMs), driver 580.178.04, CUDA 12.8, GCC 11.4, Release build
for sm_89. **The GPU was shared.** A reinforcement-learning training job of the user's ran
throughout and could not be stopped for the measurement. Across the 90 nvidia-smi samples
taken before and after each repeat, GPU utilisation was 50–96 %, the temperature 78–82 °C and
the SM clock 1485–1995 MHz; nvidia-smi reports no power limit on this laptop. So:

- the three builds were run **interleaved** (before, kernels only, after; three rounds, each
  with the full plan), so all of them saw the same load;
- the per-kernel times come from Nsight Systems, which measures each kernel's execution on
  the device and is far less affected by the other job than wall-clock times are;
- the tails (p95, p99) mostly show the other job's time slices, and the "compute" interval
  includes waiting for them: its median is 0.165 ms while the kernels themselves take 63 µs.

Absolute numbers drifted between sessions as the other job's load changed (an earlier
interleaved session measured 1.61 ms before and 1.23 ms after); within one session the
comparison is fair. A rerun on an idle GPU will give lower and tighter numbers.

## Before and after

"Before" is commit `8154c1e` (the P6 algorithm plus the timing instrumentation). "Kernels only"
is the final code with the pre-P7 copy path put back (pageable uploads and downloads straight
into the result arrays), which separates what the kernels bought from what the copies bought.
Each was built separately and run side by side.

### Process latency, 640×480, 3 × 2500 measured frames per build

| | Before | Kernels only | After |
|---|---|---|---|
| process latency, median | 1.483 ms | 1.215 ms | **0.938 ms** (−37 %) |
| process latency, p95 | 2.789 ms | 2.465 ms | 2.394 ms |
| process latency, p99 | 3.139 ms | 2.825 ms | 2.916 ms |
| upload (H2D), median | 0.184 ms | 0.184 ms | 0.104 ms |
| GPU compute, median (includes waiting for the other job) | 0.463 ms | 0.166 ms | 0.165 ms |
| download (D2H), median | 0.749 ms | 0.797 ms | 0.393 ms |
| host work outside the device intervals, median | 0.008 ms | 0.008 ms | 0.226 ms |
| everything except compute, median | 0.942 ms | 0.989 ms | 0.728 ms |

Read the copy rows together. Before, the driver copied pageable memory synchronously, so the
copies into the result arrays were inside the download interval. After, the downloads go to
pinned memory at the link's speed, and the copies into the owned result arrays happen on the
host, overlapping the remaining transfers. The row that compares like with like is the last
one: 0.989 → 0.728 ms.

### Kernel execution times (Nsight Systems, medians over 220 frames)

| Kernel | Before | After |
|---|---|---|
| fuse | 270.6 µs | **14.6 µs** |
| bilateral | 19.1 µs | 19.1 µs |
| gather | 8.9 µs | 9.0 µs |
| variance | 7.3 µs | 7.3 µs |
| sanitize | 6.9 µs | 6.8 µs |
| project | 6.2 µs | 6.2 µs |
| device-to-device history copies | 4 × 1.8 µs | none |
| **all kernels** | **319 µs** | **63 µs** |

The spec's speed target (process p95 ≤ 16.7 ms at 640×480) refers to the RTX 4070 Laptop, which
was not measured here. On the 4090 Laptop, even with the GPU shared, p95 is 2.4 ms.

The CPU backend, for context: median 101 ms, p95 105 ms (20 warm-up, 100 measured frames,
3 repeats) on the same loaded machine. The CPU code did not change in P7, and this number is
higher than the informal 53 ms of P4, which used a different protocol on a quieter machine.

## What changed, and what each step bought

Steps 1–4 are the kernel changes (1.483 → 1.215 ms above), steps 5–6 the copy changes
(1.215 → 0.938 ms). The per-step numbers come from quick A/B runs on the shared GPU while the
work was in progress.

1. **Counting per warp instead of per thread.** Every kernel counted its decisions with one
   shared-memory atomic per pixel, so 256 threads queued on the same address, and the fuse
   kernel added a double per fused pixel the same way. A ballot now counts a warp at once and
   one lane adds it. Fuse: 0.27 → 0.11 ms. A new parity test compares every counter between
   the backends: the integers agree exactly.
2. **Fewer FP64 divisions: no effect** (fuse 0.111 → 0.110 ms). That ruled out the divisions
   and pointed at FP64 throughput as such. The register report shows 32 registers and no
   spills, and the arithmetic adds up: about 120 FP64 instructions per pixel, at 1/64 of the
   FP32 rate on this GPU, is about 120 µs for 307 200 pixels.
3. **Gate and merge in FP32 on the GPU.** Spec 7.2 asks for FP32 kernels; the double gate
   was a stricter choice made in P4 for exact parity. The CPU reference keeps double. Fuse:
   271 → 15 µs. Parity and quality were re-measured (below) and did not move.
4. **History by swapping buffers, not copying.** The output buffers become the history by
   trading places with it once their downloads are queued: four device copies per frame
   (10 µs) are gone.
5. **Pinned staging for the downloads and the upload.** Copies from pageable memory go
   through the driver's bounce buffers, synchronously. The input is now copied into a pinned
   buffer and uploaded from there (0.184 → 0.104 ms), and all outputs are downloaded into one
   pinned block at the link's full speed (0.797 → 0.393 ms).
6. **Host copies overlap the remaining DMA.** Each output is copied into its owned result as
   soon as its own transfer has finished, instead of after all of them, and the result
   vectors are filled with `assign`, not `resize` plus `memcpy`, which would zero them first.
   Steps 5 and 6 were measured together: everything outside compute went from 0.989 to
   0.728 ms, and a separate four-round A/B of the two copy paths alone gave 1.230 → 0.941 ms.

### Measured and rejected

- **Shared-memory tile for the bilateral filter** (spec 7.4). Built with a halo, bit-identical
  to the global version on 80 real frames at radii 1, 2, 5 and 16 and odd image sizes, and not
  faster: 20.6 µs against 19.1 µs, in a run where the untouched kernels were also about 7 %
  slower. The 25 `expf` per pixel bound the kernel and the caches already serve the
  overlapping loads. The simpler global-memory version stays.
- **Packing the outputs into one transfer.** A micro-benchmark of this laptop's link: one
  4.9 MB pinned copy takes 0.376 ms and six copies of the same bytes 0.379 ms, both about
  13 GB/s. The download is at the link's limit; the number of copies does not matter.
- **Deriving outputs on the host** (validity from the source mask, confidence from the
  variance) would transfer 1.5 MB less (about 0.12 ms), and pay most of it back in host
  passes. Not worth changing the contract.
- **CUDA Graphs** were not tried. The host issues about 20 CUDA calls per frame at 1.5–2.5 µs
  each, which overlap the device work, so there is little left to win at this size; the
  spec defers graphs until the contract is settled anyway.

## Parity and quality after the changes

**Parity, real data.** CPU against CUDA on the first 150 frames of kt1 (46 million pixels),
before and after:

| | Before | After |
|---|---|---|
| validity differences | 0 | 0 |
| source-decision differences | 830 (0.0018 %) | 830 (0.0018 %) |
| pixels with the same decision but depth differing by > 1e-5 m | 7 664 | 7 664 |
| largest such difference | 4.20 mm | 4.20 mm |

Identical: the FP32 gate did not flip a single additional decision. The differences that
exist are the z-buffer ties described in [ALGORITHM.md](ALGORITHM.md), carried by the
recursive history for a few frames. On the synthetic noisy fixture, the near-gate count is 0
of 82 944 pixels.

**Quality.** The frozen `configs/icl.yaml`, CUDA backend, compared with the P5 runs:

| Split | Method | RMSE | p90 \|e\| | bad pixels | edge MAE | coverage |
|---|---|---|---|---|---|---|
| validation (kt1) | B4 before | 220.781 mm | 17.504 mm | 3.589 % | 457.484 mm | 100 % |
| validation (kt1) | B4 after | 220.781 mm | 17.504 mm | 3.589 % | 457.484 mm | 100 % |
| test (kt2 + kt3) | B4 before | 207.744 mm | 17.016 mm | 3.829 % | 435.134 mm | 100 % |
| test (kt2 + kt3) | B4 after | 207.744 mm | 17.016 mm | 3.829 % | 435.134 mm | 100 % |

The largest difference in any pooled B4 metric is 7.5e-12 m (RMSE on the test split); p90,
bad-pixel rate and edge MAE are identical, and so are B1 and B3.

## Where the time goes now

Of the 0.938 ms median: the download takes 0.393 ms and is at the link's limit; the host work
outside the device intervals (staging the input, copying the outputs into their owned arrays,
validation) 0.226 ms; the upload 0.104 ms; and the compute interval 0.165 ms, of which the six
kernels themselves are 0.063 ms and the rest is waiting for the other job. Faster kernels no
longer move the total. Transferring fewer bytes, or keeping results on the device for a
consumer that is also on the device, would.

## Not measured yet

- **RTX 4070 Laptop**, which the 16.7 ms target refers to. Run the command above there.
- **Jetson Orin Nano Super.** CPU and GPU share one DRAM there, so the copy picture changes
  completely: pinned staging may not pay off, and zero-copy access could replace the
  transfers. Build with `CMAKE_CUDA_ARCHITECTURES=87`, record `nvpmodel -q` (the benchmark
  does this automatically), and compare.
- **An idle GPU on this laptop.** The runs above are fair to each other, not free of load.
