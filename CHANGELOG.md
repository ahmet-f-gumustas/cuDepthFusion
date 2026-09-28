# Changelog

Versions follow the roadmap in the development spec: 0.0.x releases are phase
snapshots before the first functional release, and 0.1.0 is the first release with
CPU/CUDA fusion evaluated on ICL-NUIM.

## [0.1.0] - 2026-09-28

The first functional release: pose-aware temporal depth fusion on the CPU and on CUDA,
evaluated on ICL-NUIM with a fixed protocol. Release preparation (P8: citation file,
contributing guide, a single reproduction command, a clean-install check) is still to come.

### Added

- **Data (P1).** ICL-NUIM downloader with resumable, locked downloads and a manifest;
  `validate-data`, which establishes the dataset's conventions from evidence (a 48-candidate
  search) before any filter may read it. kt0–kt3 validated.
- **Synthetic oracle (P2).** Analytic scenes with exact ground truth and seeded noise, nine
  scenarios, no download needed.
- **CPU fusion (P3).** Mask-aware bilateral filter, forward reprojection with a 64-bit
  z-buffer, a compatibility gate on transported uncertainty, confidence-weighted merge with
  caps, optional history-only mode with a TTL, and reset handling.
- **CUDA backend (P4).** The same algorithm on the GPU; `backend="cuda"` never falls back to
  the CPU. Parity against the CPU reference and compute-sanitizer evidence.
- **Evaluation (P5).** Baselines B0–B4, a fixed evaluation mask, sequence splits, metrics,
  temporal stability, ablation, and `evaluate` / `ablate` commands. `configs/icl.yaml` frozen
  on the validation split.
- **Demo (P6).** `examples/compare_depth.py`: six panels on shared scales, PNG/MP4 output,
  separate process-latency and demo-throughput numbers.
- **Profiling (P7).** CUDA-event stage timings in `Diagnostics.device`, the `benchmark`
  command, and an optimised CUDA pipeline: 1.48 → 0.94 ms per 640×480 frame (median, RTX 4090
  Laptop), quality unchanged.

### Results (ICL-NUIM test split, kt2 + kt3)

Against the raw input, the full method lowers the p90 error from 26.0 to 17.0 mm and the
bad-pixel rate from 9.6 to 3.8 %, makes a fixed world point about 5× steadier over time, and
keeps 100 % coverage. See `docs/RESULTS.md` and `docs/PERFORMANCE.md`.

### Known limitations

- **The spec's RMSE targets are not met** (0.2 % against raw where 15 % was targeted, 0.01 %
  against the bilateral filter where 5 % was targeted). 2.7 % of pixels carry 99.4 % of the
  squared error, and the gate keeps the current measurement by design, so gross input outliers
  pass through. Whether to add outlier rejection is an open design decision.
- Measured on an RTX 4090 Laptop only, while another job shared the GPU. The RTX 4070 Laptop
  (the 16.7 ms target) and the Jetson Orin Nano Super are untested.
- Evaluated on ICL-NUIM (synthetic renderings with a noise model) only; no real-sensor
  dataset yet.

## [0.0.1] - 2026-09-11 (pre-release)

The P0 skeleton. **No depth fusion yet**: the engine returns the sanitised current
measurement. Spatial filtering and temporal fusion arrive in P3 (CPU) and P4
(CUDA).

### Added

- CMake build, CPU-only by default. Optional CUDA; the default architectures are
  `87;89` for Jetson Orin Nano Super and RTX 40xx laptop GPUs. Python packaging
  uses scikit-build-core and pybind11.
- Public C++ API:
  - config structs and validation
  - intrinsics with signed `fy`
  - rigid-transform validation
  - `DepthFusion` engine with explicit backends; `cuda` never falls back to `cpu`
  - build and device info
- CPU measurement stage:
  - input sanitising with per-category counts
  - floored measurement variance and `confidence_score`
  - `source_mask` and `history_age`
  - sequence-reset detection
  - pose-aware temporal status; a missing pose is never silently replaced by
    identity
- pybind11 boundary: rejects non-contiguous or wrong-dtype arrays without hidden
  copies, releases the GIL while processing, returns owned NumPy arrays.
- Python package `cudepthfusion`: strict YAML config loader, typed results, and a
  CLI with `info`, `check-config` and `smoke`.
- Tests: GoogleTest (`cpu`/`gpu` labels) and pytest (`gpu` marker). GPU tests
  report SKIPPED without a CUDA build and device.
- Docs: target hardware matrix, verified environment
  ([docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)) and phase log
  ([docs/PROGRESS.md](docs/PROGRESS.md)).

### Verified on

- RTX 4090 Laptop (sm_89), Ubuntu 22.04, CUDA 12.8, GCC 11.4: CPU-only, ASan/UBSan
  and CUDA builds, and a clean-venv install with the smoke test.
- RTX 4070 Laptop and Jetson Orin Nano Super: not yet verified.

[0.0.1]: https://github.com/ahmet-f-gumustas/cuDepthFusion/releases/tag/v0.0.1
