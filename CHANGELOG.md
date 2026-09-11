# Changelog

Versions follow the roadmap in the development spec: 0.0.x releases are phase
snapshots before the first functional release, and 0.1.0 is the first release with
CPU/CUDA fusion evaluated on ICL-NUIM.

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
