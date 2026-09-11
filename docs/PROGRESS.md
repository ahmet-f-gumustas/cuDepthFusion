# Progress log

Each phase entry records what was delivered, which checks were actually run and
their outcome, and what remains open. Gates follow section 8 of the development
spec.

## P0 — Skeleton (2026-09-11)

**Gate:** "import and CPU smoke test in a clean environment". **PASSED**

### Delivered

- Build system: `CMakeLists.txt` with the options `CUDEPTHFUSION_ENABLE_CUDA` (default
  OFF), `_BUILD_TESTS`, `_BUILD_PYTHON`, `_WARNINGS_AS_ERRORS` and `_SANITIZE`.
  `pyproject.toml` builds with scikit-build-core and pybind11; the package version
  is read from the CMake project version.
- Public C++ API in `include/cudepthfusion/`: config structs and validation, data
  types (intrinsics with signed `fy`, row-major `RigidTransform`, `FrameInput`,
  `FusionResult`, diagnostics enums), pose and intrinsics validation, the
  `DepthFusion` engine (pimpl, mutex-serialised, RAII) and `build_info()`.
- CPU pipeline for this phase (`src/cpu/measurement.*`, `src/engine.cpp`):
  - Input sanitising: NaN, Inf, 0 and out-of-range values become invalid, with a
    count for each category.
  - Measurement variance `(a + b z²)²`, floored.
  - `confidence_score`.
  - `source_mask` and `history_age`.
  - Sequence-reset detection: first frame, explicit reset, resolution change,
    intrinsics change, timestamp not increasing, frame gap.
  - Temporal status from pose availability: a missing pose disables the temporal
    stage and is never replaced by identity unless `fusion.assume_static_camera`
    is set; an invalid pose is rejected with a note.
- CUDA: device and runtime query only (`src/cuda/device_info.cu`). Requesting
  `backend="cuda"` raises `BackendUnavailableError` and never falls back to the CPU.
- pybind11 boundary (`bindings/module.cpp`):
  - Rejects wrong dtype, non-contiguous, misaligned or wrongly shaped arrays
    without copying.
  - Releases the GIL while processing.
  - Returns owned NumPy arrays through capsules.
- Python package `cudepthfusion`:
  - Strict YAML/mapping config loader. Unknown keys, wrong types and YAML 1.1
    `1e-6` strings are rejected.
  - Typed `FusionResult` and `Diagnostics`.
  - CLI with the commands `info`, `check-config` and `smoke`.
- `configs/default.yaml` holds the spec's section 6.3 values (`backend: cpu` until
  P4). A test checks that it matches the C++ defaults.
- Tests: 44 CPU and 1 GPU GoogleTest cases, plus 54 pytest cases.
  GPU tests are labelled or marked `gpu` and report SKIPPED without CUDA.
- `docs/ENVIRONMENT.md` covers the target matrix (RTX 4090 Laptop, RTX 4070 Laptop,
  Jetson Orin Nano Super) and the verified toolchain.

### Verification run on RTX 4090 Laptop

| Check | Command | Result |
|---|---|---|
| C++ CPU-only, `-Werror` | `cmake -S . -B build/cpu -G Ninja -DCUDEPTHFUSION_WARNINGS_AS_ERRORS=ON && cmake --build build/cpu && ctest --test-dir build/cpu` | 44 passed, 1 skipped (GPU) |
| C++ ASan + UBSan (Debug) | `-DCMAKE_BUILD_TYPE=Debug -DCUDEPTHFUSION_SANITIZE=ON` | 44 passed, 1 skipped (GPU), no sanitizer reports |
| C++ CUDA sm_89 | `-DCUDEPTHFUSION_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89` | 45 passed (GPU device test PASSED) |
| Default CUDA archs | `-DCUDEPTHFUSION_ENABLE_CUDA=ON` (no arch given) | builds; nvcc targets `compute_87/sm_87` and `compute_89/sm_89` |
| Python CUDA build | `pip install -e . -C cmake.define.CUDEPTHFUSION_ENABLE_CUDA=ON -C cmake.define.CMAKE_CUDA_ARCHITECTURES=89`, then `pytest` and `pytest -m gpu` | 54 passed; the GPU test PASSED |
| Clean-environment gate | fresh venv, non-editable `pip install '.[dev]'` (CPU-only), import, `cudepthfusion smoke --backend cpu`, `pytest` | import OK, smoke `passed: true`, 53 passed, 1 skipped (GPU) |
| Lint and format | `ruff check`, `ruff format --check`, `clang-format --dry-run --Werror`, `cppcheck --enable=warning,performance,portability` | clean |
| Code review | C++ and Python reviewer passes | C++: no HIGH issues. Python: 2 HIGH fixed with regression tests (`engine.config` now reports the engine's own validated copy; unreadable or invalid YAML raises `ConfigError` instead of a traceback) |

Not run in this phase:
- `compute-sanitizer`: there are no kernels yet (P4).
- RTX 4070 Laptop and Jetson Orin Nano Super runs: the maintainer does these.
- CI: intentionally not set up for now.

### Found and fixed along the way

- Overriding `SKIP_REGULAR_EXPRESSION` on `gtest_discover_tests` broke CMake's
  escaping. CTest then reported every test as "Skipped" even though they passed.
  The override was removed, and CMake 3.22's built-in skip handling is used.
- A multi-value `CMAKE_CUDA_ARCHITECTURES` (`87;89`) split the build-info compile
  definition in two. It is now joined with commas.
- A persistent scikit-build `build-dir` let the cached CMake option
  `CUDEPTHFUSION_ENABLE_CUDA` leak between pip installs. It was removed, so every
  pip build starts clean.

### Open items and decisions to carry forward

- The spatial and temporal stages are not implemented. When spatial filtering is
  enabled, diagnostics say so in `notes`; frames that would fuse report
  `temporal_status = "not_implemented"`. Both are replaced in P3.
- Metrics definition for P5: `raw_input_valid` must mean "valid after sanitize
  with the same `depth.min_m`/`max_m`". Otherwise the "no raw-valid loss" coverage
  target conflicts with range filtering.
- An invalid pose (non-orthonormal, bottom row wrong, non-finite) disables the
  temporal stage for that frame with a note instead of raising, following the
  spec's "reset on invalid pose". Malformed pose arrays (wrong dtype or shape) do
  raise.
- `variance_m2` and `confidence_score` are 0 on invalid pixels, like `depth_m`.
  `valid_mask` is authoritative.
