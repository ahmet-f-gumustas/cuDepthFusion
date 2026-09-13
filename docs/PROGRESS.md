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

## P1 — Data: ICL-NUIM downloader, adapter and validation (2026-09-12)

**Gate:** "single-sequence pairing report and two-frame geometry check". **PASSED** on
ICL-NUIM kt0: 1508 matched frames, and the declared conventions (Z, fy −480,
`T_world_camera`, offset 0, 5000/m) beat every rival on 32 of 32 pairs. Details are in
[DATA_VALIDATION.md](DATA_VALIDATION.md).

### Delivered

- `scripts/download_dataset.py` and `cudepthfusion.data.download`:
  - fetches only the selected sequence and checks disk space first (including room
    for extraction);
  - retries with backoff for both the size request and the transfer, resumes via
    HTTP Range, and locks each file so two runs cannot interleave writes;
  - computes a local SHA-256 for every file (the publisher provides none, and the
    manifest says so);
  - never uses a mirror;
  - extracts archives only after checking every member (absolute paths, `..`, links
    pointing outside, devices and FIFOs are refused) and reports a corrupt or
    truncated archive as such.
- `cudepthfusion.data.fetch`: extracts into a staging directory and renames it into
  place, reuses unchanged extractions, and writes `manifest.json`. The manifest holds
  sources, hashes, license and citation, declared conventions, the timestamp rule,
  the split, frame pairing, the exact command and the adapter version.
- `cudepthfusion.data.icl_nuim`:
  - frame-id indexing; duplicate and unparsable ids are reported, never picked
    silently;
  - pairing by id across clean, noisy and poses;
  - integer pose keys (not seconds);
  - depth conversion to metres (Z or ray distance);
  - `IclInputSequence` hands out only noisy depth, intrinsics, pose and timestamp,
    and only after `validation.json` passed for the exact manifest bytes;
  - `IclGroundTruth` is the only path to clean depth (evaluator only).
- `cudepthfusion.data.geometry_check` and `validation`, plus the CLI command
  `validate-data`: pairing, noisy/clean consistency and the 48-candidate two-frame
  convention search (method in [DATA_VALIDATION.md](DATA_VALIDATION.md)).
- `docs/DATASETS.md`: license and attribution (ICL-NUIM, CC BY 3.0), splits, and
  download/validate commands.
- New `[data]` extra (`opencv-python-headless`, for 16-bit PNGs). `dev` includes it
  and `pytest-cov`.

### Verification run on RTX 4090 Laptop

| Check | Command | Result |
|---|---|---|
| Real download (kt0) | `python scripts/download_dataset.py --dataset icl-nuim --sequence kt0 --variants clean noisy poses --output data/icl` | clean 711 444 709 B, noisy 940 525 575 B, poses 108 786 B (exact sizes, `gzip -t` OK); both archives extracted; manifest written |
| Pairing | same run | clean 1509, noisy 1509, poses 1508 → 1508 matched; id 0 has no pose; no duplicates |
| Validation gate | `python -m cudepthfusion.cli validate-data --manifest data/icl/kt0/manifest.json` | PASSED in 55 s; noisy/clean median diff 7 mm (max 18 mm); declared mean median \|ΔZ\|/Z 0.000184; every rival loses 32/32 pairs; stdout and `validation.json` are strict JSON |
| Filter input end to end | `IclInputSequence` → `DepthFusion.process` (CPU) on 3 real frames | 480×640 float32 metres, fy −480, reset/temporal diagnostics as expected |
| Concurrent-download lock | a second downloader started during the real download | refused with "another process is already downloading", exit 2, `.part` untouched |
| Python tests | `pytest --cov=cudepthfusion` | 114 passed, 95 % line coverage (the `data/` modules are between 87 % and 100 %) |
| C++ tests | `ctest --test-dir build/cpu` | 44 passed, 1 skipped (GPU, CPU-only build); no C++ changes in P1 |
| Lint and format | `ruff check`, `ruff format --check` | clean |
| Code review | Python reviewer pass | 1 HIGH and 2 MEDIUM findings, all fixed with regression tests (see below) |

Not done in this phase: kt1–kt3 downloads (needed from P3/P5 onward), and TUM RGB-D
(v0.2 roadmap). The real ICL download is not part of the automated tests; they use a
synthetic, offline ICL look-alike.

### Found and fixed along the way

- Scoring with nearest-pixel sampling and a "runner-up 3× worse" rule could not
  identify a one-frame pose/image offset: the rounding floor was larger than the
  offset effect. Replaced by bilinear 1/Z sampling (exact on planes) and a per-pair
  win-rate rule, before any real data was evaluated.
- Offset candidates that lacked a pose at the sequence boundary lost pairs and were
  ranked last regardless of residual. All candidates are now scored on identical
  pairs.
- The size (HEAD) request was not retried, so one transient DNS failure aborted a
  download. It now goes through the same retry/backoff as the transfer. Regression
  test added.
- After a resume, the progress log printed a line per chunk. It now starts at the
  next 5 % step. Regression test added.
- **Operational incident:** a `pkill -f` meant to stop a download killed only the
  wrapping shell. The Python child, which had read its script from stdin, kept
  appending to the clean archive's `.part` alongside a restarted download. The result
  was a 816 MB file that failed `gzip -t`. It was deleted and downloaded again.
  Prevention in the tool: a per-file exclusive lock, verified live (a second run
  exits with "another process is already downloading"). Corrupt archives are also now
  reported clearly instead of with a traceback. The noisy archive was not affected:
  its size is exact and its gzip CRC check passes.
- Review findings fixed:
  - the `validate-data` stdout JSON could contain `Infinity`;
  - a declared convention that could not be scored raised a traceback;
  - the ray-distance formula was duplicated, and now lives in
    `cudepthfusion.data.camera`.
- The P0 `.gitignore` entry `data/` was not anchored, so it also matched the new
  `python/cudepthfusion/data/` package. `git status` did not list the package, and
  ruff, which honours `.gitignore`, never linted it; a few long lines slipped through
  as a result. It is now `/data/` (and `/runs/`). This was caught before the first
  commit.

## P2 — Synthetic oracle scenes (2026-09-13)

**Gate:** "analytic Z and pose tests; fixed seed". **PASSED.** Details are in
[SYNTHETIC.md](SYNTHETIC.md).

### Delivered

- `cudepthfusion.synthetic`:
  - scenes are planar rectangles (unbounded planes, bounded patches, movers at constant
    velocity), rendered by exact ray casting into float64 camera Z and surface ids;
  - rigid trajectories: static, linear, spin, handheld;
  - seeded noise σ(z) = a + b·z² with dropouts, reproducible per (seed, frame);
  - `SyntheticSequence` renders lazily and keeps filter input (`frame.input`, an
    `InputFrame`) separate from evaluator truth (`frame.truth`: exact Z, surface ids,
    dynamic mask).
- Nine scenarios covering the spec's cases: `plane_static`, `plane_lateral`,
  `plane_dolly`, `slanted_plane`, `step`, `front_surface_pass`, `revealed_background`,
  `room_handheld`, `room_spin`.
- `scripts/make_synthetic.py` / `synthetic.export`:
  - writes the TUM-PNG layout that the ICL adapter reads, plus `oracle.npz` (surface ids
    and dynamic masks);
  - the manifest's generator block records every parameter needed to regenerate the exact
    truth;
  - exports are deterministic, and a directory that is not a synthetic export is never
    replaced.
- Shared helpers: `data.camera.PinholeCamera`, `data.frames.InputFrame` (moved out of the
  ICL adapter) and `data.poses.rotation_to_quaternion` (moved out of the test helpers). The
  P1 tests now use the package generator instead of their own room renderer.

### Verification run on RTX 4090 Laptop

| Check | Command | Result |
|---|---|---|
| Python tests | `pytest --cov=cudepthfusion` | 174 passed, 96 % coverage (the `synthetic/` modules are between 96 % and 100 %) |
| Commit 1 alone | the same, with the export files stashed | 166 passed, ruff clean |
| C++ tests | `ctest --test-dir build/cpu` | 44 passed, 1 skipped (GPU, CPU-only build); no C++ changes in P2 |
| Full export | `python scripts/make_synthetic.py --suite all --output data/synthetic` | 9 scenarios × 60 frames at 640×480 in 16.7 s, 264 MB, 0 unstorable pixels |
| Moving-camera export | `validate-data` on `room_handheld` | PASSED: declared = best, mean median \|ΔZ\|/Z 1.40e-05, every rival loses 32/32 |
| Static-camera export | `validate-data` on `plane_static` | fails as designed: every candidate ties at 0 residual, so nothing is identifiable |
| Lint and format | `ruff check`, `ruff format --check` | clean |

### Found and fixed along the way

- The ground-truth warp test used `room_spin` frames 0 → 40. That is 38° of yaw, which
  leaves only 19.5 % of the pixels comparable, below the test's 20 % coverage bound. The
  pair is now 0 → 20 (19°). The exactness requirement (rtol 1e-9) is unchanged.

### Open items and decisions to carry forward

- **Static-camera exports:** `validate-data` cannot confirm their pose conventions. They
  hold by construction, and the evidence is the generator tests. Load such exports
  with `require_validated=False` explicitly.
- **Noise model:** the synthetic noise has the same form the filter assumes. Synthetic
  results therefore test the filter against its own model, and P5 must label them so.
- **World-point tracking (spec 10.3):** exact Z, surface ids and poses are enough to
  follow fixed world points. The stability metric itself belongs to P5.
- **C++ tests in P3:** they need synthetic inputs, either from Python fixtures or from a
  small C++ port of the ray caster. This is to be decided at the start of P3.
