# cuDepthFusion

Open-source temporal depth fusion that is aware of camera motion and preserves edges.
It combines consecutive noisy depth frames with their camera poses to produce a more
stable depth image in the current view. Old surfaces are not dragged across
occlusions or moving foregrounds. A C++ reference implementation defines
correctness, a CUDA backend makes it fast, and Python drives datasets, evaluation
and demos.

> **Status: v0.0.1 pre-release (P0 skeleton). Fusion is not implemented yet.**
> The engine validates inputs, sanitises depth and returns the current measurement
> with its variance. The spatial filter and temporal fusion arrive in P3 (CPU) and
> P4 (CUDA). See [docs/PROGRESS.md](docs/PROGRESS.md) and
> [CHANGELOG.md](CHANGELOG.md). No performance or quality numbers have been
> measured yet.

## Hardware targets

| Device | Arch | Status |
|---|---|---|
| RTX 4090 Laptop | sm_89 | development machine, P0 verified |
| RTX 4070 Laptop | sm_89 | spec performance target, not yet verified |
| Jetson Orin Nano Super | sm_87, aarch64 | embedded target, not yet verified |

Toolchain details and per-device checklists are in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

## Build and install

You need Python 3.10 or newer, CMake 3.22 or newer, and a C++17 compiler. The CUDA
build also needs the CUDA toolkit. Always use a virtual environment.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# CPU-only
pip install -e '.[dev]'

# With CUDA (set the architecture: 89 = RTX 40xx laptop, 87 = Jetson Orin)
pip install -e '.[dev]' \
  -C cmake.define.CUDEPTHFUSION_ENABLE_CUDA=ON \
  -C cmake.define.CMAKE_CUDA_ARCHITECTURES=89
```

Use `CUDACXX=/usr/local/cuda-12.8/bin/nvcc` when `nvcc` is not on `PATH`.

C++ library and tests without Python:

```bash
cmake -S . -B build/cpu -G Ninja
cmake --build build/cpu && ctest --test-dir build/cpu --output-on-failure

# Optional: -DCUDEPTHFUSION_ENABLE_CUDA=ON, -DCUDEPTHFUSION_SANITIZE=ON
```

## Usage

```python
import cudepthfusion as cdf

engine = cdf.DepthFusion("configs/default.yaml", backend="cpu")
result = engine.process(
    depth_m=depth_m,                    # C-contiguous float32 (H, W), metres
    intrinsics=cdf.Intrinsics(fx=481.2, fy=480.0, cx=319.5, cy=239.5),
    T_world_camera=T_world_camera,      # float64 (4, 4) camera-to-world, or None
    timestamp_s=timestamp_s,
)
result.depth_m, result.valid_mask, result.variance_m2, result.confidence_score
result.source_mask, result.history_age, result.diagnostics
```

- Invalid input is rejected, never silently converted. The engine will not copy a
  non-contiguous array or cast a float64 one for you.
- A missing pose disables the temporal stage. It is never replaced by identity
  unless you set `fusion.assume_static_camera: true`.
- `backend="cuda"` never falls back to the CPU.

CLI:

```bash
python -m cudepthfusion.cli info                        # build, CUDA and platform info (JSON)
python -m cudepthfusion.cli check-config configs/default.yaml
python -m cudepthfusion.cli smoke --backend cpu         # plumbing check, not a benchmark
```

## Datasets

Data is downloaded on demand into the git-ignored `data/` folder and must pass validation
before the filter may read it:

```bash
python scripts/download_dataset.py --dataset icl-nuim --sequence kt0 \
    --variants clean noisy poses --output data/icl
python -m cudepthfusion.cli validate-data --manifest data/icl/kt0/manifest.json
```

Licenses and attribution are in [docs/DATASETS.md](docs/DATASETS.md). The evidence for
each dataset's conventions is in [docs/DATA_VALIDATION.md](docs/DATA_VALIDATION.md).

## Tests

```bash
pytest              # GPU tests report SKIPPED without a CUDA build and device
pytest -m gpu       # GPU-only tests
ctest --test-dir build/cpu -L cpu
```

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| P0 | Build, API types, CPU-only mode, config validation | done |
| P1 | ICL-NUIM downloader and adapter, manifest, geometry check | done (kt0 validated) |
| P2 | Synthetic oracle scenes | next |
| P3 | CPU bilateral filter, z-buffer reprojection, gating, fusion | planned |
| P4 | CUDA kernels, CPU/GPU parity, compute-sanitizer | planned |
| P5 | Baselines, metrics, ablation | planned |
| P6 | Demo (PNG/MP4, side-by-side) | planned |
| P7 | Profiling and optimisation | planned |
| P8 | Release preparation | planned |

## License

The code is licensed under Apache-2.0 (see [LICENSE](LICENSE)). Datasets keep
their own licenses (ICL-NUIM is CC BY 3.0) and are never committed to this
repository.
