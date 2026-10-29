# Environment and target hardware

Only the rows marked **verified** have been built and tested. Nothing on this page
is a performance result: quality is in `docs/RESULTS.md`, and timing, with the power and
thermal conditions it ran under, in `docs/PERFORMANCE.md`.

## Target matrix

| Device | Arch | CPU | Role | Status |
|---|---|---|---|---|
| RTX 4090 Laptop GPU (16 GB) | sm_89 | x86_64 | Main development machine | **verified** (P0 2026-09-11, last 2026-10-29 for P8) |
| RTX 4070 Laptop GPU | sm_89 | x86_64 | Spec performance target: process p95 ≤ 16.7 ms at 640×480 | not yet verified; tested by the maintainer |
| Jetson Orin Nano Super | sm_87 | aarch64 | Embedded target | not yet verified; tested by the maintainer |

A 4090 measurement is not evidence for the 4070 target. Each device's results are
reported separately.

When `CMAKE_CUDA_ARCHITECTURES` is not set, the build targets `87;89`. That covers
all three devices. Pass the value explicitly to build for one device only.

## Verified: RTX 4090 Laptop

| Component | Version |
|---|---|
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-138-generic |
| GPU | NVIDIA GeForce RTX 4090 Laptop GPU, 16 GB, CC 8.9, 76 SMs |
| NVIDIA driver | 580.178.04 for P0–P7, 580.196.07 for P8 (both report CUDA driver API 13.0) |
| CUDA toolkit | 12.8.93 (`/usr/local/cuda-12.8`) |
| Host compiler | GCC 11.4.0 |
| CMake / Ninja | 3.22.1 / 1.10.1 |
| GoogleTest | 1.11.0 (Ubuntu `libgtest-dev`) |
| Python | 3.10.12 (project venv `.venv`) |
| NumPy / PyYAML | 2.2.6 / 6.0.3 |
| pybind11 / scikit-build-core | 3.1.0 / 1.0.3 |
| pytest / pytest-cov / ruff / clang-format | 9.1.1 / 7.1.0 / 0.16.7 / 23.1.1 |
| opencv-python-headless (`[data]` extra) | 5.0.0.93 |

These Python versions are pinned in [`constraints.txt`](../constraints.txt), build
dependencies included. `pyproject.toml` keeps open ranges so the package can be installed next
to other software; the constraints rebuild the verified environment:

```bash
pip install -c constraints.txt --build-constraint constraints.txt -e '.[dev]'
```

CMake 3.22 does not support `CMAKE_CUDA_ARCHITECTURES=native` (that needs CMake
3.24 or newer), so pass the architecture number explicitly.

## Clean-install check

`scripts/check_clean_install.sh` is the P8 gate, "a new user installs and runs the tests". It
unpacks the committed tree (`git archive HEAD`, what a clone or release tarball holds) into a
temporary directory, installs `.[dev]` there into a fresh virtual environment with the pinned
versions (not editable, so a file missing from the package fails it), and runs the CPU smoke
test, pytest and the C++ tests against that install:

```bash
scripts/check_clean_install.sh                                          # CPU-only
CUDACXX=/usr/local/cuda-12.8/bin/nvcc scripts/check_clean_install.sh --cuda 89
```

On this machine both variants passed on 2026-10-29; in the CUDA one the GPU tests ran on the
RTX 4090 Laptop.

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and on every pull request. Hosted
runners have no GPU, so GPU tests report SKIPPED there, and each clean-install job says so in
its summary.

| Job | Platform | What it checks |
|---|---|---|
| lint | Ubuntu 24.04 | `ruff check`, `ruff format --check`, `clang-format` |
| clean install | Ubuntu 22.04, Python 3.10, GCC 11 | `check_clean_install.sh`, CPU-only |
| clean install | Ubuntu 24.04, Python 3.12, GCC 13 | `check_clean_install.sh`, CPU-only |
| sanitizers | Ubuntu 22.04, GCC 11 | C++ tests under AddressSanitizer and UBSan |
| CUDA build | Ubuntu 22.04, CUDA 12.8 | `check_clean_install.sh --cuda "87;89"`: the Jetson and Ada targets compile, GPU tests SKIPPED |

## Capturing an environment

```bash
python -m cudepthfusion.cli info > environment.json
```

The command prints the build type, compilers, CUDA runtime and driver versions,
visible devices, the Python and NumPy versions, the platform, and, on Jetson, the
contents of `/etc/nv_tegra_release`.

## RTX 4070 Laptop checklist

1. Build with `-C cmake.define.CUDEPTHFUSION_ENABLE_CUDA=ON -C cmake.define.CMAKE_CUDA_ARCHITECTURES=89`.
2. Run `pytest` and `pytest -m gpu`. GPU tests must show PASSED, not SKIPPED.
3. Record `cli info`, the laptop power profile, AC or battery state, and the GPU's
   TGP. Laptop GPUs with the same name differ by TGP.

## Jetson Orin Nano Super checklist

Build from source on the device. Do not assume a desktop x86_64 wheel or build
works on aarch64.

1. Record the JetPack/L4T release (`cat /etc/nv_tegra_release`) and the CUDA toolkit
   version that ships with it. The version combination still has to be confirmed
   on the device.
2. Record the power mode (`sudo nvpmodel -q`) and whether `jetson_clocks` is on.
   Both are part of any timing result.
3. Build with `-C cmake.define.CUDEPTHFUSION_ENABLE_CUDA=ON -C cmake.define.CMAKE_CUDA_ARCHITECTURES=87`.
   JetPack's CMake must be 3.22 or newer.
4. Run `pytest`, `pytest -m gpu` and `python -m cudepthfusion.cli info`, then add a
   verified row above.
