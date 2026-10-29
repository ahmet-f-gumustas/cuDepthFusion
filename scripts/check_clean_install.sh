#!/usr/bin/env bash
# Install cuDepthFusion the way a new user does and run its tests (the P8 gate).
#
#   scripts/check_clean_install.sh                # CPU-only
#   scripts/check_clean_install.sh --cuda 89      # with CUDA for these architectures
#   scripts/check_clean_install.sh --cuda "87;89" --python python3.12 --keep
#
# Only the committed tree is used: `git archive HEAD`, what a clone or a release tarball
# holds, is unpacked into a temporary directory, so no build folder, virtual environment,
# uncommitted file or downloaded dataset of this working copy can leak into the check.
# There a fresh virtual environment installs '.[dev]' (not editable) with the exact
# versions of constraints.txt, and the smoke test, pytest and the C++ tests run against it.
# GPU tests report SKIPPED when the build has no CUDA support or the machine no device.
# A CUDA build needs nvcc on PATH or CUDACXX, e.g. CUDACXX=/usr/local/cuda-12.8/bin/nvcc.
set -euo pipefail

cuda_archs=""
python_exe="python3"
keep=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda) cuda_archs="${2:?--cuda needs architectures, e.g. 89 or \"87;89\"}"; shift 2 ;;
    --python) python_exe="${2:?--python needs an interpreter}"; shift 2 ;;
    --keep) keep=1; shift ;;
    -h | --help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "error: unknown option $1" >&2; exit 2 ;;
  esac
done

repo="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
work="$(mktemp -d "${TMPDIR:-/tmp}/cudepthfusion-clean-install.XXXXXX")"
cleanup() {
  if [[ $keep -eq 1 ]]; then echo "kept $work"; else rm -rf "$work"; fi
}
trap cleanup EXIT
step() { printf '\n== %s\n' "$*"; }

step "unpacking commit $(git -C "$repo" rev-parse --short HEAD) into $work/src"
mkdir "$work/src"
git -C "$repo" archive --format=tar HEAD | tar -x -C "$work/src"
cd "$work/src"

step "creating a virtual environment with $("$python_exe" --version 2>&1)"
"$python_exe" -m venv "$work/venv"
py="$work/venv/bin/python"
"$py" -m pip install --quiet --upgrade -c constraints.txt pip

install=(-m pip install --quiet -c constraints.txt --build-constraint constraints.txt '.[dev]')
cmake_cuda=()
if [[ -n "$cuda_archs" ]]; then
  install+=(-C cmake.define.CUDEPTHFUSION_ENABLE_CUDA=ON
            -C "cmake.define.CMAKE_CUDA_ARCHITECTURES=$cuda_archs")
  cmake_cuda=(-DCUDEPTHFUSION_ENABLE_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=$cuda_archs")
fi
step "installing '.[dev]' (CUDA: ${cuda_archs:-off})"
"$py" "${install[@]}"

step "build info and smoke test"
"$py" -m cudepthfusion.cli info
gpu_status="$("$py" - <<'EOF'
import cudepthfusion as cdf

info = cdf.build_info()
if not info["cuda_compiled"]:
    print("not run: built without CUDA support")
elif not info["cuda_devices"]:
    print(f"not run: no CUDA device ({info['cuda_error']})")
else:
    print(f"run on {info['cuda_devices'][0]['name']}")
EOF
)"
"$py" -m cudepthfusion.cli smoke --backend cpu > /dev/null
echo "cpu smoke test passed"
if [[ "$gpu_status" == run* ]]; then
  "$py" -m cudepthfusion.cli smoke --backend cuda > /dev/null
  echo "cuda smoke test passed"
fi

step "pytest against the installed package"
"$py" -m pytest -q

step "C++ build and ctest"
generator=()
if command -v ninja > /dev/null; then generator=(-G Ninja); fi
cmake -S . -B build "${generator[@]}" -DCMAKE_BUILD_TYPE=Release \
  -DCUDEPTHFUSION_WARNINGS_AS_ERRORS=ON "${cmake_cuda[@]}" > /dev/null
cmake --build build
ctest --test-dir build --output-on-failure

step "clean install check passed"
echo "GPU tests: $gpu_status"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo "### Clean install ($("$py" --version 2>&1), CUDA: ${cuda_archs:-off})"
    echo "Smoke test, pytest and ctest passed. **GPU tests: $gpu_status.**"
  } >> "$GITHUB_STEP_SUMMARY"
fi
