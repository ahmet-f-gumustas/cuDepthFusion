# Contributing

Thank you for considering a contribution. Bug reports, measurements on hardware that is not
verified yet (RTX 4070 Laptop, Jetson Orin Nano Super), dataset adapters and fixes are all
welcome. For a change to the algorithm or to the evaluation protocol, please open an issue
first: those change published numbers, and the reason has to be recorded with them.

## Set up

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -c constraints.txt --build-constraint constraints.txt -e '.[dev]'

# With CUDA (89 = RTX 40xx laptop, 87 = Jetson Orin)
pip install -c constraints.txt --build-constraint constraints.txt -e '.[dev]' \
  -C cmake.define.CUDEPTHFUSION_ENABLE_CUDA=ON \
  -C cmake.define.CMAKE_CUDA_ARCHITECTURES=89
```

`constraints.txt` holds the exact versions the last release was verified with. The editable
install rebuilds the extension from scratch on every `pip install`; after changing C++ or CUDA
code, run the install command again.

## Before opening a pull request

```bash
ruff check . && ruff format --check python tests scripts examples
git ls-files '*.cpp' '*.hpp' '*.cu' | xargs clang-format --dry-run -Werror
pytest

cmake -S . -B build/cpu -G Ninja -DCUDEPTHFUSION_WARNINGS_AS_ERRORS=ON
cmake --build build/cpu && ctest --test-dir build/cpu --output-on-failure
```

CI runs these, the AddressSanitizer/UBSan build, a CUDA compile check and
`scripts/check_clean_install.sh` on two Ubuntu/Python versions. CI has no GPU, so if you change
CUDA code, also run the GPU tests on a CUDA machine and say in the pull request which one:

```bash
pytest -m gpu
cmake -S . -B build/cuda -G Ninja -DCUDEPTHFUSION_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/cuda && ctest --test-dir build/cuda --output-on-failure
```

A GPU test that did not run reports SKIPPED, and the pull request should say so rather than
count it as passed.

## Rules the code keeps

These are the project's contracts. A change that breaks one needs its own discussion.

- **The CPU implementation defines correctness.** The CUDA backend implements the same
  algorithm, and the parity tests compare the two. A CUDA change that moves a parity result
  has to explain why.
- **Nothing silent.** `backend="cuda"` never falls back to the CPU. Invalid input is rejected,
  never converted, and the bindings never copy or cast an array behind the caller's back. A
  missing pose is never replaced by identity unless `fusion.assume_static_camera` is set.
- **The filter never sees ground truth.** Clean depth is read only by the evaluator
  (`IclGroundTruth`), never through the filter's input path.
- **Numbers come from runs.** Every result in `docs/` comes from a run folder written by
  `evaluate`, `ablate`, `benchmark` or the demo, with its command, git commit and dataset
  hashes. A value that was not measured is `null` with a reason, never a zero. Parameters are
  chosen on the validation split; the test split is not used for tuning.
- **Datasets are never committed.** They are downloaded into the git-ignored `data/` folder
  and keep their own licenses (see `docs/DATASETS.md`).

## Commits and pull requests

- One logical change per commit. Messages follow the existing history:
  `type(scope): summary` with `feat`, `fix`, `perf`, `test`, `docs`, `build`, `ci` or `chore`,
  and a body that says what changed and how it was verified.
- A pull request says what was run (tests, sanitizers, GPU, datasets) and what was not.
- Add a line to `CHANGELOG.md` for anything a user would notice.

## License

By contributing you agree that your contribution is licensed under the Apache License 2.0,
the license of this repository (see section 5 of `LICENSE`). Do not add code copied from
elsewhere without stating its source and license in the pull request.
