# Datasets

Datasets are downloaded on demand into `data/`, which is git-ignored. Nothing from
a dataset is committed to this repository. Each dataset keeps its own license, which
is separate from the code license (Apache-2.0).

## ICL-NUIM (primary benchmark)

- **Source:** [official page](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html).
  Download URLs are taken from that page; no mirror is ever used.
- **License:** data released under
  [Creative Commons Attribution 3.0 (CC BY 3.0)](http://creativecommons.org/licenses/by/3.0/).
- **Attribution:** please cite
  > A. Handa, T. Whelan, J.B. McDonald and A.J. Davison, *A Benchmark for RGB-D Visual
  > Odometry, 3D Reconstruction and SLAM*, IEEE Intl. Conf. on Robotics and Automation
  > (ICRA), Hong Kong, 2014.
- **Distribution used:** "TUM RGB-D Compatible PNGs" (clean and "with noise") and the
  TUM-format pose file (`livingRoomN.gt.freiburg`).
- **Splits (fixed before any tuning):**

  | Sequence | Split |
  |---|---|
  | kt0 | development |
  | kt1 | parameter selection (validation) |
  | kt2, kt3 | final evaluation (test) |

  All four are trajectories through the same living-room scene, so results on them are
  not evidence of generalisation to other environments.

### Download and validate

```bash
pip install -e '.[dev]'            # includes the [data] extra (OpenCV for 16-bit PNGs)
python scripts/download_dataset.py --dataset icl-nuim --sequence kt0 \
    --variants clean noisy poses --output data/icl
python -m cudepthfusion.cli validate-data --manifest data/icl/kt0/manifest.json
```

The downloader:
- fetches only the selected sequence and checks free disk space first;
- retries with backoff and resumes interrupted downloads;
- extracts archives only after checking every member (no absolute paths, `..`,
  links pointing outside, devices or FIFOs);
- records a locally computed SHA-256 for every file. The publisher does not provide
  checksums, and the manifest says so.

`data/icl/<seq>/manifest.json` records:
- URLs, sizes, hashes and retrieval time;
- license and citation;
- declared conventions (units, depth kind, intrinsics, pose direction, pose-to-image
  id offset, frame rate) and the timestamp rule;
- split, frame pairing, the exact command and the adapter version.

`validate-data` writes `validation.json` next to the manifest. Filter input
(`IclInputSequence`) is refused until that report has passed for the exact manifest
on disk. Clean depth is only reachable through `IclGroundTruth`, which is used by the
evaluator and never by the filter. Evidence for the conventions is in
[DATA_VALIDATION.md](DATA_VALIDATION.md).

## TUM RGB-D (planned, P1 scope ends at ICL-NUIM)

The secondary real-sensor test comes later (v0.2 roadmap). TUM's trajectory ground
truth is not clean per-pixel depth, so no absolute depth-accuracy claims will be made
on it.
