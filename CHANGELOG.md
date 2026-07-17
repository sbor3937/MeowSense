# Changelog

All notable changes to MeowSense are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While MeowSense is pre-1.0, its public surface — the script CLIs, the feature
settings in `src/features.py`, and the metrics reported in the README — may
still change. Breaking changes to any of those will bump the **minor** version.

## [Unreleased]

Planned, in priority order (see the [README roadmap](README.md#roadmap) and
[`docs/ROADMAP.md`](docs/ROADMAP.md) for detail):

- **Fix the CNN validation protocol**: hold out a cat-grouped validation split
  for monitoring and touch the test split exactly once.
- **Cache the mel filterbank** (currently rebuilt per clip — roughly half of
  feature-extraction time).
- **Transfer-learning baseline** — frozen YAMNet embeddings + a linear probe.
- **Version pinning** (dependency upper bounds), now that CI can catch breakage.

## [0.1.2] - 2026-07-17

### Added
- **Test suite** (`tests/`, pytest). 40 hermetic unit tests covering the
  filename parser, mel filterbank, spectrograms, fixed-size framing, WAV I/O and
  dataset scanning — all on synthesized audio, no network or dataset needed.
- **Reproduction tests** (`tests/test_reproduce.py`, marked `data`) that assert
  the class balance and the baseline accuracies (RandomForest ≈ 0.49,
  SVM-RBF ≈ 0.52) actually hold on the real dataset.
- **Continuous integration** (`.github/workflows/ci.yml`): a fast hermetic unit
  job on every push, plus a `reproduce` job that downloads CatMeows (cached) and
  runs the reproduction tests — making "every number reproduces" a checked fact.
- `dev` optional-dependency group and pytest configuration in `pyproject.toml`;
  CI badge and a "Development" section in the README.

## [0.1.1] - 2026-07-17

### Added
- `pyproject.toml` with PEP 621 project metadata — the single source of truth
  for the project version.
- This changelog.
- A version badge in the README.

## [0.1.0] - 2026-07-17

Initial public release. Every reported metric reproduces from a clean clone via
the documented commands.

### Added
- `src/download_data.py` — idempotent CatMeows download from Zenodo (record
  4008297), streamed to a `.part` file so an interrupted run cannot leave a
  truncated archive.
- `src/features.py` — numpy/scipy-only DSP path (mel filterbank, log-mel
  spectrograms, MFCCs at 8 kHz / `n_fft=256` / 40 mel bands) plus CatMeows
  filename metadata parsing, including the cat ID that makes leak-free
  validation possible.
- `src/train_baseline.py` — RandomForest and SVM-RBF on MFCC summary vectors,
  validated by GroupShuffleSplit / GroupKFold over cat IDs.
- `src/train_cnn.py` — SmallCNN on log-mel spectrograms, kept as a documented
  negative result: a from-scratch CNN is not justified at 440 clips.
- `notebooks/01_eda.ipynb`, `notebooks/02_baselines.ipynb` — committed with
  executed outputs; `02` demonstrates the ~20-point accuracy inflation a random
  split produces versus grouping by cat.
- `docs/ROADMAP.md`, `CONTRIBUTING.md`, MIT `LICENSE` (with a note that the
  CatMeows dataset stays CC BY 4.0 and is not redistributed), and a README with
  the verified results table.

[Unreleased]: https://github.com/sbor3937/MeowSense/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/sbor3937/MeowSense/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/sbor3937/MeowSense/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/sbor3937/MeowSense/releases/tag/v0.1.0
