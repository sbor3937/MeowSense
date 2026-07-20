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

- **More data — Telegram collection bot** (now the binding constraint).
- **Transfer learning, continued** — try YAMNet / PANNs; partial fine-tuning.
- **Channel-normalization experiment** — decouple loudness (the `c0` cepstral
  coefficient, ~69% cat identity / ~5% context) from the context signal.
- **Version pinning** (dependency upper bounds), now that CI can catch breakage.

## [0.2.0] - 2026-07-20

First model to beat the baseline: transfer learning.

### Added
- **`src/embeddings.py`** — extracts frozen [AST](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
  (AudioSet-pretrained) embeddings, mean-pooled to one 768-d vector per clip,
  cached to `data/embeddings/`. Resamples 8 kHz → 16 kHz (documented as a
  handicap, not a fix — the missing 4–8 kHz band is not restored).
- **`src/train_transfer.py`** — LogReg and SVM probes on the frozen embeddings,
  under the identical 5-fold GroupKFold-by-cat protocol, so results are directly
  comparable to the MFCC/CNN table.
- **`tests/test_transfer.py`** — hermetic tests for the embedding cache
  round-trip, probe construction and grouped-CV evaluation (need scikit-learn,
  not `transformers`/`torch`/the dataset).
- `transfer` optional-dependency group (`transformers`) in `pyproject.toml`.

### Results
- **AST embeddings + SVM: 0.60 ± 0.10** on unseen cats — clears the roadmap's
  0.60 target and is the first model here to beat the 0.50 baseline (LogReg
  probe: 0.58 ± 0.08). Deterministic across 10 seeds.
- The advantage is +0.09 over the MFCC SVM on average and concentrates in
  `isolation` (F1 0.76 vs 0.65). `food` stays unrecoverable (F1 0.35) even for
  the 86M-parameter backbone — evidence it is a context problem, not a
  representation problem.
- README results table, key findings and roadmap updated accordingly.

### Changed
- CI unit job now installs scikit-learn so the probe plumbing tests run there
  (`transformers` and `torch` stay out of CI).

## [0.1.3] - 2026-07-19

### Changed
- **Fixed the CNN validation protocol.** The training cats are now split again
  into a train subset and a cat-grouped **validation** subset; validation
  accuracy drives both per-epoch monitoring and best-checkpoint selection, and
  the **test split is evaluated exactly once**. The previous loop watched the
  test set every 10 epochs and reported an arbitrary fixed epoch. Normalization
  statistics now come from the train subset only.
- **Result:** with a principled stopping rule, SmallCNN reaches **0.53 ± 0.06**
  (up from 0.47 — the old fixed-epoch schedule under-reported it by scoring a
  late, over-trained epoch). It ties the MFCC SVM and still sits within noise of
  the 0.50 baseline; the "from scratch is not justified" conclusion stands, via
  a corrected argument. README and roadmap updated accordingly.
- `--val-size` flag added to `src/train_cnn.py` (default 0.2).

### Performance
- **Cached the mel filterbank** (`@lru_cache`), which is identical across all
  clips at fixed settings. This removed roughly half of feature-extraction
  time. The cached array is returned read-only so a stray in-place write cannot
  corrupt other callers.

### Added
- `tests/test_cnn.py` — protocol tests asserting the train / validation / test
  cat sets stay disjoint and that only the test split is scored (torch-guarded;
  skips where torch is absent).

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

[Unreleased]: https://github.com/sbor3937/MeowSense/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/sbor3937/MeowSense/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/sbor3937/MeowSense/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/sbor3937/MeowSense/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/sbor3937/MeowSense/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/sbor3937/MeowSense/releases/tag/v0.1.0
