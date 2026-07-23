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

- **More data — Telegram collection bot.** Now the binding constraint on every
  front: it would validate the RMS gain on unselected data, firm up the
  0.54–0.60 band, and let the label space grow.
- **Promote RMS normalization to the default pipeline**, once it can be
  validated on data that was not used to select it.
- **Partial fine-tuning** — impractical on CPU (an AST forward pass alone is
  ~4 s/clip); needs a GPU.
- **Version pinning** (dependency upper bounds), now that CI can catch breakage.

## [0.4.0] - 2026-07-21

Channel normalization: the recording setup was costing ~0.07 accuracy, and most
of it is recoverable for free.

### Added
- **`src/experiment_channel_norm.py`** — compares six ways of stripping the
  recording channel (RMS gain, per-clip CMN/CMVN, per-cat and per-owner
  centering), clearly separating inductive variants from transductive ones, and
  optionally repeats the test on cached embeddings (`--embeddings ast`).
- **`rms_normalize()`** and **`cepstral_normalize()`** in `src/features.py`;
  `extract_feature_matrix()` gains `rms_norm=` and `cmvn=` options.
- **`--rms-norm`** flag on `src/train_baseline.py`.
- Twelve tests covering the new primitives, including gain-invariance and a
  regression test for the degenerate-input bug below.

### Results
- **RMS gain normalization recovers +0.074** for RandomForest (0.494 → 0.569)
  and roughly **halves fold variance** (0.120 → 0.069); the worst fold rises
  from 0.27 to 0.44. Across 10 seeds the baseline and RMS ranges are disjoint.
  It is fully inductive — nothing about the test cats is used.
- **Removing absolute loudness improves distress detection** (isolation F1
  0.58 → 0.70): recorded loudness tracked mic distance more than the cat.
- **CMVN hurts** (−0.028 RF, −0.066 SVM) — equalizing per-coefficient variance
  destroys real signal; only the channel mean should go.
- **Pretrained embeddings barely benefit** (+0.02 at best; −0.10 for the AST
  LogReg probe), because they are already largely channel-invariant. This
  reframes the transfer-learning result: a meaningful share of what an
  86M-parameter backbone buys here is channel invariance available for free.
- **Not promoted to the default pipeline.** The winner was selected among six
  variants on the same folds used for reporting, so +0.074 is an optimistic
  estimate. The effect is well-corroborated (two independent routes agree;
  disjoint seed ranges); the magnitude is not clean. Opt-in until it can be
  validated on independent data. Headline table unchanged.

### Fixed
- `cepstral_normalize(mode="meanvar")` divided near-zero variance by itself on
  near-constant input, amplifying float32 rounding noise to full scale. Found
  by its own unit test. Degenerate coefficients are now left unscaled.

## [0.3.0] - 2026-07-20

Independent replication of the transfer-learning result — and a correction to
how strongly it should be stated.

### Added
- **Second backbone: [CLAP](https://huggingface.co/laion/clap-htsat-unfused)**
  (HTSAT audio encoder, audio-text contrastive, 48 kHz). Chosen because it is
  genuinely independent of AST — different architecture, pretraining objective
  and input rate — so agreement between them is a result about the *task*, not
  about one model. Extraction takes ~3 min on CPU vs AST's ~30 min.
- `BACKBONES` registry and `resolve_backbone()` in `src/embeddings.py`; adding a
  backbone is now a config entry rather than a fork of the extraction code.
- `--model ast|clap` on both `src/embeddings.py` and `src/train_transfer.py`.
- Five tests covering the registry, including an explicit check that the two
  backbones really are distinct (architecture, sample rate, model id).

### Results
- **The direction replicates, the magnitude does not.** All four backbone ×
  probe configurations beat the 0.50 baseline (+0.04 to +0.10) and every
  from-scratch model — but CLAP tops out at **0.56 ± 0.07** against AST's
  **0.60 ± 0.10**, and AST wins only 3 of 5 folds head-to-head. Which probe
  wins even flips by backbone (SVM for AST, LogReg for CLAP).
- README now reports a **0.54–0.60 band** rather than presenting 0.60 as *the*
  number for this task. The backbone-to-backbone spread is smaller than the
  fold-to-fold spread, which points at the dataset, not the architecture, as
  the limiting factor.
- `food` fails on **both** backbones (F1 ≤ 0.37, CLAP-SVM only 0.28) —
  two unrelated models trained on millions of clips both missing it is the
  strongest evidence yet that it is a label/context problem.
- All figures deterministic across seeds.

### Changed
- Probe names in `train_transfer.py` output are now backbone-agnostic
  (`LogReg probe`, `SVM-RBF probe`) since two backbones share them.
- `--model` now accepts short names (`ast`, `clap`) as well as raw model ids.
- CLAP embedding extraction handles the `audios` → `audio` processor-argument
  rename in transformers 5.x, keeping the declared `>=4.40` floor honest.

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

[Unreleased]: https://github.com/sbor3937/MeowSense/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/sbor3937/MeowSense/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/sbor3937/MeowSense/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/sbor3937/MeowSense/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/sbor3937/MeowSense/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/sbor3937/MeowSense/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/sbor3937/MeowSense/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/sbor3937/MeowSense/releases/tag/v0.1.0
