# MeowSense

[![CI](https://github.com/sbor3937/MeowSense/actions/workflows/ci.yml/badge.svg)](https://github.com/sbor3937/MeowSense/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](CHANGELOG.md)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**Open, reproducible benchmarks for classifying cat vocalizations — and an honest account of how hard the problem actually is.**

---

## Why I'm building this

There are hundreds of millions of us — people who live with a cat and quietly wish we understood it a little better. I'm one of them. Every owner has stood in the kitchen at 6 a.m. trying to figure out whether that meow means *food*, *attention*, or *leave me alone* — and mostly guessing.

I started MeowSense because I wanted to see how far an honest, open approach could actually get on that question. Not a magic "cat translator" — I'm skeptical anyone truly has one — but a real, reproducible baseline that other cat lovers, students and researchers can build on instead of starting from zero.

Bioacoustics already has a healthy open-source ecosystem for birds, whales and bats. For the animal most of us actually share our homes with, there's almost nothing open. What exists is closed: apps like MeowTalk ship a translator to consumers, but the models, the data and — most importantly — the evaluation protocol are proprietary. There's no way to check whether the accuracy means anything.

MeowSense is my attempt to put an honest starting line in the open:

- **A reproducible benchmark** on the public CatMeows dataset — fixed seeds, one command per result, no hidden preprocessing.
- **A validation protocol that doesn't lie.** Every result is measured on cats the model has never heard. This turns out to matter enormously (see [Results](#results)).
- **An honest negative result.** Our models land near the majority-class baseline. I report that rather than tuning until a nicer number appears.
- **A path forward** in [`docs/ROADMAP.md`](docs/ROADMAP.md): transfer learning from AudioSet-pretrained models, and a Telegram bot to crowdsource a much larger dataset — so that one day the answer to "what does my cat want" is a little less of a guess.

---

## Dataset

[**CatMeows**](https://doi.org/10.5281/zenodo.4008297) (Zenodo record `4008297`):

| | |
|---|---|
| Recordings | 440 |
| Cats | 21 |
| Contexts | 3 — brushing (127), waiting for food (92), isolation in an unfamiliar environment (221) |
| Breeds | European Shorthair (252), Maine Coon (188) |
| Audio | mono WAV, 8 kHz |
| Duration | 1.09 – 4.00 s (median 1.81 s) |
| License | CC BY 4.0, intended for non-commercial research use |

The dataset is **not redistributed here**. `src/download_data.py` fetches it from Zenodo so that attribution stays with the original authors. `data/` is gitignored.

Metadata is encoded in each filename (`C_NNNNN_BB_SS_OOOOO_RXX.wav`) — context, cat ID, breed, sex and owner. `src/features.py` parses it. **The cat ID is the important field**: it is what makes leak-free validation possible.

---

## Results

All numbers below are **5-fold GroupKFold grouped by cat ID** — each cat is entirely in train or entirely in test, so every test clip comes from a cat the model has never heard. Reported as mean ± std across folds. Baseline = always predict the majority class (`isolation`, 50.2% of the data).

| Model | Features | Accuracy (unseen cats) | Baseline |
|---|---|---|---|
| RandomForest | MFCC | 0.49 ± 0.12 | 0.50 |
| SVM-RBF | MFCC | 0.52 ± 0.07 | 0.50 |
| SmallCNN (from scratch) | mel-spec | 0.53 ± 0.06 | 0.50 |
| LogReg probe | AST embeddings *(frozen)* | 0.58 ± 0.08 | 0.50 |
| **SVM-RBF** | **AST embeddings *(frozen)*** | **0.60 ± 0.10** | 0.50 |

**Two findings, one negative and one positive.**

1. **Nothing computed *from the audio itself* beats the baseline.** The MFCC models and the from-scratch CNN all land at 0.49–0.53 — at or within noise of "always guess isolation". More network does not help.
2. **A better prior does.** Freezing an AudioSet-pretrained backbone (AST, 86M params) and training a linear-ish probe on its embeddings reaches **0.60 ± 0.10** — the first model here to clear the baseline, by roughly one fold-std. It beats its per-fold baseline in 4 of 5 folds and the MFCC SVM by +0.09 on average. The gain is real but modest, and it concentrates almost entirely in `isolation`.

Reproduce with `python src/train_baseline.py`, `python src/train_cnn.py --cv 5`, and `python src/train_transfer.py`.

> **Caveat on the AST numbers.** CatMeows is 8 kHz; AST expects 16 kHz. Upsampling restores the sample rate but not the missing 4–8 kHz band, so the backbone sees a spectrogram with an empty top half. The 0.60 is achieved *despite* that handicap — and is a direct argument for recording future data at 16 kHz+. Do not compare it against literature trained on native 16 kHz audio.

### Why the honest number is so much lower than a random split

The same models, same features, evaluated two ways:

| Model | Random stratified split | Grouped by cat | Inflation |
|---|---|---|---|
| RandomForest (MFCC) | 0.686 | 0.494 | **+0.192** |
| SVM-RBF (MFCC) | 0.739 | 0.515 | **+0.223** |

A random split scatters one cat's clips across both train and test. The model then learns to recognise *which cat is meowing* — a cat it has already heard — rather than *what the meow means*, and reports ~0.70. That number is an artifact. With only 21 cats and 440 clips, it is very easy to produce by accident.

This gap is the main reason this repository exists. Reproduce it in [`notebooks/02_baselines.ipynb`](notebooks/02_baselines.ipynb).

### Key findings

- **Transfer learning is the payoff, and it lands on `isolation`.** The AST probe's whole advantage is on the isolation/stress class: F1 0.76 (vs 0.65 for the best MFCC model), precision 0.79. A model that has heard 2M AudioSet clips recognises a distress call better than 13 MFCCs do — but its brushing and food F1 barely move. The pretrained prior sharpens the one class that has an acoustic signature; it does not conjure signal where there is none.
- **`isolation` is the most separable class for every model.** It is each model's best-recovered class — isolation F1 runs 0.58 / 0.65 / 0.68 / 0.76 for RandomForest / MFCC-SVM / CNN / AST-SVM. Distress calls are longer, louder and more tonal, the one context with a plausible acoustic signature.
- **`food` is not recoverable from audio alone — not even by an 86M-parameter model** (F1 ≤ 0.37 for every model, including the AST probe). "Waiting for food" describes a *situation*, not a sound: a cat asking for dinner and a cat enjoying a brush vocalize near-identically. That the pretrained backbone cannot recover it either is strong evidence this is a **label/context problem, not a representation problem**. **A user-supplied signal — time since last meal — would help more than any better model.**
- **A from-scratch CNN ties the SVM and earns nothing for it.** With a proper validation-based stopping rule (a cat-grouped validation split picks the epoch; the test set is touched once), SmallCNN reaches 0.53 ± 0.06 — level with the MFCC SVM and, like it, within noise of the 0.50 baseline. It needs far more compute than the baselines to match them and beats neither. *(An earlier fixed-60-epoch schedule under-reported this at 0.47 by scoring an arbitrary late epoch — a reminder that the protocol, not just the model, decides the number.)* 300-odd clips cannot teach general acoustic structure from zero; the way to add real signal is a better prior, not a bigger network — which the AST probe above confirms.
- **Fold variance swamps model choice.** Per-fold accuracy ranges 0.27–0.63. With 4–5 cats per fold, *which* cats you hold out matters more than which model you pick. The binding constraint is the number of cats, not the architecture.
- **Validation is strictly grouped by cat ID.** No clip from a test cat is ever seen during training. Assertions in `train_baseline.py` and `train_cnn.py` enforce this rather than trusting it.

---

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/sbor3937/MeowSense.git
cd MeowSense

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`torch` is only needed for `src/train_cnn.py`. The transfer-learning path also needs `transformers` (a ~340 MB AudioSet model is downloaded on first run):

```bash
pip install transformers        # only for src/embeddings.py + src/train_transfer.py
```

Skip both if you only want the MFCC baselines.

---

## Usage

### 1. Download the dataset

```bash
python src/download_data.py
```

Fetches `dataset.zip` (~8.9 MB) from Zenodo and extracts 440 WAV files to `data/raw/`. Idempotent — re-running skips the download. Options: `--with-extras`, `--force`, `--data-dir PATH`.

### 2. Train the MFCC baselines

```bash
python src/train_baseline.py
```

RandomForest + SVM-RBF on 52-dim MFCC summary vectors. Prints a hold-out evaluation (classification report + confusion matrix) and the 5-fold GroupKFold numbers from the table above.

```bash
python src/train_baseline.py --save-plots        # confusion matrices -> artifacts/
python src/train_baseline.py --seed 7 --n-splits 5
```

### 3. Train the CNN

```bash
python src/train_cnn.py --cv 5        # the number in the README table
python src/train_cnn.py               # single hold-out, faster
```

SmallCNN on 40×128 log-mel spectrograms. Options: `--epochs`, `--lr`, `--batch-size`, `--dropout`, `--save-model PATH`.

### 4. Train the transfer-learning probe

```bash
pip install transformers
python src/train_transfer.py
```

Extracts frozen [AST](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) embeddings (cached to `data/embeddings/` — the first run is a ~30 min CPU pass, then instant) and fits a LogReg / SVM probe under the same GroupKFold protocol. This is the 0.60 row in the table. `src/embeddings.py` runs the extraction on its own if you want to cache first.

### 5. Explore the notebooks

```bash
jupyter notebook notebooks/
```

- [`01_eda.ipynb`](notebooks/01_eda.ipynb) — context / breed / sex / duration distributions, spectral centroids per context, example spectrograms.
- [`02_baselines.ipynb`](notebooks/02_baselines.ipynb) — trains the baselines, demonstrates the leakage gap, per-class analysis.

Both run top-to-bottom against a freshly downloaded dataset.

### Feature pipeline

`src/features.py` depends only on numpy and scipy — no librosa. Defaults, all chosen for 8 kHz audio:

| Setting | Value |
|---|---|
| Sample rate | 8000 Hz |
| FFT size | 256 (32 ms) |
| Hop | 128 (16 ms) |
| Mel filters | 40, spanning 50–4000 Hz |
| MFCCs | 13 (+ deltas) |
| CNN input | 40 × 128 (≈2.05 s, centre-cropped / padded) |

---

## Development

Run the tests with [pytest](https://docs.pytest.org):

```bash
pip install pytest

pytest -m "not data"   # 40 hermetic unit tests, no dataset needed (~5 s)
pytest                 # also runs the reproduction tests (needs data/raw)
```

The unit tests synthesize their own audio and cover the DSP path and metadata
parsing. The `data`-marked tests assert that the numbers in the [Results](#results)
table still hold on the real dataset; they auto-skip if `data/raw` is absent.

Both run in [CI](.github/workflows/ci.yml) on every push — the `reproduce` job
downloads CatMeows and re-checks the baseline accuracies, so a regression that
changed a reported number would fail the build.

---

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for detail.

- ✅ **Transfer learning** — done. Frozen AST embeddings + a linear probe hit 0.60 on unseen cats (the target), the first model here to beat the baseline. Next within this thread: try YAMNet/PANNs, and partial fine-tuning.

In priority order from here:

1. **More data — a Telegram collection bot** — grow from 21 cats to hundreds. Owners send a voice note, the bot guesses, and they answer *guessed / wrong / don't know*. Cat ID mandatory; consent and deletion handled properly from day one. With the AST probe now capped at 0.60 partly by the 8 kHz audio and the tiny cat count, data is the binding constraint again.
2. **A larger label space** — add `greeting`, `play`, and (carefully, with veterinary confirmation) `pain`.
3. **Channel normalization** — decouple loudness (the `c0` coefficient, ~69% cat identity) from the context signal.

**Non-goals:** shipping a "cat translator"; beating published numbers by relaxing the grouped-validation rule.

---

## Contributing

PRs welcome. Transfer learning (roadmap item 1) is self-contained, needs only the existing dataset, and is the best entry point. See [CONTRIBUTING.md](CONTRIBUTING.md).

One rule above all others: **results must be reported on cats the model has not seen.** A PR that improves accuracy by relaxing grouped validation will not be merged. If a number in this README cannot be reproduced by running the documented command, that is a bug — please open an issue.

---

## License

**Code** in this repository: [MIT](LICENSE).

**Data**: the CatMeows dataset is *not* covered by that licence and is not redistributed here. It is published by its authors on Zenodo under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with the stated intent that it be used for **non-commercial research purposes**. If you use it, attribute the original authors and respect those terms.

---

## Acknowledgements

The CatMeows dataset is the work of **Luca Andrea Ludovico, Stavros Ntalampiras, Giorgio Presti, Simona Cannas, Monica Battini and Silvana Mattiello** (University of Milan). This project would not exist without their decision to publish it openly.

> Ludovico, L. A., Ntalampiras, S., Presti, G., Cannas, S., Battini, M., & Mattiello, S. (2020). *CatMeows: A Publicly-Available Dataset of Cat Vocalizations* (Version 1.0.2) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.4008297

Related paper by the same group:

> Ntalampiras, S., Ludovico, L. A., Presti, G., Prato Previde, E., Battini, M., Cannas, S., Palestrini, C., & Mattiello, S. (2019). Automatic Classification of Cat Vocalizations Emitted in Different Contexts. *Animals*, 9(8), 543. https://doi.org/10.3390/ani9080543

Please cite their work, not this repository, when using the dataset.
