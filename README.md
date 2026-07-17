# MeowSense

[![Version](https://img.shields.io/badge/version-0.1.1-blue.svg)](CHANGELOG.md)
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

| Model | Accuracy (unseen cats) | Baseline |
|---|---|---|
| RandomForest (MFCC) | 0.49 ± 0.12 | 0.50 |
| SVM-RBF (MFCC) | 0.52 ± 0.07 | 0.50 |
| SmallCNN (mel-spec, from scratch) | 0.47 ± 0.06 | 0.50 |

**None of these models meaningfully beats the majority-class baseline.** That is the result. Reproduce it with `python src/train_baseline.py` and `python src/train_cnn.py --cv 5`.

### Why the honest number is so much lower than a random split

The same models, same features, evaluated two ways:

| Model | Random stratified split | Grouped by cat | Inflation |
|---|---|---|---|
| RandomForest (MFCC) | 0.686 | 0.494 | **+0.192** |
| SVM-RBF (MFCC) | 0.739 | 0.515 | **+0.223** |

A random split scatters one cat's clips across both train and test. The model then learns to recognise *which cat is meowing* — a cat it has already heard — rather than *what the meow means*, and reports ~0.70. That number is an artifact. With only 21 cats and 440 clips, it is very easy to produce by accident.

This gap is the main reason this repository exists. Reproduce it in [`notebooks/02_baselines.ipynb`](notebooks/02_baselines.ipynb).

### Key findings

- **`isolation` is the most separable class.** It has the highest precision of the three (0.64 RandomForest, 0.69 SVM, 0.74 CNN) and the best F1 (0.58 / 0.65 / 0.60). Distress calls are longer, louder and more tonal, and it is the one context with a plausible acoustic signature. Recall is nevertheless only ~0.5–0.6: "most separable" is not "solved".
- **`food` is not recoverable from audio alone** (recall 0.22–0.41, the worst class for every model). "Waiting for food" describes a *situation*, not a sound. A cat asking for dinner and a cat enjoying a brush can vocalize near-identically. **A user-supplied context signal — time since last meal — would help more than any better classifier.**
- **A CNN trained from scratch is not justified at this data size.** SmallCNN (23.6k parameters) scores below both the linear-ish baselines and the majority class. 300-odd training clips cannot teach general acoustic structure from zero. Transfer learning from AudioSet-pretrained models (YAMNet / PANNs / AST) is the sensible next step — see [`docs/ROADMAP.md`](docs/ROADMAP.md).
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

`torch` is only needed for `src/train_cnn.py`. Skip it if you only want the MFCC baselines.

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

### 4. Explore the notebooks

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

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for detail. In priority order:

1. **Transfer learning** — frozen YAMNet / PANNs / AST embeddings + a linear probe, before any fine-tuning. Highest leverage, needs no new data. Target: clear 0.60 on unseen cats.
2. **Telegram collection bot** — grow from 21 cats to hundreds. Owners send a voice note, the bot guesses, and they answer *guessed / wrong / don't know*. Cat ID mandatory; consent and deletion handled properly from day one.
3. **A larger label space** — add `greeting`, `play`, and (carefully, with veterinary confirmation) `pain`.

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
