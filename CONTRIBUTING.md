# Contributing to MeowSense

Thanks for considering a contribution. This project is small and its goals are
narrow, which makes it easy to help — and easy to describe what "help" means.

## The one rule

**Results must be reported on cats the model has never heard.**

Every evaluation groups by `cat_id`. This is not a stylistic preference; it is
the entire point of the project. On CatMeows, a random split inflates accuracy by
roughly 20 points (0.49 → 0.69) because the model learns to recognise the
individual animal rather than the meaning of the call. Numbers produced that way
look good and mean nothing.

A PR that improves accuracy by relaxing grouped validation will not be merged. A
PR that *lowers* a reported number by fixing a leak is very welcome.

If you find a number in the README that you cannot reproduce by running the
documented command, that is a bug. Please open an issue.

## Good first contributions

- **Transfer learning** (roadmap item 1) — frozen YAMNet / PANNs / AST embeddings
  plus a linear probe. Self-contained, needs only the existing dataset, and the
  highest-leverage change available. Start here.
- **More baselines** — logistic regression, gradient boosting, a GMM/HMM. Cheap
  to add, and useful for showing where the real ceiling is.
- **Better features** — pitch tracking, jitter/shimmer, formant estimates.
  Feline distress calls plausibly differ in ways MFCCs smooth over.
- **Tests** — `src/features.py` has none. The filename parser and the mel
  filterbank are both easy to test and easy to get subtly wrong.
- **Documentation fixes** — including telling us where the docs are unclear.

Roadmap item 2 (the Telegram bot) needs someone comfortable owning a
privacy/consent surface, not just bot code. Please open an issue to discuss
before starting.

## Development setup

```bash
git clone https://github.com/sbor3937/MeowSense.git
cd MeowSense

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/download_data.py      # ~8.9 MB from Zenodo
python src/train_baseline.py     # should reproduce the README table
```

## Code style

Match what is already there rather than importing your own conventions:

- Docstrings on public functions, Google style, with `Args:` / `Returns:` /
  `Raises:` where they apply.
- Type hints on signatures.
- Seed everything. `set_seed()` exists in both training scripts; use it.
- Comments explain *why*, not *what*. If a constant is non-obvious (`n_fft=256`
  because the audio is 8 kHz), say so; do not narrate the code.
- Keep `src/features.py` dependent on numpy and scipy only. The lack of a librosa
  dependency is deliberate — the DSP path should stay small and auditable.

## Submitting changes

1. Open an issue first for anything non-trivial, so effort is not wasted.
2. Branch from `main`.
3. If you change anything that affects a reported number, **re-run the affected
   script and update the README table in the same PR.** Include the command you
   ran and its output in the PR description.
4. Notebooks should be committed with their outputs executed, so they are
   readable on GitHub without running them.

## Reporting results

When adding a model to the results table, include:

- Accuracy on unseen cats, as mean ± std across GroupKFold folds — not a single
  favourable split.
- The majority-class baseline for comparison.
- The exact command that reproduces it.

Negative results are welcome and get merged. `src/train_cnn.py` is in this
repository specifically because it *does not work*, and knowing that a
from-scratch CNN fails at this data size is worth as much as knowing what
succeeds.

## Data and licensing

Do not commit audio. `data/` and `*.wav` are gitignored, and CatMeows is CC BY
4.0 with a non-commercial research intent — it stays on Zenodo, where
attribution lives with its authors.

Code contributions are accepted under the MIT licence of this repository.
