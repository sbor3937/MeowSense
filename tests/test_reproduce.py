"""Reproduction tests: the README numbers must actually hold.

These require the real CatMeows dataset and are marked ``data`` so they can be
skipped offline with ``pytest -m "not data"``. CI runs them against a cached
download; locally they auto-skip if ``data/raw`` is absent (run
``python src/download_data.py`` to enable them).

The point of this file is to make the central claim of the project -- "every
number in the README reproduces" -- a test that fails loudly if it ever stops
being true.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DATA_DIR = REPO_ROOT / "data" / "raw"

pytestmark = pytest.mark.data

# Skip the whole module cleanly if the dataset has not been downloaded.
if not DATA_DIR.exists() or not any(DATA_DIR.glob("*.wav")):
    pytest.skip(
        "CatMeows not present; run `python src/download_data.py`",
        allow_module_level=True,
    )

pytest.importorskip("sklearn", reason="scikit-learn required for baselines")

from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from features import extract_feature_matrix, scan_dataset  # noqa: E402
from train_baseline import build_models, majority_baseline  # noqa: E402

SEED = 42
N_SPLITS = 5


@pytest.fixture(scope="module")
def features():
    """MFCC features for the whole dataset, extracted once for the module."""
    recs = scan_dataset(DATA_DIR)
    return extract_feature_matrix(recs, kind="mfcc")


def _grouped_cv_accuracy(name, X, y, groups):
    """Mean GroupKFold accuracy for one named model, seeded."""
    accs = []
    for train_idx, test_idx in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        model = build_models(SEED)[name]
        model.fit(X[train_idx], y[train_idx])
        accs.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))
    return float(np.mean(accs))


def test_dataset_is_complete(features):
    X, y, groups = features
    assert X.shape == (440, 52)
    assert len(set(groups)) == 21


def test_class_balance_matches_readme(features):
    _, y, _ = features
    counts = {c: int((y == c).sum()) for c in ("B", "F", "I")}
    assert counts == {"B": 127, "F": 92, "I": 221}


def test_majority_baseline_is_half(features):
    X, y, groups = features
    bases = [
        majority_baseline(y[tr], y[te])
        for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, y, groups)
    ]
    assert np.mean(bases) == pytest.approx(0.50, abs=0.02)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("RandomForest (MFCC)", 0.49),
        ("SVM-RBF (MFCC)", 0.52),
    ],
)
def test_baseline_accuracy_matches_readme(features, name, expected):
    X, y, groups = features
    acc = _grouped_cv_accuracy(name, X, y, groups)
    # Tolerance absorbs small BLAS / library-version drift while still catching
    # any real regression in the pipeline.
    assert acc == pytest.approx(expected, abs=0.03), (
        f"{name}: got {acc:.3f}, README claims {expected:.2f}"
    )
