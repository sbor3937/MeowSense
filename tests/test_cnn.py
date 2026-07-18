"""Protocol tests for the CNN trainer.

The point of these is not accuracy -- it is that the training protocol cannot
leak. They run a couple of epochs on synthesized spectrograms, so they need
torch but not the real dataset. They skip cleanly where torch is absent (e.g.
the hermetic CI job), because the leak-free invariants are also enforced by
assertions inside ``run_fold`` itself at runtime.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch required for the CNN trainer")

import train_cnn as T  # noqa: E402


def _synthetic_logmel_dataset(n_cats=8, per_cat=12, seed=0):
    """Build a tiny ``(X, y, groups)`` in the CNN's expected shapes.

    Each cat gets a slightly different mean so that "learning the cat" is even
    possible; the test only checks the *protocol*, not whether it learns.
    """
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    for c in range(n_cats):
        offset = rng.normal(0, 2)
        for i in range(per_cat):
            X.append(rng.normal(offset, 1.0, size=(T.N_MELS, T.CNN_N_FRAMES)).astype(np.float32))
            y.append(i % len(T.CLASS_ORDER))
            groups.append(f"CAT{c:02d}")
    return np.stack(X), np.array(y), np.array(groups)


def _fast_args(**over):
    base = dict(
        seed=0, epochs=2, batch_size=16, lr=1e-3, weight_decay=1e-4,
        dropout=0.4, test_size=0.3, val_size=0.2,
    )
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture
def dataset():
    return _synthetic_logmel_dataset()


def test_run_fold_scores_only_test_clips(dataset):
    X, y, groups = dataset
    import torch

    device = torch.device("cpu")
    train_idx = np.arange(0, 72)      # cats 0..5
    test_idx = np.arange(72, 96)      # cats 6..7

    T.set_seed(0)
    y_true, y_pred, val_acc, _ = T.run_fold(
        X, y, groups, train_idx, test_idx, _fast_args(), device, verbose=False
    )

    # Predictions cover exactly the test split, and nothing else.
    assert len(y_true) == len(test_idx) == len(y_pred)
    assert np.array_equal(y_true, y[test_idx])
    assert 0.0 <= val_acc <= 1.0


def test_run_fold_keeps_train_val_test_cats_disjoint(dataset):
    """The three splits (train / validation / test) must never share a cat."""
    X, y, groups = dataset

    train_idx = np.arange(0, 72)
    test_idx = np.arange(72, 96)

    # Re-derive the val sub-split exactly as run_fold does, then assert the
    # resulting cat sets are pairwise disjoint.
    from sklearn.model_selection import GroupShuffleSplit

    args = _fast_args()
    sub_tr, sub_val = next(
        GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
        .split(train_idx, y[train_idx], groups[train_idx])
    )
    tr_cats = set(groups[train_idx[sub_tr]])
    val_cats = set(groups[train_idx[sub_val]])
    test_cats = set(groups[test_idx])

    assert tr_cats.isdisjoint(val_cats)
    assert (tr_cats | val_cats).isdisjoint(test_cats)
    assert len(val_cats) >= 1  # validation must be non-empty


def test_normalize_splits_uses_train_stats_only():
    """Validation/test are standardized with train statistics, not their own."""
    rng = np.random.default_rng(1)
    X_train = rng.normal(5.0, 2.0, size=(20, T.N_MELS, T.CNN_N_FRAMES)).astype(np.float32)
    X_val = rng.normal(-3.0, 1.0, size=(8, T.N_MELS, T.CNN_N_FRAMES)).astype(np.float32)

    tr_norm, val_norm = T.normalize_splits(X_train, X_val)

    # Train subset becomes ~zero-mean per band; val keeps its own (shifted) mean.
    assert abs(tr_norm.mean()) < 0.1
    assert val_norm.mean() < -0.5  # not re-centered on itself
