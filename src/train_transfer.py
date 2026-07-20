"""Linear probe on frozen AudioSet-pretrained embeddings.

This is the transfer-learning baseline: freeze an AudioSet-pretrained backbone,
extract one embedding per clip (see :mod:`embeddings`), and train only a small
classifier on top. Nothing is fine-tuned.

Frozen-probe-first is deliberate. It is fast, it is very hard to overfit with
~300 training clips, and it answers the question that actually matters before
any fine-tuning is attempted: **does a model that has heard 2M AudioSet clips
already encode something about cat vocalization context that 13 MFCCs do not?**
If a linear probe on those embeddings cannot beat the MFCC baselines, the
ceiling is the labels, not the representation -- which would itself be worth
knowing.

The validation protocol is identical to every other script here: 5-fold
GroupKFold over cat IDs, so the probe is only ever scored on cats it has never
heard. The numbers are therefore directly comparable to the README table.

Usage::

    python src/train_transfer.py
    python src/train_transfer.py --seed 7 --n-splits 5
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embeddings import DEFAULT_CACHE_DIR, DEFAULT_MODEL, extract_embeddings  # noqa: E402
from features import CONTEXT_LABELS  # noqa: E402
from train_baseline import majority_baseline, print_confusion  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_SEED = 42
CLASS_ORDER = ["B", "F", "I"]


def set_seed(seed: int) -> None:
    """Seed the RNGs this script can reach."""
    random.seed(seed)
    np.random.seed(seed)


def build_probes(seed: int) -> dict[str, Pipeline]:
    """Construct the probe models.

    Two heads are fitted on the same frozen embeddings:

    * **Logistic regression** -- the canonical linear probe. If the pretrained
      representation linearly separates the contexts, this finds it.
    * **SVM-RBF** -- a non-linear head, included so the comparison against the
      MFCC SVM in the README changes exactly one thing (the features).

    Both standardize first (embedding dimensions have wildly different scales)
    and use balanced class weights against the 221/127/92 imbalance. Scaling is
    fitted inside each fold.

    Args:
        seed: Random seed for the estimators.

    Returns:
        Mapping of model name -> unfitted pipeline.
    """
    return {
        "LogReg probe (AST emb.)": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000,
                        C=1.0,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "SVM-RBF (AST emb.)": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=10.0,
                        gamma="scale",
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def evaluate_grouped_cv(
    models: dict[str, Pipeline],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> dict[str, dict]:
    """Score each probe with GroupKFold over cats.

    Args:
        models: Name -> unfitted pipeline.
        X: Embedding matrix.
        y: Context labels.
        groups: Cat IDs.
        n_splits: Number of folds.

    Returns:
        Name -> dict with ``accuracies``, ``baseline``, ``y_true``, ``y_pred``.
    """
    results = {}

    for name, model in models.items():
        accs, bases, y_true, y_pred = [], [], [], []
        for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
            assert not set(groups[train_idx]) & set(groups[test_idx]), "cat leaked"
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            accs.append(accuracy_score(y[test_idx], pred))
            bases.append(majority_baseline(y[train_idx], y[test_idx]))
            y_true.extend(y[test_idx])
            y_pred.extend(pred)

        accs = np.array(accs)
        results[name] = {
            "accuracies": accs,
            "baseline": float(np.mean(bases)),
            "y_true": np.array(y_true),
            "y_pred": np.array(y_pred),
        }

        print(f"{name:<26} acc = {accs.mean():.3f} +/- {accs.std():.3f}   "
              f"(baseline {np.mean(bases):.3f})   "
              f"folds: {', '.join(f'{a:.2f}' for a in accs)}")

    return results


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Linear probe on frozen AudioSet-pretrained embeddings.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    try:
        X, y, groups = extract_embeddings(
            data_dir=args.data_dir,
            model_name=args.model,
            cache_dir=args.cache_dir,
            force=args.force_extract,
        )
    except (FileNotFoundError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nEmbeddings: X={X.shape}  cats={len(set(groups))}")
    counts = {CONTEXT_LABELS[c]: int((y == c).sum()) for c in CLASS_ORDER}
    print(f"class counts: {counts}")

    print("\n" + "=" * 72)
    print(f"GROUPED {args.n_splits}-FOLD CROSS-VALIDATION (every cat tested once)")
    print("=" * 72)
    results = evaluate_grouped_cv(
        build_probes(args.seed), X, y, groups, args.n_splits
    )

    for name, res in results.items():
        print("\n" + "-" * 72)
        print(f"{name} -- pooled over all folds")
        print("-" * 72)
        print(
            classification_report(
                res["y_true"],
                res["y_pred"],
                labels=CLASS_ORDER,
                target_names=[CONTEXT_LABELS[c] for c in CLASS_ORDER],
                zero_division=0,
            )
        )
        print("confusion matrix (rows = true, cols = predicted):")
        print_confusion(
            confusion_matrix(res["y_true"], res["y_pred"], labels=CLASS_ORDER),
            CLASS_ORDER,
        )

    print("\n" + "=" * 72)
    print("SUMMARY vs the MFCC baselines (same GroupKFold protocol)")
    print("=" * 72)
    best = max(results.items(), key=lambda kv: kv[1]["accuracies"].mean())
    base = next(iter(results.values()))["baseline"]
    for name, res in results.items():
        print(f"{name:<26} {res['accuracies'].mean():.3f} +/- {res['accuracies'].std():.3f}")
    print(f"{'-- majority baseline':<26} {base:.3f}")
    print(f"{'-- RandomForest (MFCC)':<26} 0.494")
    print(f"{'-- SVM-RBF (MFCC)':<26} 0.515")
    print(f"{'-- SmallCNN (from scratch)':<26} 0.530")

    delta = best[1]["accuracies"].mean() - 0.530
    print(f"\nBest probe: {best[0]} at {best[1]['accuracies'].mean():.3f} "
          f"({delta:+.3f} vs the best from-scratch model)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
