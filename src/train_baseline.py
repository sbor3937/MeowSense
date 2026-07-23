"""Classical MFCC baselines for cat vocalization context classification.

Trains a RandomForest and an RBF-kernel SVM on fixed-length MFCC summary
vectors (see :func:`features.mfcc_feature_vector`) to predict the emission
context -- brushing, waiting for food, or isolation.

**The validation protocol is the point of this script.** CatMeows contains only
21 cats but 440 recordings, so a naive random split puts the *same cat* in both
train and test. A model can then win by recognizing the individual animal's
voice rather than the meaning of the call, and accuracy looks far better than it
is. Every split here is therefore grouped by ``cat_id``: the model is only ever
scored on cats it has never heard. That is the number worth reporting.

Two evaluations are run:

* **GroupShuffleSplit** -- a single held-out group of cats, reported in full
  (classification report + confusion matrix). This is the headline number.
* **GroupKFold** -- every cat serves in the test fold exactly once, giving a
  mean +/- std across folds and a sense of how much the headline number moves
  depending on *which* cats you hold out.

Usage::

    python src/train_baseline.py
    python src/train_baseline.py --data-dir data/raw --seed 42
    python src/train_baseline.py --save-plots     # write confusion matrices
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import CONTEXT_LABELS, extract_feature_matrix, scan_dataset  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_SEED = 42
CLASS_ORDER = ["B", "F", "I"]


def set_seed(seed: int) -> None:
    """Seed every RNG this script can reach, for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)


def build_models(seed: int) -> dict[str, Pipeline]:
    """Construct the baseline models.

    Both are wrapped in a :class:`~sklearn.pipeline.Pipeline` with a
    :class:`~sklearn.preprocessing.StandardScaler`. The scaler is fitted inside
    each fold rather than on the full dataset, which keeps test-set statistics
    from leaking into training. It is essential for the SVM (RBF kernels are
    scale-sensitive) and harmless for the forest.

    ``class_weight="balanced"`` matters here: the contexts are imbalanced
    (221 isolation / 127 brushing / 92 food), so an unweighted model is tempted
    to just predict "isolation".

    Args:
        seed: Random seed for the estimators.

    Returns:
        Mapping of model name -> unfitted pipeline.
    """
    return {
        "RandomForest (MFCC)": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "SVM-RBF (MFCC)": Pipeline(
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


def majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Accuracy of always predicting the most frequent *training* class.

    This is the number every model must beat to be worth anything. It is
    computed from the training split only -- peeking at the test distribution
    to pick the majority class would itself be leakage.

    Args:
        y_train: Training labels.
        y_test: Test labels.

    Returns:
        Accuracy on ``y_test`` of the constant majority-class predictor.
    """
    values, counts = np.unique(y_train, return_counts=True)
    majority = values[counts.argmax()]
    return float((y_test == majority).mean())


def print_confusion(cm: np.ndarray, labels: list[str]) -> None:
    """Pretty-print a confusion matrix with readable row/column headers."""
    names = [CONTEXT_LABELS[c] for c in labels]
    width = max(len(n) for n in names) + 2

    header = " " * width + "".join(f"{n[:9]:>11}" for n in names)
    print(header)
    for i, name in enumerate(names):
        row = "".join(f"{v:>11}" for v in cm[i])
        print(f"{name:<{width}}{row}")


def evaluate_holdout(
    models: dict[str, Pipeline],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    test_size: float = 0.3,
) -> dict[str, dict]:
    """Evaluate on a single held-out set of unseen cats.

    Args:
        models: Name -> unfitted pipeline.
        X: Feature matrix, shape ``(n_samples, n_features)``.
        y: Context labels.
        groups: Cat IDs, used to keep each cat wholly on one side of the split.
        seed: Seed controlling the split and the estimators.
        test_size: Approximate fraction of *groups* held out.

    Returns:
        Name -> dict with ``accuracy``, ``baseline``, ``y_true``, ``y_pred``.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    train_cats = sorted(set(groups[train_idx]))
    test_cats = sorted(set(groups[test_idx]))
    assert not set(train_cats) & set(test_cats), "cat leaked across the split"

    print("=" * 72)
    print("HOLD-OUT EVALUATION (unseen cats)")
    print("=" * 72)
    print(f"train: {len(train_idx):3d} clips from {len(train_cats):2d} cats")
    print(f"test : {len(test_idx):3d} clips from {len(test_cats):2d} cats  {test_cats}")

    base = majority_baseline(y[train_idx], y[test_idx])
    print(f"\nmajority-class baseline on this test split: {base:.2f}")

    results = {}
    for name, model in models.items():
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], y_pred)

        print("\n" + "-" * 72)
        print(f"{name}   accuracy = {acc:.2f}   (baseline {base:.2f})")
        print("-" * 72)
        print(
            classification_report(
                y[test_idx],
                y_pred,
                labels=CLASS_ORDER,
                target_names=[CONTEXT_LABELS[c] for c in CLASS_ORDER],
                zero_division=0,
            )
        )
        print("confusion matrix (rows = true, cols = predicted):")
        print_confusion(
            confusion_matrix(y[test_idx], y_pred, labels=CLASS_ORDER), CLASS_ORDER
        )

        results[name] = {
            "accuracy": acc,
            "baseline": base,
            "y_true": y[test_idx],
            "y_pred": y_pred,
        }

    return results


def evaluate_grouped_cv(
    models: dict[str, Pipeline],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> dict[str, tuple[float, float]]:
    """Run GroupKFold cross-validation over cats.

    Args:
        models: Name -> unfitted pipeline.
        X: Feature matrix.
        y: Context labels.
        groups: Cat IDs.
        n_splits: Number of folds. Must not exceed the number of cats.

    Returns:
        Name -> ``(mean_accuracy, std_accuracy)`` across folds.
    """
    print("\n" + "=" * 72)
    print(f"GROUPED {n_splits}-FOLD CROSS-VALIDATION (every cat tested once)")
    print("=" * 72)

    cv = GroupKFold(n_splits=n_splits)
    summary = {}

    for name, model in models.items():
        fold_accs, fold_bases = [], []
        for train_idx, test_idx in cv.split(X, y, groups):
            model.fit(X[train_idx], y[train_idx])
            fold_accs.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))
            fold_bases.append(majority_baseline(y[train_idx], y[test_idx]))

        accs = np.array(fold_accs)
        summary[name] = (float(accs.mean()), float(accs.std()))
        print(
            f"{name:<24} acc = {accs.mean():.3f} +/- {accs.std():.3f}   "
            f"(baseline {np.mean(fold_bases):.3f})   "
            f"folds: {', '.join(f'{a:.2f}' for a in accs)}"
        )

    return summary


def save_confusion_plots(results: dict[str, dict], out_dir: Path) -> None:
    """Write a confusion-matrix PNG per model.

    Args:
        results: Output of :func:`evaluate_holdout`.
        out_dir: Directory to write into; created if absent.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    names = [CONTEXT_LABELS[c] for c in CLASS_ORDER]

    for name, res in results.items():
        cm = confusion_matrix(res["y_true"], res["y_pred"], labels=CLASS_ORDER)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

        fig, ax = plt.subplots(figsize=(5, 4.2))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{name}\naccuracy = {res['accuracy']:.2f} (unseen cats)")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}\n{cm_norm[i, j]:.0%}",
                    ha="center",
                    va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=9,
                )

        fig.colorbar(im, ax=ax, label="row-normalized")
        fig.tight_layout()

        slug = name.split(" ")[0].lower().replace("-", "_")
        path = out_dir / f"confusion_{slug}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  wrote {path}")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train MFCC baselines with cat-grouped validation.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--test-size", type=float, default=0.3, help="Fraction of cats held out."
    )
    parser.add_argument("--n-splits", type=int, default=5, help="GroupKFold folds.")
    parser.add_argument(
        "--save-plots", action="store_true", help="Write confusion matrices to artifacts/."
    )
    parser.add_argument(
        "--rms-norm",
        action="store_true",
        help="Normalize each waveform to a fixed RMS before extracting MFCCs, "
             "removing recording gain. Raises RandomForest from ~0.49 to ~0.57 "
             "and roughly halves fold variance -- see "
             "src/experiment_channel_norm.py. Off by default because it was "
             "selected on the same folds the README reports.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    try:
        recordings = scan_dataset(args.data_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not recordings:
        print(f"ERROR: no CatMeows WAV files found in {args.data_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(recordings)} recordings from "
          f"{len(set(r.cat_id for r in recordings))} cats in {args.data_dir}")
    print("Extracting MFCC features ...")
    if args.rms_norm:
        print("  (RMS gain normalization enabled)")
    X, y, groups = extract_feature_matrix(recordings, kind="mfcc", rms_norm=args.rms_norm)
    print(f"  X={X.shape}  y={y.shape}  cats={len(set(groups))}")

    counts = {CONTEXT_LABELS[c]: int((y == c).sum()) for c in CLASS_ORDER}
    print(f"  class counts: {counts}\n")

    models = build_models(args.seed)
    results = evaluate_holdout(models, X, y, groups, args.seed, args.test_size)
    evaluate_grouped_cv(build_models(args.seed), X, y, groups, args.n_splits)

    if args.save_plots:
        print("\nSaving plots ...")
        save_confusion_plots(results, REPO_ROOT / "artifacts")

    print("\n" + "=" * 72)
    print("SUMMARY (hold-out on unseen cats)")
    print("=" * 72)
    for name, res in results.items():
        print(f"{name:<24} accuracy = {res['accuracy']:.2f}   "
              f"baseline = {res['baseline']:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
