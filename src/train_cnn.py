"""SmallCNN on log-mel spectrograms, trained from scratch.

This script exists to answer one question honestly: **is a convolutional network
trained from scratch worth it on 440 clips?** It is included as a documented
negative result, not as the recommended model. See ``docs/ROADMAP.md`` for the
transfer-learning path that should replace it.

Design notes:

* The architecture is deliberately small (3 conv blocks, ~25k parameters).
  Anything larger overfits 300-odd training clips almost immediately.
* Validation is grouped by ``cat_id``, exactly as in
  :mod:`train_baseline` -- the model is only scored on cats it has never heard.
  Without this, accuracy is inflated by voice recognition rather than context
  recognition.
* Class weights compensate for the 221/127/92 context imbalance.
* Per-band normalization statistics are computed on the *training* split only
  and then applied to the test split, so no test statistics leak.

Usage::

    python src/train_cnn.py                  # single hold-out of unseen cats
    python src/train_cnn.py --cv 5           # GroupKFold, the number in the README
    python src/train_cnn.py --epochs 60 --seed 42
    python src/train_cnn.py --save-model artifacts/smallcnn.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import (  # noqa: E402
    CNN_N_FRAMES,
    CONTEXT_LABELS,
    N_MELS,
    extract_feature_matrix,
    scan_dataset,
)
from train_baseline import majority_baseline, print_confusion  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_SEED = 42
CLASS_ORDER = ["B", "F", "I"]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch, and force deterministic cuDNN kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SmallCNN(nn.Module):
    """A compact 3-block CNN for log-mel spectrogram classification.

    Each block is ``Conv2d(3x3) -> BatchNorm -> ReLU -> MaxPool(2)``, widening
    16 -> 32 -> 64 channels. An :class:`~torch.nn.AdaptiveAvgPool2d` then
    collapses whatever spatial extent remains to 1x1, which makes the model
    agnostic to the exact input size, followed by dropout and a linear head.

    Global average pooling is used instead of a flattened dense layer
    specifically to keep the parameter count low -- with only ~300 training
    clips, a wide dense layer is the fastest route to memorizing the training
    set.

    Args:
        n_classes: Number of output classes.
        dropout: Dropout probability before the classifier head.
    """

    def __init__(self, n_classes: int = 3, dropout: float = 0.4) -> None:
        super().__init__()

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, 1, n_mels, n_frames)`` spectrograms to class logits."""
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(self.dropout(x))


def normalize_splits(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize per mel band using training-split statistics only.

    Args:
        X_train: Training spectrograms, shape ``(n, n_mels, n_frames)``.
        X_test: Test spectrograms, same trailing shape.

    Returns:
        The normalized ``(X_train, X_test)``.
    """
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Wrap the numpy splits in deterministic Torch dataloaders.

    Args:
        X_train: Normalized training spectrograms.
        y_train: Integer training labels.
        X_test: Normalized test spectrograms.
        y_test: Integer test labels.
        batch_size: Minibatch size.
        seed: Seed for the shuffling generator.

    Returns:
        ``(train_loader, test_loader)``.
    """

    def to_tensor(X: np.ndarray, y: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(X).float().unsqueeze(1),  # (N, 1, n_mels, n_frames)
            torch.from_numpy(y).long(),
        )

    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(
            to_tensor(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        ),
        DataLoader(to_tensor(X_test, y_test), batch_size=batch_size, shuffle=False),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run a single training epoch.

    Returns:
        Mean training loss over the epoch.
    """
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Predict over a loader.

    Returns:
        ``(y_true, y_pred)`` as integer arrays.
    """
    model.eval()
    trues, preds = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(yb.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def run_fold(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, nn.Module]:
    """Train SmallCNN on one train/test split and predict the test split.

    Args:
        X: All spectrograms, shape ``(n, n_mels, n_frames)``.
        y: All integer labels.
        train_idx: Indices of the training clips.
        test_idx: Indices of the test clips.
        args: Parsed CLI arguments (epochs, lr, batch size, ...).
        device: Torch device to train on.
        verbose: Whether to print per-epoch progress.

    Returns:
        ``(y_true, y_pred, model)`` for the test split.
    """
    X_train, X_test = normalize_splits(X[train_idx], X[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]

    train_loader, test_loader = make_loaders(
        X_train, y_train, X_test, y_test, args.batch_size, args.seed
    )

    model = SmallCNN(n_classes=len(CLASS_ORDER), dropout=args.dropout).to(device)

    # Inverse-frequency class weights, computed on the training split only.
    counts = np.bincount(y_train, minlength=len(CLASS_ORDER)).astype(np.float64)
    weights = counts.sum() / (len(CLASS_ORDER) * np.maximum(counts, 1))

    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"SmallCNN: {n_params:,} trainable parameters")
        pretty = {CONTEXT_LABELS[c]: round(float(w), 2) for c, w in zip(CLASS_ORDER, weights)}
        print(f"class weights: {pretty}\n")

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        if verbose and (epoch % 10 == 0 or epoch == 1):
            _, y_pred = evaluate(model, test_loader, device)
            acc = accuracy_score(y_test, y_pred)
            print(f"  epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}  test_acc={acc:.3f}")

    y_true, y_pred = evaluate(model, test_loader, device)
    return y_true, y_pred, model


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train SmallCNN on log-mel spectrograms (validation grouped by cat).",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument(
        "--cv",
        type=int,
        default=0,
        metavar="N",
        help="Run GroupKFold with N folds instead of a single hold-out. "
             "The README table is produced with --cv 5.",
    )
    parser.add_argument("--save-model", type=Path, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        recordings = scan_dataset(args.data_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not recordings:
        print(f"ERROR: no CatMeows WAV files found in {args.data_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(recordings)} recordings from "
          f"{len(set(r.cat_id for r in recordings))} cats")
    print(f"Extracting log-mel spectrograms ({N_MELS} x {CNN_N_FRAMES}) ...")
    X, y_str, groups = extract_feature_matrix(recordings, kind="logmel")

    label_to_int = {c: i for i, c in enumerate(CLASS_ORDER)}
    y = np.array([label_to_int[c] for c in y_str])
    print(f"  X={X.shape}  device={device}")

    if args.cv:
        # GroupKFold: every cat is tested exactly once. This is the protocol
        # behind the README table, and its baseline reflects the true class
        # balance rather than whichever cats a single split happened to pick.
        print(f"\nGroupKFold({args.cv}) -- every cat tested exactly once\n")
        cv = GroupKFold(n_splits=args.cv)
        fold_accs, fold_bases, all_true, all_pred = [], [], [], []

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
            assert not set(groups[train_idx]) & set(groups[test_idx]), "cat leaked"
            set_seed(args.seed)  # each fold starts from the same initialization
            y_true, y_pred, _ = run_fold(X, y, train_idx, test_idx, args, device, verbose=False)

            fold_acc = accuracy_score(y_true, y_pred)
            fold_base = majority_baseline(y_str[train_idx], y_str[test_idx])
            fold_accs.append(fold_acc)
            fold_bases.append(fold_base)
            all_true.append(y_true)
            all_pred.append(y_pred)
            print(f"  fold {fold}/{args.cv}: {len(test_idx):3d} clips from "
                  f"{len(set(groups[test_idx])):2d} cats  acc={fold_acc:.3f}  "
                  f"baseline={fold_base:.3f}")

        accs = np.array(fold_accs)
        base = float(np.mean(fold_bases))
        acc = float(accs.mean())
        y_true, y_pred = np.concatenate(all_true), np.concatenate(all_pred)

        print("\n" + "=" * 72)
        print(f"SmallCNN (mel-spec, from scratch)   accuracy = {acc:.2f} "
              f"+/- {accs.std():.2f}   (baseline {base:.2f})")
        print("=" * 72)
        print("Pooled over all folds:\n")
    else:
        # Single hold-out of unseen cats -- faster, but noisier.
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=args.test_size, random_state=args.seed
        )
        train_idx, test_idx = next(splitter.split(X, y, groups))
        assert not set(groups[train_idx]) & set(groups[test_idx]), "cat leaked across split"

        print(f"train: {len(train_idx):3d} clips from {len(set(groups[train_idx])):2d} cats")
        print(f"test : {len(test_idx):3d} clips from {len(set(groups[test_idx])):2d} cats")

        base = majority_baseline(y_str[train_idx], y_str[test_idx])
        print(f"majority-class baseline on this test split: {base:.2f}\n")

        y_true, y_pred, model = run_fold(X, y, train_idx, test_idx, args, device)
        acc = accuracy_score(y_true, y_pred)

        print("\n" + "=" * 72)
        print(f"SmallCNN (mel-spec, from scratch)   accuracy = {acc:.2f}   "
              f"(baseline {base:.2f})")
        print("=" * 72)
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_ORDER))),
            target_names=[CONTEXT_LABELS[c] for c in CLASS_ORDER],
            zero_division=0,
        )
    )
    print("confusion matrix (rows = true, cols = predicted):")
    print_confusion(
        confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_ORDER)))),
        CLASS_ORDER,
    )

    if acc < base:
        print(
            "\nNOTE: the CNN does not beat the majority-class baseline. This is the "
            "expected outcome at this dataset size and is reported as-is; see "
            "docs/ROADMAP.md for the transfer-learning path."
        )

    if args.save_model and args.cv:
        print(
            "\nNOTE: --save-model is ignored with --cv (each fold trains its own "
            "model). Re-run without --cv to save a single hold-out model.",
            file=sys.stderr,
        )
    elif args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "classes": CLASS_ORDER,
                "n_mels": N_MELS,
                "n_frames": CNN_N_FRAMES,
                "seed": args.seed,
            },
            args.save_model,
        )
        print(f"\nSaved model to {args.save_model}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
