"""Does decoupling the recording channel from the context signal help?

**The finding this tests.** The energy coefficient ``c0`` is ~69% explained by
which cat produced the clip and only ~5.5% by the emission context. It looks
like pure nuisance -- but deleting it makes accuracy *worse* (-0.03 to -0.04).
Signal and nuisance share the coefficient, so the fix cannot be deletion. It has
to be normalization: strip the channel, keep the context.

**The variants.** Two are inductive (nothing about the test cats is used before
prediction) and two are transductive (they use the test cats' own clips, though
never their labels):

============  =============================================================
``baseline``  The current 52-d MFCC summary. No normalization.
``rms``       Waveform rescaled to fixed RMS before MFCCs. Removes recording
              *gain* -- mic distance and device volume. Inductive.
``cmn``       Per-clip cepstral mean normalization. A fixed channel is
              convolutional in time, hence additive in the cepstral domain,
              so subtracting the clip mean cancels it. Inductive.
``cmvn``      As ``cmn``, also dividing by each coefficient's std.
              Inductive.
``per_cat``   Subtract each cat's mean feature vector. The strongest form of
              channel removal, because a cat's channel is essentially fixed.
              **Transductive.**
``per_owner`` Subtract each owner's mean feature vector -- owner is a proxy
              for the recording setup itself. **Transductive.**
============  =============================================================

**On the transductive variants.** ``per_cat`` and ``per_owner`` use no labels,
but they compute statistics over clips that include the test set. That is a
weaker guarantee than the rest of this repository provides, and results from
them are *not* comparable to the README table. They are included because the
setting is realistic -- in deployment you would have several clips from a cat
before classifying a new one -- and because they upper-bound how much of the
gap is channel contamination at all. They are labelled everywhere they appear.

Usage::

    python src/experiment_channel_norm.py
    python src/experiment_channel_norm.py --embeddings ast
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import (  # noqa: E402
    load_wav,
    mfcc_feature_vector,
    rms_normalize,
    scan_dataset,
)
from train_baseline import build_models, majority_baseline  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_SEED = 42
N_SPLITS = 5

# Variants that peek at test-set clips (never labels). Reported separately.
TRANSDUCTIVE = {"per_cat", "per_owner"}


def set_seed(seed: int) -> None:
    """Seed the RNGs this script can reach."""
    random.seed(seed)
    np.random.seed(seed)


def group_mean_normalize(X: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Subtract the mean feature vector of each group.

    This is the transductive step: the mean for a test cat is computed from that
    cat's own clips. No labels are involved.

    Args:
        X: Feature matrix, shape ``(n_samples, n_features)``.
        keys: Group key per row (cat ID or owner ID).

    Returns:
        The group-centred feature matrix.
    """
    out = X.astype(np.float64, copy=True)
    for key in np.unique(keys):
        mask = keys == key
        out[mask] -= out[mask].mean(axis=0, keepdims=True)
    return out.astype(np.float32)


def build_mfcc_variants(data_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Extract every MFCC normalization variant.

    Args:
        data_dir: Directory of CatMeows WAV files.

    Returns:
        ``(variants, y, cats, owners)`` where ``variants`` maps a variant name
        to its feature matrix.
    """
    recordings = scan_dataset(data_dir)
    y = np.array([r.context for r in recordings])
    cats = np.array([r.cat_id for r in recordings])
    owners = np.array([r.owner_id for r in recordings])

    plain, rms, cmn, cmvn = [], [], [], []
    for rec in recordings:
        signal, sr = load_wav(rec.path)
        plain.append(mfcc_feature_vector(signal, sample_rate=sr))
        rms.append(mfcc_feature_vector(rms_normalize(signal), sample_rate=sr))
        cmn.append(mfcc_feature_vector(signal, sample_rate=sr, cmvn="mean"))
        cmvn.append(mfcc_feature_vector(signal, sample_rate=sr, cmvn="meanvar"))

    plain = np.stack(plain)
    variants = {
        "baseline": plain,
        "rms": np.stack(rms),
        "cmn": np.stack(cmn),
        "cmvn": np.stack(cmvn),
        "per_cat": group_mean_normalize(plain, cats),
        "per_owner": group_mean_normalize(plain, owners),
    }
    return variants, y, cats, owners


def score_variant(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, factory, seed: int
) -> dict[str, dict]:
    """Score every model on one feature variant with GroupKFold over cats.

    Args:
        X: Feature matrix for this variant.
        y: Context labels.
        groups: Cat IDs (the grouping for cross-validation).
        factory: Callable taking a seed and returning ``{name: estimator}``.
            Called per fold so each fold trains a fresh estimator.
        seed: Seed passed to ``factory``.

    Returns:
        Model name -> dict with ``acc`` (mean), ``std``, ``iso_f1``, ``food_f1``.
    """
    out = {}
    for name in factory(seed):
        accs, yt, yp = [], [], []
        for train_idx, test_idx in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
            assert not set(groups[train_idx]) & set(groups[test_idx]), "cat leaked"
            model = factory(seed)[name]
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            accs.append(accuracy_score(y[test_idx], pred))
            yt.extend(y[test_idx])
            yp.extend(pred)

        accs = np.array(accs)
        yt, yp = np.array(yt), np.array(yp)
        out[name] = {
            "acc": float(accs.mean()),
            "std": float(accs.std()),
            "iso_f1": float(f1_score(yt, yp, labels=["I"], average="macro", zero_division=0)),
            "food_f1": float(f1_score(yt, yp, labels=["F"], average="macro", zero_division=0)),
        }
    return out


def run_mfcc_experiment(data_dir: Path, seed: int) -> None:
    """Run and report the MFCC normalization comparison."""
    print("Extracting MFCC variants ...")
    variants, y, cats, owners = build_mfcc_variants(data_dir)
    for name, X in variants.items():
        print(f"  {name:<10} {X.shape}")

    baseline_acc = np.mean(
        [
            majority_baseline(y[tr], y[te])
            for tr, te in GroupKFold(n_splits=N_SPLITS).split(variants["baseline"], y, cats)
        ]
    )

    model_names = list(build_models(seed))
    results = {v: score_variant(X, y, cats, build_models, seed)
               for v, X in variants.items()}

    print("\n" + "=" * 78)
    print(f"CHANNEL NORMALIZATION -- {N_SPLITS}-fold GroupKFold by cat "
          f"(majority baseline {baseline_acc:.3f})")
    print("=" * 78)

    for model in model_names:
        base = results["baseline"][model]["acc"]
        print(f"\n{model}")
        print(f"  {'variant':<12} {'acc':>14}  {'delta':>7}  {'isoF1':>6} {'foodF1':>7}")
        print("  " + "-" * 54)
        for variant in variants:
            r = results[variant][model]
            tag = "  [transductive]" if variant in TRANSDUCTIVE else ""
            delta = "" if variant == "baseline" else f"{r['acc'] - base:+.3f}"
            print(f"  {variant:<12} {r['acc']:.3f} +/- {r['std']:.3f}  {delta:>7}  "
                  f"{r['iso_f1']:.2f}   {r['food_f1']:.2f}{tag}")

    print("\n" + "-" * 78)
    print("Inductive variants are comparable to the README table; transductive")
    print("ones use the test cats' own clips (never labels) and are not.")


def run_embedding_experiment(backbone: str, seed: int) -> None:
    """Check whether group normalization also helps pretrained embeddings.

    If channel contamination is a general problem rather than an MFCC artifact,
    the same normalization should move the probe numbers too.

    Args:
        backbone: Backbone short name passed through to the embedding cache.
        seed: Random seed.
    """
    from embeddings import extract_embeddings
    from train_transfer import build_probes

    try:
        X, y, cats = extract_embeddings(model_name=backbone, verbose=False)
    except (FileNotFoundError, ImportError) as exc:
        print(f"\nSkipping embedding check: {exc}", file=sys.stderr)
        return

    recordings = scan_dataset(DEFAULT_DATA_DIR)
    owners = np.array([r.owner_id for r in recordings])

    variants = {
        "baseline": X,
        "per_cat": group_mean_normalize(X, cats),
        "per_owner": group_mean_normalize(X, owners),
    }

    print("\n" + "=" * 78)
    print(f"SAME TEST ON FROZEN EMBEDDINGS -- backbone: {backbone}")
    print("=" * 78)

    probe_names = list(build_probes(seed))
    for model in probe_names:
        print(f"\n{model}")
        print(f"  {'variant':<12} {'acc':>14}  {'delta':>7}")
        print("  " + "-" * 40)
        base = None
        for variant, Xv in variants.items():
            accs = []
            for tr, te in GroupKFold(n_splits=N_SPLITS).split(Xv, y, cats):
                probe = build_probes(seed)[model]
                probe.fit(Xv[tr], y[tr])
                accs.append(accuracy_score(y[te], probe.predict(Xv[te])))
            accs = np.array(accs)
            if base is None:
                base = accs.mean()
                delta = ""
            else:
                delta = f"{accs.mean() - base:+.3f}"
            tag = "  [transductive]" if variant in TRANSDUCTIVE else ""
            print(f"  {variant:<12} {accs.mean():.3f} +/- {accs.std():.3f}  {delta:>7}{tag}")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Channel-normalization experiment (the c0 finding).",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        metavar="BACKBONE",
        help="Also run the group-normalization check on cached embeddings "
             "(e.g. --embeddings ast).",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    try:
        run_mfcc_experiment(args.data_dir, args.seed)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.embeddings:
        run_embedding_experiment(args.embeddings, args.seed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
