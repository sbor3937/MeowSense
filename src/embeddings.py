"""Frozen embeddings from an AudioSet-pretrained model.

The from-scratch CNN in :mod:`train_cnn` only manages to tie a one-line MFCC
SVM, which is the expected ceiling for 23k parameters trained on ~300 clips.
The way past it is a better *prior*, not a bigger network: a model pretrained on
AudioSet (~2M clips) already knows what animal vocalizations sound like, so we
freeze it and learn only the last step.

This module extracts embeddings once and caches them to disk, because a forward
pass over the whole dataset costs minutes on CPU and the probe is re-fit many
times during cross-validation.

**The 8 kHz caveat, stated up front.** Every AudioSet model expects 16 kHz.
CatMeows was recorded at 8 kHz, so its Nyquist limit is 4 kHz: upsampling to
16 kHz restores the *sample rate* but not the missing 4-8 kHz band, which is
exactly where a lot of feline vocalization energy sits. The pretrained model
therefore sees a spectrogram whose top half is empty -- nothing like its
training distribution. Expect a real penalty from this, and do **not** compare
these numbers against literature trained on native 16 kHz audio. It is also an
argument for recording any crowdsourced dataset at 16 kHz or higher from day
one (see ``docs/ROADMAP.md``).

Two independent backbones are supported (see :data:`BACKBONES`), because a
single pretrained model scoring well is a result about that model, while two
unrelated ones agreeing is a result about the task:

* ``ast``  -- AST fine-tuned on AudioSet, 16 kHz, ViT-style patches.
* ``clap`` -- CLAP HTSAT audio encoder, 48 kHz, audio-text contrastive.

Usage::

    python src/embeddings.py                 # AST (default), extract + cache
    python src/embeddings.py --model clap    # the second backbone
    python src/embeddings.py --force         # ignore an existing cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import SAMPLE_RATE, load_wav, scan_dataset  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "embeddings"

# Supported backbones. Both are PyTorch-native (torch is already a dependency
# for train_cnn) and need no TensorFlow, which is why YAMNet -- a TF Hub model --
# is still untried; see docs/ROADMAP.md.
#
# Two *independent* backbones are wired up on purpose. A single pretrained model
# scoring well is a result about that model; two unrelated ones agreeing is a
# result about the task. AST and CLAP differ in architecture (ViT-style patches
# vs HTSAT), in training objective (AudioSet classification vs audio-text
# contrastive) and in expected sample rate, so they are a genuine replication
# rather than a near-duplicate.
BACKBONES = {
    "ast": {
        "model_id": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "sample_rate": 16000,
        "kind": "ast",
        "note": "AST, fine-tuned on AudioSet (~86M params)",
    },
    "clap": {
        "model_id": "laion/clap-htsat-unfused",
        "sample_rate": 48000,
        "kind": "clap",
        "note": "CLAP HTSAT audio encoder, audio-text contrastive (LAION-Audio-630k)",
    },
}

DEFAULT_BACKBONE = "ast"
DEFAULT_MODEL = BACKBONES[DEFAULT_BACKBONE]["model_id"]
MODEL_SAMPLE_RATE = BACKBONES[DEFAULT_BACKBONE]["sample_rate"]


def resolve_backbone(name_or_id: str) -> dict:
    """Resolve a short backbone name (or a raw HuggingFace id) to its config.

    Args:
        name_or_id: A key of :data:`BACKBONES` (e.g. ``"ast"``) or a full model
            id, which is assumed to be AST-like.

    Returns:
        The backbone config dict.
    """
    if name_or_id in BACKBONES:
        return BACKBONES[name_or_id]
    for cfg in BACKBONES.values():
        if cfg["model_id"] == name_or_id:
            return cfg
    return {
        "model_id": name_or_id,
        "sample_rate": 16000,
        "kind": "ast",
        "note": "custom model id (assumed AST-like)",
    }


def _cache_path(cache_dir: Path, model_name: str) -> Path:
    """Cache file for one model, named after a filesystem-safe model slug."""
    slug = model_name.replace("/", "__").replace(".", "_")
    return Path(cache_dir) / f"{slug}.npz"


def _load_backbone(config: dict):
    """Load a pretrained processor and backbone, in eval mode.

    Args:
        config: A backbone config from :func:`resolve_backbone`.

    Returns:
        ``(processor, model, torch)`` -- the torch module is returned so callers
        do not need their own import just to build a no_grad context.

    Raises:
        ImportError: If ``transformers`` is not installed.
    """
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoProcessor, ASTModel, ClapModel
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Transfer-learning embeddings need `transformers` and `torch`:\n"
            "    pip install transformers torch\n"
            f"(original error: {exc})"
        ) from exc

    model_id = config["model_id"]
    if config["kind"] == "clap":
        processor = AutoProcessor.from_pretrained(model_id)
        model = ClapModel.from_pretrained(model_id)
    else:
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        model = ASTModel.from_pretrained(model_id)

    model.eval()
    return processor, model, torch


def _embed_batch(processor, model, torch, config: dict, waves: list) -> np.ndarray:
    """Embed one batch of waveforms with the given backbone.

    AST exposes token-level hidden states, which we mean-pool over time. CLAP
    exposes a purpose-built pooled audio embedding, which we use directly rather
    than re-pooling something it already pooled.

    Args:
        processor: The model's feature extractor / processor.
        model: The loaded backbone.
        torch: The torch module.
        config: Backbone config.
        waves: List of 1-D float waveforms at ``config["sample_rate"]``.

    Returns:
        Array of shape ``(len(waves), embedding_dim)``.
    """
    sr = config["sample_rate"]

    if config["kind"] == "clap":
        # transformers 5.x renamed this argument from `audios` to `audio`.
        # Support both so the declared >=4.40 floor is honest.
        try:
            inputs = processor(audio=waves, sampling_rate=sr, return_tensors="pt")
        except (TypeError, ValueError):
            inputs = processor(audios=waves, sampling_rate=sr, return_tensors="pt")

        with torch.no_grad():
            out = model.get_audio_features(**inputs)

        # Older versions return the projected tensor directly; 5.x returns a
        # ModelOutput whose pooler_output is that same 512-d CLAP audio
        # embedding (the audio-text shared space). We use the projected vector
        # rather than the 768-d pre-projection encoder state: it is the
        # canonical "CLAP embedding", and fixing the choice up front avoids
        # picking whichever variant happens to score better on the test folds.
        pooled = getattr(out, "pooler_output", out)
        return pooled.cpu().numpy()

    inputs = processor(waves, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**inputs).last_hidden_state  # (B, tokens, dim)
    return hidden.mean(dim=1).cpu().numpy()  # mean-pool over tokens


def extract_embeddings(
    data_dir: Path = DEFAULT_DATA_DIR,
    model_name: str = DEFAULT_MODEL,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force: bool = False,
    batch_size: int = 8,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (or load cached) frozen embeddings for the whole dataset.

    Clips are resampled 8 kHz -> 16 kHz, passed through the frozen backbone, and
    summarized by mean-pooling the final hidden states over time.

    Args:
        data_dir: Directory of CatMeows WAV files.
        model_name: Short backbone name (``"ast"``, ``"clap"``) or a raw
            HuggingFace model id. See :data:`BACKBONES`.
        cache_dir: Where to store/read the ``.npz`` cache.
        force: Re-extract even if a cache exists.
        batch_size: Clips per forward pass.
        verbose: Print progress.

    Returns:
        ``(X, y, groups)`` -- embeddings, context letters, and cat IDs, in the
        deterministic order returned by :func:`features.scan_dataset`.

    Raises:
        FileNotFoundError: If ``data_dir`` holds no recordings.
    """
    config = resolve_backbone(model_name)
    cache_file = _cache_path(cache_dir, config["model_id"])

    if cache_file.exists() and not force:
        cached = np.load(cache_file, allow_pickle=False)
        if verbose:
            print(f"Loaded cached embeddings from {cache_file}  X={cached['X'].shape}")
        return cached["X"], cached["y"].astype(str), cached["groups"].astype(str)

    recordings = scan_dataset(data_dir)
    if not recordings:
        raise FileNotFoundError(
            f"No CatMeows WAV files in {data_dir}. Run `python src/download_data.py`."
        )

    processor, model, torch = _load_backbone(config)
    target_sr = config["sample_rate"]

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Backbone: {config['model_id']}  ({n_params / 1e6:.1f}M params, frozen)")
        print(f"  {config['note']}")
        print(f"Resampling {SAMPLE_RATE} Hz -> {target_sr} Hz "
              "(does NOT restore the missing 4-8 kHz band)")
        print(f"Extracting embeddings for {len(recordings)} clips ...")

    vectors: list[np.ndarray] = []
    for start in range(0, len(recordings), batch_size):
        batch = recordings[start : start + batch_size]
        waves = [load_wav(r.path, target_sr=target_sr)[0] for r in batch]
        vectors.append(_embed_batch(processor, model, torch, config, waves))

        if verbose:
            done = min(start + batch_size, len(recordings))
            print(f"\r  {done}/{len(recordings)} clips", end="", flush=True)

    if verbose:
        print()

    X = np.concatenate(vectors).astype(np.float32)
    y = np.array([r.context for r in recordings])
    groups = np.array([r.cat_id for r in recordings])

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, X=X, y=y, groups=groups)
    if verbose:
        print(f"Cached embeddings to {cache_file}  X={X.shape}")

    return X, y, groups


def main() -> int:
    """CLI entry point: extract and cache embeddings."""
    parser = argparse.ArgumentParser(
        description="Extract frozen AudioSet-pretrained embeddings for CatMeows.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_BACKBONE,
        help=f"Backbone: {' | '.join(BACKBONES)}, or a raw HuggingFace model id.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Ignore any cache.")
    args = parser.parse_args()

    try:
        X, y, groups = extract_embeddings(
            data_dir=args.data_dir,
            model_name=args.model,
            cache_dir=args.cache_dir,
            force=args.force,
            batch_size=args.batch_size,
        )
    except (FileNotFoundError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nDone. X={X.shape}  cats={len(set(groups))}")
    print("Next: python src/train_transfer.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
