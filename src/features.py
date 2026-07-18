"""Signal processing and metadata utilities for the CatMeows dataset.

This module deliberately depends only on ``numpy`` and ``scipy`` -- there is no
``librosa`` dependency -- so that the feature pipeline is small, auditable and
easy to port. Everything needed to go from a raw ``.wav`` file to either an
MFCC summary vector (for the classical baselines) or a fixed-size log-mel
spectrogram patch (for the CNN) lives here.

Two things matter for reproducibility and are handled explicitly:

1. **Fixed DSP settings.** The CatMeows recordings are 8 kHz mono. The defaults
   below (``n_fft=256``, 40 mel filters spanning 50-4000 Hz) are chosen for that
   sample rate and are exposed as module-level constants rather than scattered
   magic numbers.
2. **Cat identity.** Every filename encodes the cat that produced the
   vocalization. :func:`parse_filename` recovers it so that downstream code can
   group by cat and avoid train/test leakage across a single animal.

Filename convention (documented by the dataset authors)::

    C_NNNNN_BB_SS_OOOOO_RXX.wav

    C     emission context -- B = brushing, F = waiting for food,
                              I = isolation in an unfamiliar environment
    NNNNN cat unique ID
    BB    breed -- MC = Maine Coon, EU = European Shorthair
    SS    sex   -- FI/FN = female intact/neutered, MI/MN = male intact/neutered
    OOOOO owner unique ID
    R     recording session (1..3)
    XX    vocalization counter within the session

Example: ``F_BAC01_MC_MN_SIM01_101.wav``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import resample_poly

__all__ = [
    "SAMPLE_RATE",
    "N_FFT",
    "HOP_LENGTH",
    "N_MELS",
    "FMIN",
    "FMAX",
    "N_MFCC",
    "CNN_N_FRAMES",
    "CONTEXT_LABELS",
    "BREED_LABELS",
    "SEX_LABELS",
    "Recording",
    "parse_filename",
    "load_wav",
    "mel_filterbank",
    "log_mel_spectrogram",
    "mfcc",
    "mfcc_feature_vector",
    "fixed_size_log_mel",
    "scan_dataset",
    "build_dataset_frame",
]

# --------------------------------------------------------------------------
# DSP configuration. CatMeows is distributed at 8 kHz, so the Nyquist limit is
# 4000 Hz; FMAX sits exactly there and FMIN=50 discards DC / room rumble.
# --------------------------------------------------------------------------

SAMPLE_RATE = 8000
N_FFT = 256          # 32 ms analysis window at 8 kHz
HOP_LENGTH = 128     # 16 ms hop -> 50% overlap
N_MELS = 40
FMIN = 50.0
FMAX = 4000.0
N_MFCC = 13
CNN_N_FRAMES = 128   # ~2.06 s at the hop above; covers the median call length

CONTEXT_LABELS = {
    "B": "brushing",
    "F": "food",
    "I": "isolation",
}

BREED_LABELS = {
    "MC": "Maine Coon",
    "EU": "European Shorthair",
}

SEX_LABELS = {
    "FI": "female intact",
    "FN": "female neutered",
    "MI": "male intact",
    "MN": "male neutered",
}

_FILENAME_RE = re.compile(
    r"^(?P<context>[BFI])"
    r"_(?P<cat_id>[A-Za-z0-9]+)"
    r"_(?P<breed>MC|EU)"
    r"_(?P<sex>FI|FN|MI|MN)"
    r"_(?P<owner_id>[A-Za-z0-9]+)"
    r"_(?P<session>\d)(?P<counter>\d{2})$"
)


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Recording:
    """Metadata for one vocalization, recovered from its filename."""

    path: Path
    context: str            # "B" | "F" | "I"
    context_label: str      # "brushing" | "food" | "isolation"
    cat_id: str
    breed: str              # "MC" | "EU"
    breed_label: str
    sex: str                # "FI" | "FN" | "MI" | "MN"
    sex_label: str
    is_female: bool
    is_neutered: bool
    owner_id: str
    session: int
    counter: int

    def as_dict(self) -> dict:
        """Return a plain dict, with ``path`` stringified (pandas-friendly)."""
        d = asdict(self)
        d["path"] = str(self.path)
        return d


def parse_filename(path: str | Path) -> Recording:
    """Parse CatMeows metadata out of a filename.

    Args:
        path: Path to a ``.wav`` file. Only the stem is inspected, so the file
            does not need to exist.

    Returns:
        A :class:`Recording` with the decoded fields.

    Raises:
        ValueError: If the stem does not follow the documented convention.

    Example:
        >>> rec = parse_filename("F_BAC01_MC_MN_SIM01_101.wav")
        >>> rec.context_label, rec.cat_id, rec.session
        ('food', 'BAC01', 1)
    """
    path = Path(path)
    match = _FILENAME_RE.match(path.stem)
    if match is None:
        raise ValueError(
            f"Filename {path.name!r} does not match the CatMeows convention "
            "'C_NNNNN_BB_SS_OOOOO_RXX.wav'"
        )

    g = match.groupdict()
    sex = g["sex"]
    return Recording(
        path=path,
        context=g["context"],
        context_label=CONTEXT_LABELS[g["context"]],
        cat_id=g["cat_id"],
        breed=g["breed"],
        breed_label=BREED_LABELS[g["breed"]],
        sex=sex,
        sex_label=SEX_LABELS[sex],
        is_female=sex.startswith("F"),
        is_neutered=sex.endswith("N"),
        owner_id=g["owner_id"],
        session=int(g["session"]),
        counter=int(g["counter"]),
    )


def scan_dataset(data_dir: str | Path) -> list[Recording]:
    """Recursively collect and parse every ``.wav`` under ``data_dir``.

    Files whose names do not follow the CatMeows convention are skipped rather
    than raising, so that stray files (e.g. an unzipped ``README.txt`` renamed
    by hand) do not break a whole run.

    Args:
        data_dir: Directory to walk.

    Returns:
        Recordings sorted by filename, for a deterministic ordering.

    Raises:
        FileNotFoundError: If ``data_dir`` does not exist.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"{data_dir} not found. Run `python src/download_data.py` first."
        )

    recordings: list[Recording] = []
    for wav_path in sorted(data_dir.rglob("*.wav")):
        try:
            recordings.append(parse_filename(wav_path))
        except ValueError:
            continue
    return recordings


def build_dataset_frame(data_dir: str | Path):
    """Build a :class:`pandas.DataFrame` of per-recording metadata + duration.

    Reading the duration requires opening each file, which is why this is kept
    separate from :func:`scan_dataset`.

    Args:
        data_dir: Directory containing the ``.wav`` files.

    Returns:
        One row per recording, with all :class:`Recording` fields plus a
        ``duration_s`` column.
    """
    import pandas as pd  # local import: keeps the DSP path pandas-free

    rows = []
    for rec in scan_dataset(data_dir):
        row = rec.as_dict()
        sr, data = wavfile.read(rec.path)
        row["duration_s"] = len(data) / float(sr)
        row["source_sr"] = sr
        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Audio I/O
# --------------------------------------------------------------------------


def load_wav(path: str | Path, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load a WAV file as mono float32 in [-1, 1], resampling if needed.

    Args:
        path: Path to the ``.wav`` file.
        target_sr: Desired sample rate. CatMeows is natively 8 kHz, so for this
            dataset no resampling normally occurs.

    Returns:
        ``(samples, sample_rate)`` where ``samples`` is 1-D float32.
    """
    sr, data = wavfile.read(str(path))

    # Integer PCM -> float in [-1, 1]. float files are already in range.
    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)

    if data.ndim > 1:  # downmix any stereo stragglers
        data = data.mean(axis=1)

    if sr != target_sr:
        data = resample_poly(data, target_sr, sr).astype(np.float32)
        sr = target_sr

    return np.ascontiguousarray(data, dtype=np.float32), sr


# --------------------------------------------------------------------------
# Spectral features
# --------------------------------------------------------------------------


def _hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    """Convert Hz to mels (O'Shaughnessy / HTK formula)."""
    return 2595.0 * np.log10(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    """Inverse of :func:`_hz_to_mel`."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


@lru_cache(maxsize=32)
def mel_filterbank(
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """Build a triangular mel filterbank matrix.

    Filters are area-normalized (Slaney style) so that wide high-frequency
    filters do not dominate narrow low-frequency ones.

    The result depends only on the (hashable) arguments, so it is cached: a full
    dataset pass calls this once per unique setting instead of once per clip,
    which removed roughly half of feature-extraction time. Because the cached
    array is shared across all callers, it is returned read-only; callers that
    need to mutate a filterbank should copy it first.

    Args:
        sample_rate: Sample rate of the signal.
        n_fft: FFT size used for the spectrogram.
        n_mels: Number of mel bands.
        fmin: Lowest band edge, in Hz.
        fmax: Highest band edge, in Hz. Must not exceed Nyquist.

    Returns:
        Read-only array of shape ``(n_mels, n_fft // 2 + 1)``.

    Raises:
        ValueError: If the band edges are not ``0 <= fmin < fmax <= Nyquist``.
    """
    nyquist = sample_rate / 2.0
    if not 0 <= fmin < fmax:
        raise ValueError(f"Require 0 <= fmin < fmax, got fmin={fmin}, fmax={fmax}")
    if fmax > nyquist:
        raise ValueError(f"fmax={fmax} exceeds Nyquist={nyquist} for sr={sample_rate}")

    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, nyquist, n_bins)

    # n_mels + 2 edges -> n_mels overlapping triangles.
    mel_edges = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_edges = _mel_to_hz(mel_edges)

    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = hz_edges[i], hz_edges[i + 1], hz_edges[i + 2]
        rising = (fft_freqs - left) / (center - left)
        falling = (right - fft_freqs) / (right - center)
        fb[i] = np.maximum(0.0, np.minimum(rising, falling))

    # Slaney normalization: equal area per filter.
    enorm = 2.0 / (hz_edges[2 : n_mels + 2] - hz_edges[:n_mels])
    fb *= enorm[:, None]

    # Cached and shared -> freeze so an accidental in-place write can't corrupt
    # every future caller's filterbank.
    fb.flags.writeable = False
    return fb


def _frame(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Slice ``y`` into overlapping frames of shape ``(n_frames, frame_length)``.

    Signals shorter than one frame are zero-padded up to a single frame, so the
    function always returns at least one frame.
    """
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))

    n_frames = 1 + (len(y) - frame_length) // hop_length
    idx = np.arange(frame_length)[None, :] + hop_length * np.arange(n_frames)[:, None]
    return y[idx]


def power_spectrogram(
    y: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Magnitude-squared STFT with a periodic Hann window.

    Args:
        y: 1-D signal.
        n_fft: FFT / frame size.
        hop_length: Hop between consecutive frames.

    Returns:
        Array of shape ``(n_fft // 2 + 1, n_frames)``.
    """
    frames = _frame(np.asarray(y, dtype=np.float64), n_fft, hop_length)
    window = np.hanning(n_fft + 1)[:-1]  # periodic, matches STFT convention
    spec = np.fft.rfft(frames * window[None, :], n=n_fft, axis=1)
    return (np.abs(spec) ** 2).T


def log_mel_spectrogram(
    y: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
    top_db: float | None = 80.0,
) -> np.ndarray:
    """Compute a log-mel spectrogram in decibels.

    Args:
        y: 1-D signal.
        sample_rate: Sample rate of ``y``.
        n_fft: FFT / frame size.
        hop_length: Hop between frames.
        n_mels: Number of mel bands.
        fmin: Lowest mel band edge, in Hz.
        fmax: Highest mel band edge, in Hz.
        top_db: If set, clamp the output floor to ``max - top_db`` dB. This
            stops near-silent frames from producing huge negative outliers that
            would dominate feature scaling.

    Returns:
        Array of shape ``(n_mels, n_frames)``, in dB.
    """
    power = power_spectrogram(y, n_fft=n_fft, hop_length=hop_length)
    fb = mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax)
    mel_power = fb @ power

    log_mel = 10.0 * np.log10(np.maximum(mel_power, 1e-10))
    if top_db is not None:
        log_mel = np.maximum(log_mel, log_mel.max() - top_db)

    return log_mel.astype(np.float32)


def mfcc(
    y: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    **kwargs,
) -> np.ndarray:
    """Compute MFCCs as the orthonormal DCT-II of the log-mel spectrogram.

    Args:
        y: 1-D signal.
        sample_rate: Sample rate of ``y``.
        n_mfcc: Number of cepstral coefficients to keep (including c0).
        **kwargs: Forwarded to :func:`log_mel_spectrogram`.

    Returns:
        Array of shape ``(n_mfcc, n_frames)``.
    """
    log_mel = log_mel_spectrogram(y, sample_rate=sample_rate, **kwargs)
    coeffs = dct(log_mel, type=2, axis=0, norm="ortho")
    return coeffs[:n_mfcc].astype(np.float32)


def _delta(x: np.ndarray) -> np.ndarray:
    """First-order temporal difference, edge-padded to preserve frame count."""
    if x.shape[1] < 2:
        return np.zeros_like(x)
    return np.diff(x, axis=1, prepend=x[:, :1])


def mfcc_feature_vector(
    y: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    **kwargs,
) -> np.ndarray:
    """Summarize a variable-length clip as one fixed-length MFCC vector.

    Calls differ in length, but RandomForest and SVM need a fixed-width input.
    We therefore collapse the time axis into per-coefficient statistics: the
    mean and standard deviation of each MFCC, plus the same for its first
    temporal difference (which captures how the call *evolves*, not just its
    average timbre).

    Args:
        y: 1-D signal.
        sample_rate: Sample rate of ``y``.
        n_mfcc: Number of cepstral coefficients.
        **kwargs: Forwarded to :func:`log_mel_spectrogram` via :func:`mfcc`.

    Returns:
        1-D array of length ``4 * n_mfcc`` (52 for the default n_mfcc=13),
        ordered as ``[mfcc_mean, mfcc_std, delta_mean, delta_std]``.
    """
    coeffs = mfcc(y, sample_rate=sample_rate, n_mfcc=n_mfcc, **kwargs)
    d = _delta(coeffs)
    return np.concatenate(
        [coeffs.mean(axis=1), coeffs.std(axis=1), d.mean(axis=1), d.std(axis=1)]
    ).astype(np.float32)


def fixed_size_log_mel(
    y: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_frames: int = CNN_N_FRAMES,
    **kwargs,
) -> np.ndarray:
    """Log-mel spectrogram cropped or padded to exactly ``n_frames`` frames.

    The CNN needs a fixed input size. Clips longer than ``n_frames`` are
    center-cropped (the informative part of a meow is rarely at the very edge);
    shorter clips are symmetrically padded with the clip's own minimum dB value,
    which is quieter-than-signal and so reads as silence rather than as an
    artificial edge.

    Args:
        y: 1-D signal.
        sample_rate: Sample rate of ``y``.
        n_frames: Target number of time frames.
        **kwargs: Forwarded to :func:`log_mel_spectrogram`.

    Returns:
        Array of shape ``(n_mels, n_frames)``.
    """
    log_mel = log_mel_spectrogram(y, sample_rate=sample_rate, **kwargs)
    cur = log_mel.shape[1]

    if cur == n_frames:
        return log_mel
    if cur > n_frames:
        start = (cur - n_frames) // 2
        return log_mel[:, start : start + n_frames]

    total_pad = n_frames - cur
    left = total_pad // 2
    right = total_pad - left
    return np.pad(
        log_mel, ((0, 0), (left, right)), mode="constant", constant_values=log_mel.min()
    )


def spectral_centroid(
    y: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Per-frame spectral centroid in Hz (the "center of mass" of the spectrum).

    Used in the EDA notebook as an interpretable, single-number proxy for
    "how bright / high-pitched does this call sound".

    Args:
        y: 1-D signal.
        sample_rate: Sample rate of ``y``.
        n_fft: FFT / frame size.
        hop_length: Hop between frames.

    Returns:
        1-D array of length ``n_frames``.
    """
    power = power_spectrogram(y, n_fft=n_fft, hop_length=hop_length)
    freqs = np.linspace(0.0, sample_rate / 2.0, power.shape[0])
    total = power.sum(axis=0)
    # Silent frames would divide by zero; report 0 Hz for them.
    total = np.where(total <= 0, 1e-12, total)
    return (freqs[:, None] * power).sum(axis=0) / total


def extract_feature_matrix(
    recordings: Iterable[Recording],
    kind: str = "mfcc",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features for a set of recordings.

    Args:
        recordings: Recordings to process, e.g. from :func:`scan_dataset`.
        kind: ``"mfcc"`` for flat summary vectors (baselines) or ``"logmel"``
            for fixed-size spectrogram patches (CNN).

    Returns:
        ``(X, y, groups)`` where ``X`` is the feature array, ``y`` holds the
        context letters (``B``/``F``/``I``) and ``groups`` holds the cat IDs to
        be passed to a grouped cross-validator.

    Raises:
        ValueError: If ``kind`` is not one of the supported values.
    """
    if kind not in {"mfcc", "logmel"}:
        raise ValueError(f"kind must be 'mfcc' or 'logmel', got {kind!r}")

    feats, labels, groups = [], [], []
    for rec in recordings:
        signal, sr = load_wav(rec.path)
        if kind == "mfcc":
            feats.append(mfcc_feature_vector(signal, sample_rate=sr))
        else:
            feats.append(fixed_size_log_mel(signal, sample_rate=sr))
        labels.append(rec.context)
        groups.append(rec.cat_id)

    return np.stack(feats), np.array(labels), np.array(groups)
