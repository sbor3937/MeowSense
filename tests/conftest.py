"""Shared pytest fixtures and configuration.

The unit tests are hermetic: they synthesize their own audio and never touch
the network or the real dataset. The data-dependent reproduction tests live in
``test_reproduce.py`` behind the ``data`` marker so they can be deselected with
``-m "not data"``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

# Make src/ importable exactly the way the scripts and notebooks do.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from features import SAMPLE_RATE  # noqa: E402


def make_tone(freq: float, duration_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a unit-amplitude sine wave as float32 in [-1, 1].

    Args:
        freq: Tone frequency in Hz.
        duration_s: Length in seconds.
        sample_rate: Sample rate.

    Returns:
        1-D float32 signal.
    """
    t = np.arange(int(round(duration_s * sample_rate))) / sample_rate
    return (0.8 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def write_wav(
    path: Path,
    signal: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    """Write a float signal to a 16-bit PCM WAV file.

    Args:
        path: Destination path.
        signal: 1-D float signal in [-1, 1].
        sample_rate: Sample rate to record in the header.

    Returns:
        The path written.
    """
    pcm = np.clip(signal, -1.0, 1.0)
    pcm = (pcm * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(str(path), sample_rate, pcm)
    return path


@pytest.fixture
def tone() -> np.ndarray:
    """A 1000 Hz, 1.5 s tone at the dataset sample rate."""
    return make_tone(1000.0, 1.5)


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """A tiny on-disk CatMeows-shaped dataset for I/O and scanning tests.

    Builds 6 correctly-named WAVs across 2 cats and all 3 contexts, plus one
    file with an invalid name that a scanner must skip. Returns the directory.
    """
    specs = [
        "B_AAA01_MC_FN_OWN01_101",
        "F_AAA01_MC_FN_OWN01_201",
        "I_AAA01_MC_FN_OWN01_301",
        "B_BBB02_EU_MN_OWN02_101",
        "F_BBB02_EU_MN_OWN02_202",
        "I_BBB02_EU_MN_OWN02_303",
    ]
    for i, stem in enumerate(specs):
        write_wav(tmp_path / f"{stem}.wav", make_tone(500 + 200 * i, 1.2))

    # A file that does not follow the convention -- scanners must ignore it.
    write_wav(tmp_path / "not_a_catmeow.wav", make_tone(440.0, 0.5))

    return tmp_path
