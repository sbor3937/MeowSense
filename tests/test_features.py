"""Hermetic unit tests for the DSP and metadata code in ``features``.

Everything here runs on synthesized audio in a temp directory. No network, no
dataset, no torch/sklearn -- numpy and scipy only, so this suite is fast enough
to run on every push.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import make_tone, write_wav

import features as F


# --------------------------------------------------------------------------
# Filename parsing
# --------------------------------------------------------------------------


class TestParseFilename:
    def test_parses_all_fields(self):
        rec = F.parse_filename("F_BAC01_MC_MN_SIM01_101.wav")
        assert rec.context == "F"
        assert rec.context_label == "food"
        assert rec.cat_id == "BAC01"
        assert rec.breed == "MC"
        assert rec.breed_label == "Maine Coon"
        assert rec.sex == "MN"
        assert rec.is_female is False
        assert rec.is_neutered is True
        assert rec.owner_id == "SIM01"
        assert rec.session == 1
        assert rec.counter == 1

    @pytest.mark.parametrize(
        "letter,label",
        [("B", "brushing"), ("F", "food"), ("I", "isolation")],
    )
    def test_all_contexts(self, letter, label):
        rec = F.parse_filename(f"{letter}_AAA01_EU_FN_OWN01_101.wav")
        assert rec.context == letter
        assert rec.context_label == label

    def test_sex_flags(self):
        assert F.parse_filename("B_A_MC_FI_O_101.wav").is_female is True
        assert F.parse_filename("B_A_MC_FI_O_101.wav").is_neutered is False
        assert F.parse_filename("B_A_MC_MN_O_101.wav").is_female is False
        assert F.parse_filename("B_A_MC_MN_O_101.wav").is_neutered is True

    def test_session_and_counter_are_split(self):
        rec = F.parse_filename("I_CAT01_EU_FN_OWN01_312.wav")
        assert rec.session == 3
        assert rec.counter == 12

    def test_path_is_preserved(self):
        rec = F.parse_filename("sub/dir/B_AAA01_MC_FN_OWN01_101.wav")
        assert rec.path == Path("sub/dir/B_AAA01_MC_FN_OWN01_101.wav")

    def test_as_dict_stringifies_path(self):
        d = F.parse_filename("B_AAA01_MC_FN_OWN01_101.wav").as_dict()
        assert isinstance(d["path"], str)
        assert d["context_label"] == "brushing"

    @pytest.mark.parametrize(
        "bad",
        [
            "garbage.wav",
            "X_AAA01_MC_FN_OWN01_101.wav",  # bad context letter
            "B_AAA01_XX_FN_OWN01_101.wav",  # bad breed
            "B_AAA01_MC_ZZ_OWN01_101.wav",  # bad sex
            "B_AAA01_MC_FN_OWN01_1.wav",    # session/counter too short
        ],
    )
    def test_invalid_names_raise(self, bad):
        with pytest.raises(ValueError):
            F.parse_filename(bad)


# --------------------------------------------------------------------------
# Mel filterbank
# --------------------------------------------------------------------------


class TestMelFilterbank:
    def test_shape(self):
        fb = F.mel_filterbank()
        assert fb.shape == (F.N_MELS, F.N_FFT // 2 + 1)

    def test_nonnegative(self):
        assert (F.mel_filterbank() >= 0).all()

    def test_no_dead_filters_at_dataset_settings(self):
        # Regression guard: at 8 kHz / n_fft=256 every mel band must catch at
        # least one FFT bin. If this ever fails, the low bands have collapsed.
        fb = F.mel_filterbank()
        assert (fb.sum(axis=1) > 0).all(), "some mel filter has zero energy"

    def test_rejects_fmax_above_nyquist(self):
        with pytest.raises(ValueError):
            F.mel_filterbank(sample_rate=8000, fmax=5000)

    def test_rejects_inverted_band(self):
        with pytest.raises(ValueError):
            F.mel_filterbank(fmin=3000, fmax=1000)

    def test_more_bands_still_valid(self):
        fb = F.mel_filterbank(n_mels=64)
        assert fb.shape == (64, F.N_FFT // 2 + 1)
        assert (fb.sum(axis=1) > 0).all()


# --------------------------------------------------------------------------
# Spectrograms, MFCCs and fixed-size framing
# --------------------------------------------------------------------------


class TestSpectralFeatures:
    def test_power_spectrogram_shape(self, tone):
        spec = F.power_spectrogram(tone)
        assert spec.shape[0] == F.N_FFT // 2 + 1

    def test_log_mel_shape(self, tone):
        S = F.log_mel_spectrogram(tone)
        assert S.shape[0] == F.N_MELS
        assert S.dtype == np.float32

    def test_log_mel_top_db_floor(self, tone):
        S = F.log_mel_spectrogram(tone, top_db=80.0)
        assert (S.max() - S.min()) <= 80.0 + 1e-3

    def test_mfcc_shape(self, tone):
        m = F.mfcc(tone, n_mfcc=13)
        assert m.shape[0] == 13

    def test_feature_vector_length_and_finite(self, tone):
        v = F.mfcc_feature_vector(tone, n_mfcc=13)
        assert v.shape == (4 * 13,)
        assert np.isfinite(v).all()

    def test_feature_vector_scales_with_n_mfcc(self, tone):
        assert F.mfcc_feature_vector(tone, n_mfcc=20).shape == (4 * 20,)

    def test_short_signal_still_produces_one_frame(self):
        # Shorter than a single FFT window -> must not crash, must give 1 frame.
        tiny = make_tone(1000.0, 0.01)
        assert F.log_mel_spectrogram(tiny).shape == (F.N_MELS, 1)

    def test_tone_centroid_tracks_frequency(self):
        # A pure tone's spectral centroid should sit near the tone frequency.
        low = np.mean(F.spectral_centroid(make_tone(500.0, 1.0)))
        high = np.mean(F.spectral_centroid(make_tone(2500.0, 1.0)))
        assert high > low
        assert abs(low - 500.0) < 300.0


class TestFixedSizeLogMel:
    def test_exact_frame_count_when_padded(self):
        S = F.fixed_size_log_mel(make_tone(1000.0, 0.3))  # short -> padded
        assert S.shape == (F.N_MELS, F.CNN_N_FRAMES)

    def test_exact_frame_count_when_cropped(self):
        S = F.fixed_size_log_mel(make_tone(1000.0, 6.0))  # long -> cropped
        assert S.shape == (F.N_MELS, F.CNN_N_FRAMES)

    def test_padding_uses_min_value(self):
        S = F.fixed_size_log_mel(make_tone(1000.0, 0.3), n_frames=200)
        # Padded columns are filled with the spectrogram minimum, so the global
        # min appears in the first (padded) column.
        assert np.isclose(S[:, 0].min(), S.min())

    def test_custom_frame_count(self):
        S = F.fixed_size_log_mel(make_tone(1000.0, 2.0), n_frames=64)
        assert S.shape == (F.N_MELS, 64)


# --------------------------------------------------------------------------
# Audio I/O
# --------------------------------------------------------------------------


class TestLoadWav:
    def test_roundtrip_range_and_dtype(self, tmp_path):
        path = write_wav(tmp_path / "t.wav", make_tone(1000.0, 0.5))
        y, sr = F.load_wav(path)
        assert sr == F.SAMPLE_RATE
        assert y.dtype == np.float32
        assert y.ndim == 1
        assert np.abs(y).max() <= 1.0

    def test_resamples_to_target(self, tmp_path):
        # Written at 16 kHz, requested at 8 kHz -> about half the samples.
        path = write_wav(tmp_path / "hi.wav", make_tone(1000.0, 1.0, 16000), 16000)
        y, sr = F.load_wav(path, target_sr=8000)
        assert sr == 8000
        assert abs(len(y) - 8000) < 200

    def test_stereo_is_downmixed(self, tmp_path):
        from scipy.io import wavfile

        mono = (make_tone(1000.0, 0.5) * 32767).astype(np.int16)
        stereo = np.stack([mono, mono], axis=1)
        path = tmp_path / "st.wav"
        wavfile.write(str(path), F.SAMPLE_RATE, stereo)
        y, _ = F.load_wav(path)
        assert y.ndim == 1


# --------------------------------------------------------------------------
# Dataset scanning
# --------------------------------------------------------------------------


class TestScanDataset:
    def test_finds_valid_and_skips_invalid(self, synthetic_dataset):
        recs = F.scan_dataset(synthetic_dataset)
        assert len(recs) == 6  # the malformed name is skipped, not fatal
        assert {r.cat_id for r in recs} == {"AAA01", "BBB02"}
        assert {r.context for r in recs} == {"B", "F", "I"}

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            F.scan_dataset(tmp_path / "does_not_exist")

    def test_deterministic_order(self, synthetic_dataset):
        a = [r.path.name for r in F.scan_dataset(synthetic_dataset)]
        b = [r.path.name for r in F.scan_dataset(synthetic_dataset)]
        assert a == b == sorted(a)

    def test_extract_feature_matrix_mfcc(self, synthetic_dataset):
        recs = F.scan_dataset(synthetic_dataset)
        X, y, groups = F.extract_feature_matrix(recs, kind="mfcc")
        assert X.shape == (6, 4 * F.N_MFCC)
        assert len(y) == len(groups) == 6
        assert set(groups) == {"AAA01", "BBB02"}

    def test_extract_feature_matrix_logmel(self, synthetic_dataset):
        recs = F.scan_dataset(synthetic_dataset)
        X, y, groups = F.extract_feature_matrix(recs, kind="logmel")
        assert X.shape == (6, F.N_MELS, F.CNN_N_FRAMES)

    def test_extract_feature_matrix_rejects_bad_kind(self, synthetic_dataset):
        recs = F.scan_dataset(synthetic_dataset)
        with pytest.raises(ValueError):
            F.extract_feature_matrix(recs, kind="nope")
