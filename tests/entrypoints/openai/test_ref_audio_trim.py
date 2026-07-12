"""Tests for reference-audio leading-silence trimming."""

import numpy as np
import pytest

from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

SR = 24000
_trim = OmniOpenAIServingSpeech._trim_leading_silence


def _tone(dur_s: float, amp: float = 0.5, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(dur_s * SR), dtype=np.float32) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(dur_s: float) -> np.ndarray:
    return np.zeros(int(dur_s * SR), dtype=np.float32)


def test_trims_leading_silence():
    wav = np.concatenate([_silence(0.4), _tone(1.0)])
    out = _trim(wav, SR)
    removed_ms = (len(wav) - len(out)) / SR * 1000.0
    assert 370.0 <= removed_ms <= 390.0
    assert np.abs(out).max() == pytest.approx(np.abs(wav).max(), rel=1e-3)


def test_clean_audio_is_untouched():
    wav = _tone(1.0)
    out = _trim(wav, SR)
    assert len(out) == len(wav)
    assert np.array_equal(out, wav)


def test_all_silence_returns_input_unchanged():
    wav = _silence(1.0)
    out = _trim(wav, SR)
    assert len(out) == len(wav)
    assert out.size > 0


def test_quiet_recording_is_not_gutted():
    wav = np.concatenate([_silence(0.4), _tone(1.0, amp=0.05)])
    out = _trim(wav, SR)
    removed_ms = (len(wav) - len(out)) / SR * 1000.0
    assert 370.0 <= removed_ms <= 390.0
    assert np.abs(out).max() == pytest.approx(0.05, rel=1e-2)


def test_refuses_to_remove_most_of_the_clip():
    wav = np.concatenate([_silence(2.0), _tone(0.05)])
    out = _trim(wav, SR)
    assert len(out) == len(wav)


def test_output_is_contiguous():
    wav = np.concatenate([_silence(0.4), _tone(1.0)])
    out = _trim(wav, SR)
    assert out.flags["C_CONTIGUOUS"]


def test_empty_input():
    wav = np.array([], dtype=np.float32)
    out = _trim(wav, SR)
    assert out.size == 0


def test_dc_only_input():
    wav = np.zeros(SR, dtype=np.float32)
    out = _trim(wav, SR)
    assert len(out) == len(wav)
