"""Tests for text and audio watermark engines."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stegmark.core.audio_engine import AudioWatermarkEngine
from stegmark.core.text_engine import TextWatermarkEngine
from stegmark.exceptions import InvalidInputError, MessageTooLongError


def test_text_engine_round_trip_recovers_bits():
    text = "The quick brown fox jumps over the lazy dog and the system tracks every change carefully today"
    payload = [1, 0, 1, 1]
    engine = TextWatermarkEngine()

    result = engine.embed(text, payload, key=42)
    recovered = engine.extract(result.text, num_bits=len(payload), key=42)

    assert recovered == payload


def test_text_engine_different_keys_can_diverge():
    text = "The quick brown fox jumps over the lazy dog and the system tracks every change carefully today"
    payload = [1, 0, 1, 1]
    engine = TextWatermarkEngine()

    result = engine.embed(text, payload, key=1)
    recovered_wrong = engine.extract(result.text, num_bits=len(payload), key=999)

    assert isinstance(recovered_wrong, list)
    assert len(recovered_wrong) == len(payload)


def test_text_engine_capacity_bits_reasonable():
    text = "one two three four five six seven eight nine ten eleven twelve"
    engine = TextWatermarkEngine()

    capacity = engine.capacity_bits(text)

    assert capacity == 12 // 4


def test_text_engine_rejects_oversized_payload():
    text = "short text only here"
    engine = TextWatermarkEngine()

    with pytest.raises(MessageTooLongError):
        engine.embed(text, [1] * 100)


def test_audio_engine_round_trip_recovers_byte(tmp_path: Path):
    src = AudioWatermarkEngine.synth_noise(tmp_path / "in.wav", duration_seconds=4.0, rate=16000, seed=42)
    engine = AudioWatermarkEngine()
    payload_bits = [1, 0, 1, 1, 0, 0, 1, 0]
    out = tmp_path / "out.wav"

    engine.embed(src, payload_bits, out)
    recovered = engine.extract(out, num_bits=8)

    matches = sum(1 for a, b in zip(payload_bits, recovered) if a == b)
    assert matches >= 7  # noise carrier is friendly to echo-hiding


def test_audio_engine_rejects_oversized_payload(tmp_path: Path):
    src = AudioWatermarkEngine.synth_sine(tmp_path / "tiny.wav", duration_seconds=0.5, rate=16000)
    engine = AudioWatermarkEngine()

    with pytest.raises(MessageTooLongError):
        engine.embed(src, [1] * 1000, tmp_path / "out.wav")


def test_audio_engine_rejects_stereo(tmp_path: Path):
    import wave

    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(2)
        fh.setsampwidth(2)
        fh.setframerate(16000)
        fh.writeframes(np.zeros(16000 * 2, dtype=np.int16).tobytes())

    with pytest.raises(InvalidInputError):
        AudioWatermarkEngine().extract(path, num_bits=1)


def test_audio_engine_capacity_bits_matches_segment_size(tmp_path: Path):
    src = AudioWatermarkEngine.synth_sine(tmp_path / "in.wav", duration_seconds=2.0, rate=16000)
    engine = AudioWatermarkEngine()

    capacity = engine.capacity_bits(src)

    assert capacity == (16000 * 2) // engine.segment_samples
