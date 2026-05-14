"""Tests for the FEC layer."""
from __future__ import annotations

import numpy as np
import pytest

from stegmark.core.ecc import ECCCodec, ECCLevel, HammingCode74, RepetitionCode


def test_repetition_encoded_length():
    code = RepetitionCode(3)
    bits = np.array([1, 0, 1, 1], dtype=np.uint8)

    encoded = code.encode(bits)

    assert encoded.size == 12
    assert encoded.tolist() == [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]


def test_repetition_majority_vote_corrects_single_error():
    code = RepetitionCode(3)
    original = np.array([1, 0, 1], dtype=np.uint8)
    encoded = code.encode(original).copy()
    encoded[1] ^= 1  # flip one bit in first group

    decoded, corrected = code.decode(encoded, original_length=3)

    assert decoded.tolist() == original.tolist()
    assert corrected >= 1


def test_repetition_rejects_even_factor():
    with pytest.raises(ValueError):
        RepetitionCode(2)


def test_hamming74_round_trip_no_error():
    code = HammingCode74()
    bits = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)

    encoded = code.encode(bits)
    decoded, corrected = code.decode(encoded, original_length=bits.size)

    assert decoded.tolist() == bits.tolist()
    assert corrected == 0


def test_hamming74_corrects_single_bit_error():
    code = HammingCode74()
    bits = np.array([1, 0, 1, 1], dtype=np.uint8)
    encoded = code.encode(bits).copy()
    encoded[2] ^= 1

    decoded, corrected = code.decode(encoded, original_length=4)

    assert decoded.tolist() == bits.tolist()
    assert corrected == 1


def test_eccc_codec_none_is_identity():
    codec = ECCCodec()
    bits = np.array([1, 1, 0, 1, 0], dtype=np.uint8)

    encoded, stats = codec.encode(bits, ECCLevel.NONE)
    decoded, _ = codec.decode(encoded, ECCLevel.NONE, original_length=bits.size)

    assert decoded.tolist() == bits.tolist()
    assert stats.encoded_bits == bits.size


def test_ecc_codec_light_round_trip():
    codec = ECCCodec()
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)

    encoded, _ = codec.encode(bits, ECCLevel.LIGHT)
    decoded, _ = codec.decode(encoded, ECCLevel.LIGHT, original_length=bits.size)

    assert decoded.tolist() == bits.tolist()


def test_ecc_codec_heavy_tolerates_20pct_bit_errors():
    codec = ECCCodec()
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=32, dtype=np.uint8)
    encoded, _ = codec.encode(bits, ECCLevel.HEAVY)

    error_mask = rng.choice([0, 1], size=encoded.size, p=[0.8, 0.2]).astype(np.uint8)
    corrupted = (encoded ^ error_mask).astype(np.uint8)

    decoded, stats = codec.decode(corrupted, ECCLevel.HEAVY, original_length=bits.size)

    ber = float(np.mean(decoded != bits))
    assert ber < 0.10
    assert stats.corrected_errors > 0


def test_ecc_codec_medium_round_trip():
    codec = ECCCodec()
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=np.uint8)

    encoded, _ = codec.encode(bits, ECCLevel.MEDIUM)
    decoded, stats = codec.decode(encoded, ECCLevel.MEDIUM, original_length=bits.size)

    assert decoded.tolist() == bits.tolist()
    assert stats.encoded_bits == encoded.size


def test_ecc_codec_encoded_length_helper():
    assert ECCCodec.encoded_length(8, ECCLevel.NONE) == 8
    assert ECCCodec.encoded_length(8, ECCLevel.LIGHT) == 14
    assert ECCCodec.encoded_length(8, ECCLevel.MEDIUM) == 42
    assert ECCCodec.encoded_length(8, ECCLevel.HEAVY) == 70


def test_ecc_codec_rejects_unknown_level():
    class FakeLevel:
        value = 99

    with pytest.raises(ValueError):
        ECCCodec.encoded_length(8, FakeLevel)  # type: ignore[arg-type]
