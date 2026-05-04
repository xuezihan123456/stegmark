from __future__ import annotations

import pytest

from stegmark.core.codec import bytes_to_bits, decode_bitstream, encode_text, resolve_payload_bits
from stegmark.exceptions import InvalidInputError


def test_codec_round_trip_text() -> None:
    bits = encode_text("Alice 2026")
    result = decode_bitstream(bits)

    assert result.valid is True
    assert result.message == "Alice 2026"


def test_codec_detects_corruption() -> None:
    bits = encode_text("Alice 2026")
    bits[-1] = 1 - bits[-1]

    result = decode_bitstream(bits)

    assert result.valid is False
    assert result.error == "crc_mismatch"


def test_resolve_payload_bits_uses_exactly_one_input_mode() -> None:
    assert resolve_payload_bits("Alice 2026", None) == tuple(encode_text("Alice 2026"))
    assert resolve_payload_bits(None, [1, 0, 1]) == (1, 0, 1)

    with pytest.raises(InvalidInputError, match="provide either message or payload bits"):
        resolve_payload_bits("Alice 2026", [1, 0, 1])

    with pytest.raises(InvalidInputError, match="missing message or payload bits"):
        resolve_payload_bits(None, None)


def test_bytes_to_bits_preserves_big_endian_bit_order() -> None:
    assert bytes_to_bits(bytes.fromhex("a53c")) == [
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
    ]
