from __future__ import annotations

from pathlib import Path

import pytest

from stegmark.core.codec import decode_bitstream, encode_bits_hex
from stegmark.exceptions import InvalidInputError
from stegmark.service import embed_file, extract_file


def test_encode_bits_hex_round_trip() -> None:
    bits = encode_bits_hex("deadbeef")
    decoded = decode_bitstream(bits)

    assert decoded.valid is True
    assert decoded.payload == bytes.fromhex("deadbeef")


def test_encode_bits_hex_rejects_invalid_hex() -> None:
    with pytest.raises(InvalidInputError):
        encode_bits_hex("not-hex")


def test_embed_file_with_bits_round_trip(
    sample_image_path: Path, tmp_path: Path
) -> None:
    output = tmp_path / "bits-output.png"

    embed_result = embed_file(
        sample_image_path,
        output,
        bits_hex="deadbeef",
        engine="native",
    )
    extract_result = extract_file(output, engine="native")

    assert embed_result.output_path == output
    assert extract_result.payload == bytes.fromhex("deadbeef")
