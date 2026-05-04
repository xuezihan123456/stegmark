from __future__ import annotations

from pathlib import Path

from stegmark.service import embed_file, extract_file, info_file, verify_file


def test_embed_file_then_extract(tmp_path: Path, sample_image_path: Path) -> None:
    output = tmp_path / "output.png"

    embed_result = embed_file(sample_image_path, output, message="Alice 2026", engine="native")
    extract_result = extract_file(output, engine="native")

    assert output.exists()
    assert embed_result.output_path == output
    assert extract_result.message == "Alice 2026"


def test_verify_and_info(sample_image_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "verify.png"
    embed_file(sample_image_path, output, message="Alice 2026", engine="native")

    verify_result = verify_file(output, "Alice 2026", engine="native")
    info_result = info_file(output, engine="native")

    assert verify_result.matched is True
    assert info_result.found is True
    assert info_result.width == 128
    assert info_result.height == 128


def test_embed_file_supports_bits_with_0x_prefix(
    sample_image_path: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "bits-prefix.png"

    embed_result = embed_file(
        sample_image_path,
        output,
        bits_hex="0xdeadbeef",
        engine="native",
    )
    extract_result = extract_file(output, engine="native")

    assert embed_result.payload == bytes.fromhex("deadbeef")
    assert extract_result.payload_hex == "deadbeef"
