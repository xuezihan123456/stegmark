from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import stegmark

TMP_ROOT = Path(__file__).resolve().parent
TMP_FILES = (
    "sample.png",
    "api-output.png",
    "session-output.png",
    "bits-output.png",
    "immutable-metadata.png",
)


@pytest.fixture()
def tmp_path() -> Iterator[Path]:
    for name in TMP_FILES:
        path = TMP_ROOT / name
        if path.exists():
            path.unlink()
    yield TMP_ROOT
    for name in TMP_FILES:
        path = TMP_ROOT / name
        if path.exists():
            path.unlink()


def test_public_api_round_trip(sample_image_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "api-output.png"

    embed_result = stegmark.embed(sample_image_path, "Alice 2026", output=output, engine="native")
    extract_result = stegmark.extract(output, engine="native")
    verify_result = stegmark.verify(output, "Alice 2026", engine="native")
    info_result = stegmark.info(output, engine="native")

    assert output.exists()
    assert embed_result.output_path == output
    assert extract_result.message == "Alice 2026"
    assert verify_result.matched is True
    assert info_result.found is True


def test_session_api_round_trip(sample_image_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "session-output.png"

    with stegmark.StegMark(engine="native") as client:
        client.embed(sample_image_path, "Studio X", output=output)
        extracted = client.extract(output)

    assert extracted.message == "Studio X"


def test_public_api_bits_round_trip(sample_image_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "bits-output.png"

    stegmark.embed(sample_image_path, bits="deadbeef", output=output, engine="native")
    result = stegmark.extract(output, engine="native")

    assert result.payload_hex == "deadbeef"


def test_public_api_embed_returns_immutable_metadata(sample_image_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "immutable-metadata.png"

    result = stegmark.embed(sample_image_path, "Alice 2026", output=output, engine="native")

    with pytest.raises(TypeError):
        result.metadata.extras["author"] = "Alice"


def test_public_api_exposes_engine_lookup_and_capabilities() -> None:
    if not hasattr(stegmark, "get_engine"):
        pytest.skip("extended engine lookup API is not available in this worktree")

    engine = stegmark.get_engine("native")

    assert engine.name == "native"
    assert engine.capabilities.supports_payload_bits is True
