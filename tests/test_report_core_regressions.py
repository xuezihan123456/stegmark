from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

import stegmark
from stegmark.core.codec import bytes_to_bits, resolve_payload_bits
from stegmark.core.native import _dct_matrix
from stegmark.core.registry import get_engine
from stegmark.types import ImageMetadata


def test_image_metadata_extras_is_immutable() -> None:
    metadata = ImageMetadata(extras={"key": "value"})

    assert isinstance(metadata.extras, MappingProxyType)
    with pytest.raises(TypeError):
        metadata.extras["other"] = "x"  # type: ignore[index]


def test_registry_caches_engine_instances() -> None:
    left = get_engine("native")
    right = get_engine("native")

    assert left is right


def test_stegmark_repr_exposes_engine_and_strength() -> None:
    client = stegmark.StegMark(engine="native", strength=0.5)

    assert "native" in repr(client)
    assert "0.5" in repr(client)


def test_dct_matrix_is_read_only() -> None:
    matrix = _dct_matrix(8)

    assert matrix.flags.writeable is False
    with pytest.raises(ValueError):
        matrix[0, 0] = 0.0


def test_resolve_payload_bits_rejects_dual_input() -> None:
    with pytest.raises(Exception):
        resolve_payload_bits("hello", (1, 0, 1))


def test_bytes_to_bits_preserves_existing_bit_order() -> None:
    assert bytes_to_bits(bytes([0xA5])) == [1, 0, 1, 0, 0, 1, 0, 1]


def test_image_metadata_repr_summarizes_large_array() -> None:
    result = stegmark.EmbedResult(
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        engine="native",
        message="hello",
        bits=(1, 0, 1),
        metadata=ImageMetadata(),
    )

    rendered = repr(result)

    assert "shape=(16, 16, 3)" in rendered
    assert "[[" not in rendered
