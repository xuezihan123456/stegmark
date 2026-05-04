from __future__ import annotations

import pickle

import numpy as np

from stegmark import EmbedResult
from stegmark.types import ImageMetadata


def test_image_metadata_copies_and_freezes_extra_values() -> None:
    preview = np.arange(16, dtype=np.uint8).reshape(4, 4)
    tags = ["alpha", "beta"]
    metadata = ImageMetadata(extras={"preview": preview, "tags": tags})

    preview[0, 0] = 255
    tags.append("gamma")

    frozen_preview = metadata.extras["preview"]
    assert isinstance(frozen_preview, np.ndarray)
    assert frozen_preview[0, 0] == 0
    assert frozen_preview.flags.writeable is False
    assert metadata.extras["tags"] == ("alpha", "beta")


def test_image_metadata_repr_summarizes_large_arrays() -> None:
    metadata = ImageMetadata(extras={"preview": np.arange(64, dtype=np.uint8).reshape(8, 8)})

    rendered = repr(metadata)

    assert "ndarray(shape=(8, 8), dtype=uint8)" in rendered
    assert "[[0" not in rendered


def test_embed_result_no_longer_exposes_save_method() -> None:
    assert not hasattr(EmbedResult, "save")


def test_image_metadata_is_pickleable_for_process_workers() -> None:
    metadata = ImageMetadata(extras={"preview": np.arange(4, dtype=np.uint8)})

    restored = pickle.loads(pickle.dumps(metadata))

    preview = restored.extras["preview"]
    assert isinstance(preview, np.ndarray)
    assert preview.tolist() == [0, 1, 2, 3]
    assert preview.flags.writeable is False
