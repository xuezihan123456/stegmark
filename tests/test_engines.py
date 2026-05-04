from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from stegmark.core import registry
from stegmark.core.engine import WatermarkEngine
from stegmark.core.hidden import HiddenEngine
from stegmark.core.native import NativeEngine, _block_view, _dct_matrix, _iter_blocks
from stegmark.core.registry import get_engine
from stegmark.core.trustmark import TrustMarkEngine
from stegmark.types import ExtractResult

try:
    from stegmark.core.engine import EngineCapabilities
except ImportError:  # pragma: no cover - compatibility with parallel worktrees
    EngineCapabilities = None


def test_native_engine_round_trip(sample_image) -> None:
    engine = get_engine("native")

    encoded = engine.encode(sample_image, "Alice 2026")
    extracted = engine.decode(encoded)

    assert extracted.found is True
    assert extracted.message == "Alice 2026"


def test_auto_engine_resolves_to_available_backend() -> None:
    engine = get_engine("auto")

    assert engine.name == "native"


def test_native_block_view_exposes_usable_blocks_in_place() -> None:
    channel = np.arange(10 * 18, dtype=np.float32).reshape(10, 18)

    blocks = _block_view(channel)

    assert blocks.shape == (1, 2, 8, 8)
    blocks[0, 1, 0, 0] = -1.0
    assert channel[0, 8] == -1.0


def test_native_iter_blocks_is_lazy() -> None:
    blocks = _iter_blocks(16, 24)

    assert not isinstance(blocks, list)
    assert list(blocks) == [(0, 0), (0, 8), (0, 16), (8, 0), (8, 8), (8, 16)]


def test_native_dct_matrix_is_read_only() -> None:
    matrix = _dct_matrix(8)

    assert matrix.flags.writeable is False


CAPABILITY_CASES = (
    []
    if EngineCapabilities is None
    else [
        (
            NativeEngine,
            EngineCapabilities(
                supports_text_messages=True,
                supports_payload_bits=True,
                supports_strength_control=True,
                requires_optional_dependency=False,
                requires_model_files=False,
            ),
        ),
        (
            HiddenEngine,
            EngineCapabilities(
                supports_text_messages=True,
                supports_payload_bits=True,
                supports_strength_control=False,
                requires_optional_dependency=True,
                requires_model_files=True,
            ),
        ),
        (
            TrustMarkEngine,
            EngineCapabilities(
                supports_text_messages=True,
                supports_payload_bits=False,
                supports_strength_control=False,
                requires_optional_dependency=True,
                requires_model_files=False,
            ),
        ),
    ]
)


@pytest.mark.skipif(EngineCapabilities is None, reason="engine capabilities are not implemented in this worktree")
@pytest.mark.parametrize(("engine_cls", "expected"), CAPABILITY_CASES)
def test_engine_classes_declare_capabilities(
    engine_cls: type[WatermarkEngine], expected: object
) -> None:
    assert engine_cls.describe_capabilities() == expected


def test_get_engine_caches_instances() -> None:
    registry.clear_engine_cache()

    native = get_engine("native")
    alias = get_engine("auto")

    assert native is get_engine("native")
    assert alias is native


def test_get_engine_discovers_entry_point_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    registry.clear_engine_cache()
    if EngineCapabilities is None:
        pytest.skip("engine capabilities are not implemented in this worktree")

    class PluginEngine(WatermarkEngine):
        name = "plugin"
        declared_capabilities = EngineCapabilities(
            supports_text_messages=True,
            supports_payload_bits=False,
            supports_strength_control=False,
            requires_optional_dependency=False,
            requires_model_files=False,
        )

        def encode(
            self,
            image: np.ndarray,
            message: str | None = None,
            *,
            payload_bits: Sequence[int] | None = None,
            strength: float = 1.0,
        ) -> np.ndarray:
            del message, payload_bits, strength
            return image

        def decode(self, image: np.ndarray) -> ExtractResult:
            del image
            return ExtractResult(found=False, engine=self.name, error="not_implemented")

    class FakeEntryPoint:
        name = "plugin"

        def load(self) -> type[PluginEngine]:
            return PluginEngine

    monkeypatch.setattr(registry, "_iter_entry_points", lambda: (FakeEntryPoint(),))

    engine = registry.get_engine("plugin")

    assert isinstance(engine, PluginEngine)
    assert registry.get_engine("plugin") is engine
    assert engine.capabilities.supports_payload_bits is False
