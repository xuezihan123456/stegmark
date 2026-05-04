from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from stegmark.core.hidden import HiddenEngine
from stegmark.core.registry import get_engine
from stegmark.exceptions import EngineUnavailableError, MessageTooLongError


def test_hidden_engine_lookup_returns_hidden_backend() -> None:
    engine = get_engine("hidden")

    assert engine.name == "hidden"


def test_hidden_engine_reports_missing_weights(sample_image: np.ndarray) -> None:
    engine = get_engine("hidden")

    with pytest.raises(EngineUnavailableError):
        engine.decode(sample_image)


def test_hidden_engine_round_trip_with_onnx_models(
    tmp_path: Path, sample_image: np.ndarray
) -> None:
    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")
    try:
        import ml_dtypes  # noqa: F401
        ml_dtypes.float4_e2m1fn
    except AttributeError:
        pytest.skip("ml_dtypes version incompatible with onnx (missing float4_e2m1fn)")

    model_dir = tmp_path / "hidden"
    model_dir.mkdir()
    _export_test_hidden_models(torch, model_dir, message_bits=136)
    engine = HiddenEngine(model_dir=model_dir)

    encoded = engine.encode(sample_image, "HiDDeN")
    decoded = engine.decode(encoded)

    assert encoded.shape == sample_image.shape
    assert decoded.found is True
    assert decoded.message == "HiDDeN"


def test_hidden_engine_rejects_messages_beyond_model_capacity(
    tmp_path: Path, sample_image: np.ndarray
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    try:
        import ml_dtypes  # noqa: F401
        ml_dtypes.float4_e2m1fn
    except AttributeError:
        pytest.skip("ml_dtypes version incompatible with onnx (missing float4_e2m1fn)")

    model_dir = tmp_path / "hidden"
    model_dir.mkdir()
    _export_test_hidden_models(torch, model_dir, message_bits=16)
    engine = HiddenEngine(model_dir=model_dir)

    with pytest.raises(MessageTooLongError):
        engine.encode(sample_image, "this message is much too long")


def test_hidden_engine_initializes_encoder_session_once_under_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = HiddenEngine(model_dir=tmp_path)
    session = object()
    call_count = 0
    call_count_lock = threading.Lock()
    create_started = threading.Event()
    release_create = threading.Event()
    results: list[object] = []

    def fake_create_session(path: Path) -> object:
        del path
        nonlocal call_count
        with call_count_lock:
            call_count += 1
            create_started.set()
        release_create.wait(timeout=1.0)
        return session

    monkeypatch.setattr(engine, "_create_session", fake_create_session)

    def access_encoder() -> None:
        results.append(engine._encoder)

    first = threading.Thread(target=access_encoder)
    second = threading.Thread(target=access_encoder)
    first.start()
    assert create_started.wait(timeout=1.0)
    second.start()
    time.sleep(0.05)
    assert call_count == 1
    release_create.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert call_count == 1
    assert results == [session, session]


def test_hidden_engine_prefers_configured_gpu_provider_before_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "stegmark.core.hidden.load_config",
        lambda: SimpleNamespace(
            hidden_model_dir=tmp_path,
            hidden_providers=("CUDAExecutionProvider",),
        ),
    )
    fake_ort = ModuleType("onnxruntime")
    calls: list[list[str]] = []

    def fake_inference_session(path: str, *, providers: list[str]) -> object:
        del path
        calls.append(list(providers))
        return object()

    fake_ort.InferenceSession = fake_inference_session  # type: ignore[attr-defined]
    fake_ort.get_available_providers = lambda: [  # type: ignore[attr-defined]
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    engine = HiddenEngine()
    _ = engine._create_session(tmp_path / "encoder.onnx")

    assert calls == [["CUDAExecutionProvider", "CPUExecutionProvider"]]


def test_hidden_engine_falls_back_to_cpu_when_gpu_session_init_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "stegmark.core.hidden.load_config",
        lambda: SimpleNamespace(
            hidden_model_dir=tmp_path,
            hidden_providers=("CUDAExecutionProvider",),
        ),
    )
    fake_ort = ModuleType("onnxruntime")
    calls: list[list[str]] = []

    def fake_inference_session(path: str, *, providers: list[str]) -> object:
        del path
        calls.append(list(providers))
        if providers[0] == "CUDAExecutionProvider":
            raise RuntimeError("CUDA init failed")
        return object()

    fake_ort.InferenceSession = fake_inference_session  # type: ignore[attr-defined]
    fake_ort.get_available_providers = lambda: [  # type: ignore[attr-defined]
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    engine = HiddenEngine()
    _ = engine._create_session(tmp_path / "encoder.onnx")

    assert calls == [
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]


def _export_test_hidden_models(
    torch_module: Any, model_dir: Path, *, message_bits: int
) -> None:
    class DeterministicEncoder(torch_module.nn.Module):
        def __init__(self, bits: int) -> None:
            super().__init__()
            self.bits = bits

        def forward(self, image, message):
            batch_size = image.shape[0]
            flat = image[:, 0, :, :].reshape(batch_size, -1)
            indices = torch_module.arange(self.bits, device=image.device).unsqueeze(0).expand(batch_size, -1)
            updated = flat.scatter(1, indices, message)
            channel = updated.reshape(batch_size, image.shape[2], image.shape[3]).unsqueeze(1)
            return torch_module.cat([channel, image[:, 1:, :, :]], dim=1)

    class DeterministicDecoder(torch_module.nn.Module):
        def __init__(self, bits: int) -> None:
            super().__init__()
            self.bits = bits

        def forward(self, image):
            flat = image[:, 0, :, :].reshape(image.shape[0], -1)
            return (flat[:, : self.bits] * 2.0) - 1.0

    image = torch_module.rand(1, 3, 128, 128)
    message = torch_module.randint(
        0,
        2,
        (1, message_bits),
        dtype=torch_module.float32,
    )

    torch_module.onnx.export(
        DeterministicEncoder(message_bits).eval(),
        (image, message),
        model_dir / "encoder.onnx",
        input_names=["image", "message"],
        output_names=["encoded"],
        opset_version=17,
    )
    torch_module.onnx.export(
        DeterministicDecoder(message_bits).eval(),
        (image,),
        model_dir / "decoder.onnx",
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
    )
