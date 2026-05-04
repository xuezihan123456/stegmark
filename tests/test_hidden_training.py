from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from PIL import Image


def _require_onnx_export_stack() -> None:
    try:
        importlib.import_module("onnxscript")
    except Exception as exc:
        pytest.skip(f"onnx export stack unavailable: {exc}")


def test_hidden_trainer_config_defaults() -> None:
    from stegmark.training.trainer import HiddenTrainerConfig

    config = HiddenTrainerConfig(message_bits=32)

    assert config.image_size == 128
    assert config.batch_size > 0


def test_hidden_image_dataset_loads_tensor_and_message(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    from stegmark.data.dataset import HiddenImageDataset

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (24, 24), color=(12, 34, 56)).save(image_path)
    dataset = HiddenImageDataset(tmp_path, message_bits=12, image_size=16)

    image, message = dataset[0]

    assert image.shape == (3, 16, 16)
    assert image.dtype == torch.float32
    assert torch.all((0.0 <= image) & (image <= 1.0))
    assert message.shape == (12,)
    assert set(message.unique().tolist()).issubset({0.0, 1.0})


def test_hidden_trainer_train_step_returns_loss_values() -> None:
    torch = pytest.importorskip("torch")

    from stegmark.training.trainer import HiddenTrainer, HiddenTrainerConfig

    trainer = HiddenTrainer(HiddenTrainerConfig(message_bits=8, image_size=32, batch_size=2))
    batch = {
        "image": torch.rand(2, 3, 32, 32),
        "message": torch.randint(0, 2, (2, 8), dtype=torch.float32),
    }

    metrics = trainer.train_step(batch)

    assert set(metrics) == {"encoder_loss", "decoder_loss", "total_loss"}
    assert all(isinstance(value, float) for value in metrics.values())
    assert metrics["total_loss"] >= metrics["decoder_loss"]


def test_export_hidden_onnx_writes_encoder_and_decoder(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    _require_onnx_export_stack()

    from stegmark.training.export import export_hidden_onnx

    encoder_path = tmp_path / "encoder.onnx"
    decoder_path = tmp_path / "decoder.onnx"

    result = export_hidden_onnx(
        message_bits=32,
        encoder_output=encoder_path,
        decoder_output=decoder_path,
        image_size=32,
    )

    assert result == (encoder_path, decoder_path)
    assert encoder_path.exists()
    assert decoder_path.exists()
    assert encoder_path.stat().st_size > 0
    assert decoder_path.stat().st_size > 0


def test_export_hidden_onnx_raises_clear_error_when_onnxscript_stack_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("torch")

    from stegmark.training import export as export_module

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "onnxscript":
            raise ModuleNotFoundError("No module named 'onnxscript'")
        return real_import_module(name, package)

    monkeypatch.setattr(export_module.importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="ONNX export dependencies are unavailable or incompatible"):
        export_module.export_hidden_onnx(
            message_bits=8,
            encoder_output=tmp_path / "encoder.onnx",
            decoder_output=tmp_path / "decoder.onnx",
            image_size=16,
        )
