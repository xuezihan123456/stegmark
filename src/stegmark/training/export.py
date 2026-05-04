from __future__ import annotations

import importlib
from pathlib import Path

import torch

from stegmark.logging_utils import logger
from stegmark.nn.hidden.decoder import HiddenDecoder
from stegmark.nn.hidden.encoder import HiddenEncoder


def _ensure_onnx_export_stack() -> None:
    try:
        importlib.import_module("onnxscript")
    except Exception as exc:  # pragma: no cover - exercised via tests with monkeypatch
        raise RuntimeError(
            "ONNX export dependencies are unavailable or incompatible. "
            "Install the train extra, and on Python 3.13 keep the export stack separate from trustmark."
        ) from exc


def export_hidden_onnx(
    *,
    message_bits: int,
    encoder_output: Path,
    decoder_output: Path,
    image_size: int = 128,
    encoder_ckpt: Path | None = None,
    decoder_ckpt: Path | None = None,
) -> tuple[Path, Path]:
    _ensure_onnx_export_stack()

    encoder = HiddenEncoder(message_bits=message_bits).eval()
    decoder = HiddenDecoder(message_bits=message_bits).eval()

    if encoder_ckpt is not None:
        encoder.load_state_dict(torch.load(encoder_ckpt, map_location="cpu"))
        logger.info("Loaded encoder from %s", encoder_ckpt)
    if decoder_ckpt is not None:
        decoder.load_state_dict(torch.load(decoder_ckpt, map_location="cpu"))
        logger.info("Loaded decoder from %s", decoder_ckpt)

    encoder_output.parent.mkdir(parents=True, exist_ok=True)
    decoder_output.parent.mkdir(parents=True, exist_ok=True)

    image = torch.rand(1, 3, image_size, image_size)
    message = torch.randint(0, 2, (1, message_bits), dtype=torch.float32)

    torch.onnx.export(
        encoder,
        (image, message),
        encoder_output,
        input_names=["image", "message"],
        output_names=["encoded"],
        opset_version=17,
    )
    torch.onnx.export(
        decoder,
        (image,),
        decoder_output,
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
    )
    return encoder_output, decoder_output
