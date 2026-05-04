from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from stegmark.config import load_config
from stegmark.core.codec import decode_bitstream, resolve_payload_bits
from stegmark.core.engine import EngineCapabilities, WatermarkEngine
from stegmark.core.weights import resolve_hidden_execution_providers
from stegmark.exceptions import EngineUnavailableError, MessageTooLongError
from stegmark.types import ExtractResult, ImageArray


class HiddenEngine(WatermarkEngine):
    name = "hidden"
    declared_capabilities = EngineCapabilities(
        supports_strength_control=False,
        requires_optional_dependency=True,
        requires_model_files=True,
    )

    def __init__(self, *, model_dir: Path | None = None) -> None:
        config = load_config()
        self.model_dir = model_dir or config.hidden_model_dir
        self.providers = config.hidden_providers
        self.encoder_path = self.model_dir / "encoder.onnx"
        self.decoder_path = self.model_dir / "decoder.onnx"
        self._encoder_session: Any | None = None
        self._decoder_session: Any | None = None
        self._session_lock = threading.Lock()

    def encode(
        self,
        image: ImageArray,
        message: str | None = None,
        *,
        payload_bits: Sequence[int] | None = None,
        strength: float = 1.0,
    ) -> ImageArray:
        self._ensure_available()
        del strength

        image_tensor = self._prepare_image(image, session_kind="encoder")
        message_bits = self._message_bits
        resolved_bits = resolve_payload_bits(message, payload_bits)
        if len(resolved_bits) > message_bits:
            raise MessageTooLongError(
                "message exceeds the hidden engine bit capacity",
                hint=f"Use a shorter message or export a hidden model with at least {len(resolved_bits)} bits.",
            )
        message_tensor = np.zeros((1, message_bits), dtype=np.float32)
        message_tensor[0, : len(resolved_bits)] = np.asarray(resolved_bits, dtype=np.float32)

        outputs = self._encoder.run(
            None,
            {
                self._encoder.get_inputs()[0].name: image_tensor,
                self._encoder.get_inputs()[1].name: message_tensor,
            },
        )
        encoded = np.asarray(outputs[0], dtype=np.float32)
        return self._restore_image(
            encoded,
            original_shape=(int(image.shape[0]), int(image.shape[1])),
        )

    def decode(self, image: ImageArray) -> ExtractResult:
        self._ensure_available()

        image_tensor = self._prepare_image(image, session_kind="decoder")
        outputs = self._decoder.run(
            None,
            {self._decoder.get_inputs()[0].name: image_tensor},
        )
        logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        bits = tuple(int(value >= 0.0) for value in logits.tolist())
        decoded = decode_bitstream(bits)
        return ExtractResult(
            found=decoded.valid,
            engine=self.name,
            bits=decoded.bits,
            payload=decoded.payload,
            message=decoded.message,
            confidence=1.0 if decoded.valid else 0.0,
            error=decoded.error,
        )

    def _ensure_available(self) -> None:
        missing = [path for path in (self.encoder_path, self.decoder_path) if not path.exists()]
        if missing:
            missing_names = ", ".join(path.name for path in missing)
            raise EngineUnavailableError(
                f"hidden engine weights are missing: {missing_names}",
                hint=f"Place encoder.onnx and decoder.onnx under {self.model_dir}.",
            )
        _ = self._encoder
        _ = self._decoder

    @property
    def _encoder(self) -> Any:
        if self._encoder_session is None:
            with self._session_lock:
                if self._encoder_session is None:
                    self._encoder_session = self._create_session(self.encoder_path)
        return self._encoder_session

    @property
    def _decoder(self) -> Any:
        if self._decoder_session is None:
            with self._session_lock:
                if self._decoder_session is None:
                    self._decoder_session = self._create_session(self.decoder_path)
        return self._decoder_session

    @property
    def _message_bits(self) -> int:
        shape = self._encoder.get_inputs()[1].shape
        bits = shape[-1]
        if not isinstance(bits, int):
            raise EngineUnavailableError(
                "hidden encoder message input shape is not statically defined",
                hint="Export hidden ONNX models with a fixed message bit dimension.",
            )
        return bits

    def _create_session(self, path: Path) -> Any:
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError as exc:
            raise EngineUnavailableError(
                "onnxruntime is required for the hidden backend",
                hint="Install the hidden runtime dependencies, for example `pip install -e .[hidden]`.",
            ) from exc
        available_providers = ()
        if hasattr(ort, "get_available_providers"):
            available_providers = tuple(str(provider) for provider in ort.get_available_providers())
        providers = list(resolve_hidden_execution_providers(self.providers, available_providers))
        try:
            return ort.InferenceSession(str(path), providers=providers)
        except Exception:
            fallback_providers = list(
                resolve_hidden_execution_providers(("CPUExecutionProvider",), available_providers)
            )
            if providers == fallback_providers:
                raise
            return ort.InferenceSession(str(path), providers=fallback_providers)

    def _prepare_image(self, image: ImageArray, *, session_kind: str) -> np.ndarray[Any, Any]:
        input_meta = self._encoder.get_inputs()[0] if session_kind == "encoder" else self._decoder.get_inputs()[0]
        height, width = self._spatial_shape(input_meta.shape)
        pil_image = Image.fromarray(image, mode="RGB").resize((width, height), Image.Resampling.BILINEAR)
        array = np.asarray(pil_image, dtype=np.float32) / 255.0
        chw = np.transpose(array, (2, 0, 1))[None, ...]
        return chw.astype(np.float32, copy=False)

    def _restore_image(
        self, encoded: np.ndarray[Any, Any], *, original_shape: tuple[int, int]
    ) -> ImageArray:
        chw = np.asarray(encoded[0], dtype=np.float32)
        hwc = np.transpose(chw, (1, 2, 0))
        clipped = np.clip(hwc, 0.0, 1.0)
        scaled = (clipped * 255.0).round().astype(np.uint8)
        height, width = original_shape
        if scaled.shape[:2] == (height, width):
            return scaled
        resized = Image.fromarray(scaled, mode="RGB").resize((width, height), Image.Resampling.BILINEAR)
        return cast(ImageArray, np.asarray(resized, dtype=np.uint8))

    def _spatial_shape(self, shape: list[Any]) -> tuple[int, int]:
        height = shape[-2]
        width = shape[-1]
        if not isinstance(height, int) or not isinstance(width, int):
            raise EngineUnavailableError(
                "hidden ONNX image input shape is not statically defined",
                hint="Export hidden ONNX models with fixed image dimensions.",
            )
        return height, width
