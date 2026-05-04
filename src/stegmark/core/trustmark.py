from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

from stegmark.core.engine import EngineCapabilities, WatermarkEngine
from stegmark.exceptions import EngineUnavailableError, InvalidInputError
from stegmark.types import ExtractResult, ImageArray


class TrustMarkEngine(WatermarkEngine):
    name = "trustmark"
    declared_capabilities = EngineCapabilities(
        supports_text_messages=True,
        supports_payload_bits=False,
        supports_strength_control=False,
        requires_optional_dependency=True,
        requires_model_files=False,
    )

    def __init__(self) -> None:
        try:
            from trustmark import TrustMark  # type: ignore[import-not-found]
        except ImportError as exc:
            raise EngineUnavailableError(
                "trustmark integration is not installed",
                hint="Install the optional trustmark package to enable this backend.",
            ) from exc

        self._encoder = TrustMark(verbose=False, model_type="Q")
        self._max_ascii_chars = self._encoder.schemaCapacity() // 7

    def encode(
        self,
        image: ImageArray,
        message: str | None = None,
        *,
        payload_bits: Sequence[int] | None = None,
        strength: float = 1.0,
    ) -> ImageArray:
        del strength
        if payload_bits is not None:
            raise InvalidInputError(
                "bits mode is not supported by the trustmark backend",
                hint="Use the native or hidden backend for explicit bits payloads.",
            )
        if message is None:
            raise InvalidInputError(
                "trustmark encode requires a text message",
                hint="Provide a text message when using the trustmark backend.",
            )
        try:
            message.encode("ascii")
        except UnicodeEncodeError as exc:
            raise InvalidInputError(
                "trustmark only supports ASCII text messages",
                hint="Use ASCII text only, or switch to the native/hidden backend for arbitrary payloads.",
            ) from exc
        if len(message) > self._max_ascii_chars:
            raise InvalidInputError(
                f"trustmark supports at most {self._max_ascii_chars} ASCII characters in text mode",
                hint="Shorten the message or switch to the native/hidden backend for larger payloads.",
            )
        encoded = self._encoder.encode(Image.fromarray(image, mode="RGB"), message)
        return np.asarray(encoded, dtype=np.uint8).copy()

    def decode(self, image: ImageArray) -> ExtractResult:
        secret, present, schema = self._encoder.decode(Image.fromarray(image, mode="RGB"))
        message = str(secret) if present else None
        return ExtractResult(
            found=bool(present),
            engine=self.name,
            message=message,
            confidence=1.0 if present else 0.0,
            error=None if present else f"schema:{schema}",
        )
