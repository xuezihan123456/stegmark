from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import ClassVar

from stegmark.types import ExtractResult, ImageArray


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    supports_text_messages: bool = True
    supports_payload_bits: bool = True
    supports_strength_control: bool = True
    requires_optional_dependency: bool = False
    requires_model_files: bool = False

    def as_dict(self) -> dict[str, bool]:
        return dict(asdict(self))


_BUILTIN_CAPABILITIES: dict[str, EngineCapabilities] = {
    "native": EngineCapabilities(),
    "hidden": EngineCapabilities(
        supports_strength_control=False,
        requires_optional_dependency=True,
        requires_model_files=True,
    ),
    "trustmark": EngineCapabilities(
        supports_payload_bits=False,
        supports_strength_control=False,
        requires_optional_dependency=True,
    ),
}


class WatermarkEngine(ABC):
    name: str
    declared_capabilities: ClassVar[EngineCapabilities | None] = None

    @classmethod
    def describe_capabilities(cls) -> EngineCapabilities:
        declared = cls.declared_capabilities
        if declared is not None:
            return declared
        return _BUILTIN_CAPABILITIES.get(getattr(cls, "name", ""), EngineCapabilities())

    @property
    def capabilities(self) -> EngineCapabilities:
        return type(self).describe_capabilities()

    @property
    def supports_text_messages(self) -> bool:
        return self.capabilities.supports_text_messages

    @property
    def supports_payload_bits(self) -> bool:
        return self.capabilities.supports_payload_bits

    @property
    def supports_strength_control(self) -> bool:
        return self.capabilities.supports_strength_control

    @property
    def requires_optional_dependency(self) -> bool:
        return self.capabilities.requires_optional_dependency

    @property
    def requires_model_files(self) -> bool:
        return self.capabilities.requires_model_files

    @abstractmethod
    def encode(
        self,
        image: ImageArray,
        message: str | None = None,
        *,
        payload_bits: Sequence[int] | None = None,
        strength: float = 1.0,
    ) -> ImageArray:
        raise NotImplementedError

    @abstractmethod
    def decode(self, image: ImageArray) -> ExtractResult:
        raise NotImplementedError
