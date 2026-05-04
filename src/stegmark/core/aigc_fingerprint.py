from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from stegmark.core.engine import WatermarkEngine
from stegmark.types import ImageArray

AIGC_FRAME_TYPE: int = 0x03  # 帧协议中的 AIGC_METADATA 类型标记


@dataclass(frozen=True)
class AIGCMetadata:
    """AI 生成内容的溯源元数据，兼容 C2PA 清单格式。"""

    generator: str
    model_version: str | None = None
    seed: int | None = None
    prompt_hash: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    custom: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        d: dict[str, Any] = {"generator": self.generator}
        if self.model_version:
            d["model_version"] = self.model_version
        if self.seed is not None:
            d["seed"] = self.seed
        if self.prompt_hash:
            d["prompt_hash"] = self.prompt_hash
        d["timestamp"] = self.timestamp
        if self.custom:
            d["custom"] = self.custom
        return json.dumps(d, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, json_str: str) -> AIGCMetadata:
        d = json.loads(json_str)
        return cls(
            generator=d["generator"],
            model_version=d.get("model_version"),
            seed=d.get("seed"),
            prompt_hash=d.get("prompt_hash"),
            timestamp=d.get("timestamp", datetime.now(timezone.utc).isoformat()),
            custom=d.get("custom", {}),
        )

    def to_c2pa_manifest(self) -> dict[str, Any]:
        """生成 C2PA 兼容的清单字典。"""
        assertion: dict[str, Any] = {
            "label": "stegmark.aigc",
            "data": {
                "generator": self.generator,
                "timestamp": self.timestamp,
            },
        }
        if self.model_version:
            assertion["data"]["model_version"] = self.model_version
        if self.seed is not None:
            assertion["data"]["seed"] = self.seed
        if self.prompt_hash:
            assertion["data"]["prompt_hash"] = self.prompt_hash
        if self.custom:
            assertion["data"]["custom"] = self.custom
        return {
            "claim_generator": "StegMark-AIGC/1.0",
            "assertions": [assertion],
        }


def compute_prompt_hash(prompt: str) -> str:
    """计算 prompt 字符串的 SHA256 前 8 位 hex。"""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]


def stamp_image(
    image: ImageArray,
    engine: WatermarkEngine,
    metadata: AIGCMetadata,
    *,
    strength: float = 1.0,
) -> ImageArray:
    """将 AIGC 元数据嵌入图像。

    将 AIGCMetadata 序列化为 JSON，然后作为消息嵌入。
    """
    message = metadata.to_json()
    return engine.encode(image, message, strength=strength)


def extract_aigc_metadata(
    image: ImageArray,
    engine: WatermarkEngine,
) -> AIGCMetadata | None:
    """从水印图像中提取 AIGC 元数据。

    返回 AIGCMetadata 或 None（提取失败时）。
    """
    result = engine.decode(image)
    if not result.found or not result.message:
        return None
    try:
        return AIGCMetadata.from_json(result.message)
    except (json.JSONDecodeError, KeyError):
        return None


__all__ = [
    "AIGCMetadata",
    "stamp_image",
    "extract_aigc_metadata",
    "compute_prompt_hash",
    "AIGC_FRAME_TYPE",
]
