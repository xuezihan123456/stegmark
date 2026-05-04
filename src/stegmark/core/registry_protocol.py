from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stegmark.types import ImageArray

DEFAULT_REGISTRY_DB = Path.home() / ".stegmark" / "registry.db"


@dataclass(frozen=True)
class WatermarkRegistration:
    """水印注册记录。"""

    watermark_id: str
    image_hash: str
    message_hash: str
    timestamp: str
    engine: str
    extra: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WatermarkRegistration:
        return cls(
            watermark_id=d["watermark_id"],
            image_hash=d["image_hash"],
            message_hash=d["message_hash"],
            timestamp=d["timestamp"],
            engine=d["engine"],
            extra=d.get("extra", {}),
        )


def compute_image_hash(image: ImageArray) -> str:
    """计算图像 SHA256 前 16 位 hex。"""
    return hashlib.sha256(image.tobytes()).hexdigest()[:16]


def compute_message_hash(message: str) -> str:
    """计算消息 SHA256 前 16 位 hex。"""
    return hashlib.sha256(message.encode("utf-8")).hexdigest()[:16]


def generate_watermark_id(image_hash: str, message_hash: str, timestamp: str) -> str:
    """生成唯一水印 ID = SHA256(image_hash + message_hash + timestamp)[:16]。"""
    combined = f"{image_hash}:{message_hash}:{timestamp}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


class LocalRegistry:
    """本地 SQLite 水印注册表。"""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DEFAULT_REGISTRY_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS registrations (
                    watermark_id TEXT PRIMARY KEY,
                    image_hash TEXT NOT NULL,
                    message_hash TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    engine TEXT NOT NULL,
                    extra_json TEXT NOT NULL DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_image_hash
                ON registrations(image_hash)
            """)
            conn.commit()

    def register(self, registration: WatermarkRegistration) -> None:
        """插入一条注册记录。"""
        with self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO registrations
                   (watermark_id, image_hash, message_hash, timestamp, engine, extra_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    registration.watermark_id,
                    registration.image_hash,
                    registration.message_hash,
                    registration.timestamp,
                    registration.engine,
                    json.dumps(registration.extra, ensure_ascii=False),
                ),
            )
            conn.commit()

    def lookup_by_image(self, image_hash: str) -> list[WatermarkRegistration]:
        """根据图像哈希查询所有关联注册记录。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM registrations WHERE image_hash = ? ORDER BY timestamp DESC",
                (image_hash,),
            ).fetchall()
        return [_row_to_registration(row) for row in rows]

    def lookup_by_id(self, watermark_id: str) -> WatermarkRegistration | None:
        """精确查询单条注册记录。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM registrations WHERE watermark_id = ?",
                (watermark_id,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_registration(row)

    def list_all(self, limit: int = 100) -> list[WatermarkRegistration]:
        """列出最近注册记录。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM registrations ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_registration(row) for row in rows]

    def export_json(self, output_path: Path) -> None:
        """导出全部注册记录为 JSON 文件。"""
        records = [reg.to_dict() for reg in self.list_all(limit=999999)]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _row_to_registration(row: sqlite3.Row) -> WatermarkRegistration:
    extra = {}
    try:
        extra = json.loads(row["extra_json"])
    except (json.JSONDecodeError, KeyError):
        pass
    return WatermarkRegistration(
        watermark_id=row["watermark_id"],
        image_hash=row["image_hash"],
        message_hash=row["message_hash"],
        timestamp=row["timestamp"],
        engine=row["engine"],
        extra=extra,
    )


__all__ = [
    "WatermarkRegistration",
    "LocalRegistry",
    "compute_image_hash",
    "compute_message_hash",
    "generate_watermark_id",
    "DEFAULT_REGISTRY_DB",
]
