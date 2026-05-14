"""Resumable batch-processing manifest backed by SQLite.

Records the per-input-file state of long-running batch operations so
that aborted runs can be restarted with ``--resume`` without re-processing
files that already completed successfully.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Literal

ManifestStatus = Literal["pending", "in_progress", "done", "failed"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    input_path TEXT PRIMARY KEY,
    input_sha256 TEXT NOT NULL,
    output_path TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    started_at REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_status ON tasks(status);
"""


@dataclass(frozen=True)
class ManifestRow:
    input_path: str
    input_sha256: str
    output_path: str
    status: ManifestStatus
    error: str | None
    started_at: float | None
    finished_at: float | None


class BatchManifest:
    """Thread-safe SQLite-backed batch manifest."""

    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def __enter__(self) -> BatchManifest:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()

    def register(self, input_path: Path, output_path: Path) -> Literal["process", "skip"]:
        """Register an input file. Returns ``'skip'`` if it is already done."""
        digest = _file_sha256(input_path)
        key = str(input_path.resolve())
        with self._lock:
            existing = self._conn.execute(
                "SELECT status, input_sha256 FROM tasks WHERE input_path = ?", (key,)
            ).fetchone()
            if existing and existing["status"] == "done" and existing["input_sha256"] == digest:
                return "skip"
            self._conn.execute(
                """
                INSERT INTO tasks(input_path, input_sha256, output_path, status, started_at, finished_at, error)
                VALUES (?, ?, ?, 'pending', NULL, NULL, NULL)
                ON CONFLICT(input_path) DO UPDATE SET
                    input_sha256 = excluded.input_sha256,
                    output_path = excluded.output_path,
                    status = 'pending',
                    started_at = NULL,
                    finished_at = NULL,
                    error = NULL
                """,
                (key, digest, str(output_path.resolve())),
            )
            self._conn.commit()
        return "process"

    def mark_started(self, input_path: Path) -> None:
        key = str(input_path.resolve())
        with self._lock:
            self._conn.execute(
                "UPDATE tasks SET status='in_progress', started_at=? WHERE input_path=?",
                (time.time(), key),
            )
            self._conn.commit()

    def mark_done(self, input_path: Path, output_path: Path) -> None:
        key = str(input_path.resolve())
        with self._lock:
            self._conn.execute(
                "UPDATE tasks SET status='done', output_path=?, finished_at=?, error=NULL WHERE input_path=?",
                (str(output_path.resolve()), time.time(), key),
            )
            self._conn.commit()

    def mark_failed(self, input_path: Path, error: str) -> None:
        key = str(input_path.resolve())
        with self._lock:
            self._conn.execute(
                "UPDATE tasks SET status='failed', error=?, finished_at=? WHERE input_path=?",
                (error[:1024], time.time(), key),
            )
            self._conn.commit()

    def pending_tasks(self) -> list[ManifestRow]:
        return self._query_rows("status IN ('pending', 'in_progress')")

    def all_rows(self) -> list[ManifestRow]:
        return self._query_rows("1=1")

    def summary(self) -> dict[str, int]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM tasks GROUP BY status"
            )
            base = {"pending": 0, "in_progress": 0, "done": 0, "failed": 0}
            for row in cursor:
                base[str(row["status"])] = int(row["cnt"])
        return base

    def iter_pending(self) -> Iterator[ManifestRow]:
        yield from self.pending_tasks()

    def _query_rows(self, where: str) -> list[ManifestRow]:
        with self._lock:
            cursor = self._conn.execute(
                f"SELECT input_path, input_sha256, output_path, status, error, started_at, finished_at "
                f"FROM tasks WHERE {where} ORDER BY input_path"
            )
            return [
                ManifestRow(
                    input_path=str(r["input_path"]),
                    input_sha256=str(r["input_sha256"]),
                    output_path=str(r["output_path"]),
                    status=str(r["status"]),  # type: ignore[arg-type]
                    error=r["error"],
                    started_at=r["started_at"],
                    finished_at=r["finished_at"],
                )
                for r in cursor
            ]


def _file_sha256(path: Path, chunk_size: int = 65536) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["BatchManifest", "ManifestRow", "ManifestStatus"]
