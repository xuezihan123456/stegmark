"""Tests for the resumable batch manifest."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from stegmark.core.batch_manifest import BatchManifest


@pytest.fixture()
def manifest_path(tmp_path: Path) -> Path:
    return tmp_path / "manifest.db"


def _make_input(tmp_path: Path, name: str, content: bytes = b"hello") -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


def test_register_new_file_returns_process(manifest_path: Path, tmp_path: Path):
    src = _make_input(tmp_path, "a.png")
    dst = tmp_path / "out" / "a.png"

    with BatchManifest(manifest_path) as m:
        result = m.register(src, dst)

    assert result == "process"


def test_register_done_with_same_hash_returns_skip(manifest_path: Path, tmp_path: Path):
    src = _make_input(tmp_path, "a.png")
    dst = tmp_path / "out" / "a.png"
    with BatchManifest(manifest_path) as m:
        m.register(src, dst)
        m.mark_done(src, dst)

    with BatchManifest(manifest_path) as m:
        result = m.register(src, dst)

    assert result == "skip"


def test_register_changed_file_returns_process(manifest_path: Path, tmp_path: Path):
    src = _make_input(tmp_path, "a.png", b"v1")
    dst = tmp_path / "out" / "a.png"
    with BatchManifest(manifest_path) as m:
        m.register(src, dst)
        m.mark_done(src, dst)

    src.write_bytes(b"v2-different")

    with BatchManifest(manifest_path) as m:
        result = m.register(src, dst)

    assert result == "process"


def test_summary_reflects_state_transitions(manifest_path: Path, tmp_path: Path):
    files = [_make_input(tmp_path, f"f{i}.png", f"c{i}".encode()) for i in range(3)]
    out_dir = tmp_path / "out"

    with BatchManifest(manifest_path) as m:
        for src in files:
            m.register(src, out_dir / src.name)
        m.mark_done(files[0], out_dir / files[0].name)
        m.mark_failed(files[1], "boom")
        summary = m.summary()

    assert summary["done"] == 1
    assert summary["failed"] == 1
    assert summary["pending"] == 1


def test_mark_failed_records_error(manifest_path: Path, tmp_path: Path):
    src = _make_input(tmp_path, "a.png")
    dst = tmp_path / "out" / "a.png"
    with BatchManifest(manifest_path) as m:
        m.register(src, dst)
        m.mark_failed(src, "disk full")

        rows = m.all_rows()

    assert len(rows) == 1
    assert rows[0].status == "failed"
    assert rows[0].error == "disk full"


def test_pending_tasks_excludes_done(manifest_path: Path, tmp_path: Path):
    src1 = _make_input(tmp_path, "a.png")
    src2 = _make_input(tmp_path, "b.png", b"different")
    out_dir = tmp_path / "out"
    with BatchManifest(manifest_path) as m:
        m.register(src1, out_dir / src1.name)
        m.register(src2, out_dir / src2.name)
        m.mark_done(src1, out_dir / src1.name)

        pending = m.pending_tasks()

    assert len(pending) == 1
    assert pending[0].input_path.endswith("b.png")


def test_persisted_db_can_be_reopened(manifest_path: Path, tmp_path: Path):
    src = _make_input(tmp_path, "a.png")
    dst = tmp_path / "out" / "a.png"
    with BatchManifest(manifest_path) as m:
        m.register(src, dst)
        m.mark_done(src, dst)

    with BatchManifest(manifest_path) as m:
        summary = m.summary()

    assert summary["done"] == 1


def test_concurrent_register_is_thread_safe(manifest_path: Path, tmp_path: Path):
    files = [_make_input(tmp_path, f"f{i}.png", f"c{i}".encode()) for i in range(20)]
    out_dir = tmp_path / "out"

    with BatchManifest(manifest_path) as m:
        def worker(src: Path):
            m.register(src, out_dir / src.name)

        threads = [threading.Thread(target=worker, args=(src,)) for src in files]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        summary = m.summary()

    assert summary["pending"] == 20
