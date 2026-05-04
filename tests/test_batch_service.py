from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from stegmark.service import (
    embed_directory,
    extract_directory,
    info_directory,
    verify_directory,
)


def test_embed_directory_processes_multiple_files(
    sample_image_path: Path, tmp_path: Path
) -> None:
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    for name in ("a.png", "b.png"):
        (source_dir / name).write_bytes(sample_image_path.read_bytes())

    result = embed_directory(source_dir, message="Alice 2026", engine="native")

    assert result.total == 2
    assert result.succeeded == 2
    assert result.failed == 0
    assert (source_dir / "a_wm.png").exists()
    assert (source_dir / "b_wm.png").exists()


def test_extract_directory_supports_recursive(
    sample_image_path: Path, tmp_path: Path
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    watermarked = nested / "sample_wm.png"
    from stegmark.service import embed_file

    embed_file(sample_image_path, watermarked, message="Alice 2026", engine="native")

    result = extract_directory(tmp_path, engine="native", recursive=True)

    assert result.total == 2
    assert result.succeeded == 1
    assert any(item.result and item.result.message == "Alice 2026" for item in result.items)


def test_verify_directory_reports_matches(sample_image_path: Path, tmp_path: Path) -> None:
    watermarked = tmp_path / "sample_wm.png"
    from stegmark.service import embed_file

    embed_file(sample_image_path, watermarked, message="Alice 2026", engine="native")

    result = verify_directory(tmp_path, expected="Alice 2026", engine="native")

    assert result.total == 2
    assert result.succeeded == 1
    assert any(item.result and item.result.matched is True for item in result.items)


def test_info_directory_reports_watermark_status(sample_image_path: Path, tmp_path: Path) -> None:
    watermarked = tmp_path / "sample_wm.png"
    from stegmark.service import embed_file

    embed_file(sample_image_path, watermarked, message="Alice 2026", engine="native")

    result = info_directory(tmp_path, engine="native")

    assert result.total == 2
    assert result.succeeded == 1
    assert any(item.result and item.result.found is True for item in result.items)


def test_embed_directory_caps_workers_to_safe_limit(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    for name in ("a.png", "b.png"):
        (source_dir / name).write_bytes(sample_image_path.read_bytes())

    captured: dict[str, int] = {}

    class DummyExecutor:
        def __init__(self, *, max_workers: int) -> None:
            captured["max_workers"] = max_workers

        def __enter__(self) -> DummyExecutor:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def map(self, worker, paths):
            return [worker(path) for path in paths]

    def fake_embed_file(*args, **kwargs) -> SimpleNamespace:
        return SimpleNamespace(output_path=kwargs.get("output_path"))

    monkeypatch.setattr("stegmark.service.ThreadPoolExecutor", DummyExecutor)
    monkeypatch.setattr("stegmark.service.ProcessPoolExecutor", DummyExecutor)
    monkeypatch.setattr("stegmark.service.embed_file", fake_embed_file)

    result = embed_directory(source_dir, message="Alice 2026", workers=99_999, engine="hidden")

    assert result.total == 2
    assert captured["max_workers"] == 32


def test_embed_directory_rejects_output_paths_outside_target_dir(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "input"
    output_dir = tmp_path / "allowed"
    source_dir.mkdir()
    output_dir.mkdir()
    (source_dir / "a.png").write_bytes(sample_image_path.read_bytes())
    escaped_output = tmp_path / "escaped.png"

    monkeypatch.setattr(
        "stegmark.service._default_output_path",
        lambda path, *, output_format=None: escaped_output,
    )

    result = embed_directory(
        source_dir,
        message="Alice 2026",
        output_dir=output_dir,
    )

    assert result.failed == 1
    assert "output path escapes allowed directory" in (result.items[0].error or "")


def test_embed_directory_reports_progress(
    sample_image_path: Path,
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    for name in ("a.png", "b.png"):
        (source_dir / name).write_bytes(sample_image_path.read_bytes())

    calls: list[tuple[int, int, str]] = []

    def on_progress(index: int, total: int, path: Path) -> None:
        calls.append((index, total, path.name))

    result = embed_directory(
        source_dir,
        message="Alice 2026",
        engine="native",
        progress=on_progress,
    )

    assert result.total == 2
    assert calls == [
        (1, 2, "a.png"),
        (2, 2, "b.png"),
    ]


def test_embed_directory_releases_embedded_image_arrays_after_write(
    sample_image_path: Path,
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    (source_dir / "a.png").write_bytes(sample_image_path.read_bytes())

    result = embed_directory(source_dir, message="Alice 2026", engine="native")

    embed_result = result.items[0].result
    assert embed_result is not None
    assert embed_result.image.size == 0
