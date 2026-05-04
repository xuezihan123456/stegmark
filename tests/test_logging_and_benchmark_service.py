from __future__ import annotations

import logging
from pathlib import Path

import stegmark


def test_package_exposes_named_logger() -> None:
    assert stegmark.logger.name == "stegmark"


def test_embed_directory_logs_failures(sample_image_path: Path, tmp_path: Path, caplog) -> None:
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    (source_dir / "a.png").write_bytes(sample_image_path.read_bytes())

    existing = source_dir / "a_wm.png"
    existing.write_bytes(sample_image_path.read_bytes())

    with caplog.at_level(logging.WARNING, logger="stegmark"):
        result = stegmark.embed_directory(source_dir, message="Alice 2026", engine="native")

    assert result.failed == 1
    assert any("output exists" in message for message in caplog.messages)


def test_api_benchmark_round_trips_through_service(sample_image_path: Path, tmp_path: Path) -> None:
    sentinel = object()

    import stegmark.service as service

    original = service.benchmark_file_service

    def fake_benchmark_file_service(*args, **kwargs):
        return sentinel

    service.benchmark_file_service = fake_benchmark_file_service
    try:
        result = stegmark.benchmark(
            sample_image_path,
            "Alice 2026",
            engine="native",
            output_dir=tmp_path,
            report_format="json",
        )
    finally:
        service.benchmark_file_service = original

    assert result is sentinel
