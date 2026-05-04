from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from stegmark.evaluation.benchmark import benchmark_file


class _FakeBackend:
    name = "fake"

    def encode(self, array: object, message: str) -> str:
        return f"encoded::{message}"

    def decode(self, attacked: str) -> SimpleNamespace:
        return SimpleNamespace(
            message="Alice 2026",
            found=True,
            bits=(1, 0, 1),
            error=None,
        )


def _patch_benchmark_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "stegmark.evaluation.benchmark.load_image",
        lambda path: SimpleNamespace(array="input-image"),
    )
    monkeypatch.setattr(
        "stegmark.evaluation.benchmark.get_engine",
        lambda engine: _FakeBackend(),
    )
    monkeypatch.setattr(
        "stegmark.evaluation.benchmark.encode_text",
        lambda message: (1, 0, 1),
    )


def test_benchmark_file_uses_metrics_compute_psnr(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_benchmark_dependencies(monkeypatch)
    psnr_calls: list[tuple[object, object]] = []

    monkeypatch.setattr(
        "stegmark.evaluation.benchmark.apply_attack",
        lambda image, attack_name, seed=0: f"attacked::{attack_name}",
    )
    monkeypatch.setattr(
        "stegmark.evaluation.benchmark.compute_psnr",
        lambda reference, candidate: psnr_calls.append((reference, candidate)) or 27.5,
        raising=False,
    )

    result = benchmark_file(
        "unused.png",
        message="Alice 2026",
        attacks=["jpeg_q90"],
    )

    assert result.attack_results["jpeg_q90"].psnr == 27.5
    assert psnr_calls == [("encoded::Alice 2026", "attacked::jpeg_q90")]


def test_benchmark_file_runs_attacks_in_parallel_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_benchmark_dependencies(monkeypatch)
    running = 0
    max_running = 0
    lock = threading.Lock()
    delays = {
        "slow": 0.06,
        "fast": 0.01,
        "medium": 0.03,
    }

    def fake_apply_attack(image: object, attack_name: str, seed: int | None = 0) -> str:
        nonlocal running, max_running
        with lock:
            running += 1
            max_running = max(max_running, running)
        time.sleep(delays[attack_name])
        with lock:
            running -= 1
        return attack_name

    monkeypatch.setattr("stegmark.evaluation.benchmark.apply_attack", fake_apply_attack)
    monkeypatch.setattr(
        "stegmark.evaluation.benchmark.compute_psnr",
        lambda reference, candidate: 42.0,
        raising=False,
    )

    result = benchmark_file(
        "unused.png",
        message="Alice 2026",
        attacks=["slow", "fast", "medium"],
    )

    assert list(result.attack_results) == ["slow", "fast", "medium"]
    assert max_running >= 2


def test_benchmark_file_parallel_errors_follow_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_benchmark_dependencies(monkeypatch)

    def fake_apply_attack(image: object, attack_name: str, seed: int | None = 0) -> str:
        if attack_name == "ordered_failure":
            time.sleep(0.04)
            raise RuntimeError("ordered failure")
        if attack_name == "later_failure":
            time.sleep(0.01)
            raise RuntimeError("later failure")
        return attack_name

    monkeypatch.setattr("stegmark.evaluation.benchmark.apply_attack", fake_apply_attack)
    monkeypatch.setattr(
        "stegmark.evaluation.benchmark.compute_psnr",
        lambda reference, candidate: 42.0,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="ordered failure"):
        benchmark_file(
            "unused.png",
            message="Alice 2026",
            attacks=["ok", "ordered_failure", "later_failure"],
        )
