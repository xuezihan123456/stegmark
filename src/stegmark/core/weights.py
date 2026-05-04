from __future__ import annotations

import importlib.util
import platform
import sys
from collections.abc import Sequence
from typing import Any

from stegmark.config import load_config


def available_engines() -> dict[str, bool]:
    config = load_config()
    hidden_dir = config.hidden_model_dir
    hidden_ready = (hidden_dir / "encoder.onnx").exists() and (hidden_dir / "decoder.onnx").exists()
    trustmark_ready = _module_is_available("trustmark")
    onnxruntime_ready = _module_is_available("onnxruntime")
    return {
        "native": True,
        "hidden": hidden_ready and onnxruntime_ready,
        "trustmark": trustmark_ready,
    }


def is_available(engine: str) -> bool:
    return available_engines().get(engine.lower(), False)


def device_info() -> dict[str, object]:
    config = load_config()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "engines": available_engines(),
        "model_dir": str(config.model_dir),
        "hidden_model_dir": str(config.hidden_model_dir),
        "hidden_providers": list(config.hidden_providers),
        "hidden": hidden_runtime_info(config.hidden_model_dir, config.hidden_providers),
    }


def hidden_runtime_info(model_dir, configured_providers: Sequence[str]) -> dict[str, object]:
    ort = _load_onnxruntime()
    available_providers = _available_hidden_providers(ort)
    preferred_providers = list(resolve_hidden_execution_providers(configured_providers, available_providers))
    return {
        "model_dir": str(model_dir),
        "available_providers": list(available_providers),
        "preferred_providers": preferred_providers,
    }


def resolve_hidden_execution_providers(
    configured_providers: Sequence[str],
    available_providers: Sequence[str] | None = None,
) -> tuple[str, ...]:
    configured = [provider for provider in configured_providers if provider]
    available = tuple(available_providers or ())
    if available:
        selected = [provider for provider in configured if provider in available]
    else:
        selected = list(configured)
    if "CPUExecutionProvider" not in selected and (not available or "CPUExecutionProvider" in available):
        selected.append("CPUExecutionProvider")
    if not selected:
        return ("CPUExecutionProvider",)
    return tuple(dict.fromkeys(selected))


def _module_is_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ValueError:
        return name in sys.modules


def _load_onnxruntime() -> Any | None:
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError:
        return None
    return ort


def _available_hidden_providers(ort_module: Any | None) -> tuple[str, ...]:
    if ort_module is None or not hasattr(ort_module, "get_available_providers"):
        return ()
    providers = ort_module.get_available_providers()
    return tuple(str(provider) for provider in providers)
