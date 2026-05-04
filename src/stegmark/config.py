from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import tomli as tomllib

from stegmark.exceptions import ConfigError, InvalidInputError

DEFAULT_HIDDEN_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")
_MISSING = object()


@dataclass(frozen=True)
class HiddenEngineConfig:
    model_dir: Path
    providers: tuple[str, ...] = DEFAULT_HIDDEN_PROVIDERS

    def to_dict(self) -> dict[str, object]:
        return {
            "model_dir": str(self.model_dir),
            "providers": list(self.providers),
        }


@dataclass(frozen=True)
class EngineConfig:
    hidden: HiddenEngineConfig

    def to_dict(self) -> dict[str, object]:
        return {"hidden": self.hidden.to_dict()}


@dataclass(frozen=True)
class StegMarkConfig:
    engine: str = "auto"
    strength: float = 1.0
    workers: int = 4
    min_image_size: int = 128
    model_dir: Path = Path.home() / ".stegmark" / "models"
    engines: EngineConfig = field(
        default_factory=lambda: EngineConfig(
            hidden=HiddenEngineConfig(Path.home() / ".stegmark" / "models" / "hidden")
        )
    )

    @property
    def hidden_model_dir(self) -> Path:
        return self.engines.hidden.model_dir

    @property
    def hidden(self) -> HiddenEngineConfig:
        return self.engines.hidden

    @property
    def hidden_providers(self) -> tuple[str, ...]:
        return self.engines.hidden.providers

    def to_dict(self) -> dict[str, object]:
        return {
            "engine": self.engine,
            "strength": self.strength,
            "workers": self.workers,
            "min_image_size": self.min_image_size,
            "model_dir": str(self.model_dir),
            "hidden_model_dir": str(self.hidden_model_dir),
            "hidden_providers": list(self.hidden_providers),
            "hidden": self.hidden.to_dict(),
            "engines": self.engines.to_dict(),
        }


CONFIG_KEYS = {
    "engine": str,
    "strength": float,
    "workers": int,
    "min_image_size": int,
    "model_dir": Path,
    "engines.hidden.model_dir": Path,
    "engines.hidden.providers": tuple,
}
CONFIG_KEY_ALIASES = {
    "hidden.model_dir": "engines.hidden.model_dir",
    "hidden.providers": "engines.hidden.providers",
    "hidden_model_dir": "engines.hidden.model_dir",
    "hidden_providers": "engines.hidden.providers",
}


def default_config_path() -> Path:
    configured = os.getenv("STEGMARK_CONFIG")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".stegmark" / "config.toml"


def load_config(*, config_path: Path | None = None) -> StegMarkConfig:
    path = config_path or default_config_path()
    defaults = StegMarkConfig()
    values: dict[str, object] = {
        "engine": defaults.engine,
        "strength": defaults.strength,
        "workers": defaults.workers,
        "min_image_size": defaults.min_image_size,
        "model_dir": defaults.model_dir,
        "engines.hidden.model_dir": _MISSING,
        "engines.hidden.providers": defaults.hidden_providers,
    }
    if path.exists():
        values.update(_read_config_file(path))
    values.update(_environment_overrides())
    model_dir = cast(Path, values["model_dir"])
    hidden_model_dir_value = values["engines.hidden.model_dir"]
    hidden_model_dir = model_dir / "hidden" if hidden_model_dir_value is _MISSING else cast(Path, hidden_model_dir_value)
    return StegMarkConfig(
        engine=cast(str, values["engine"]),
        strength=cast(float, values["strength"]),
        workers=cast(int, values["workers"]),
        min_image_size=cast(int, values["min_image_size"]),
        model_dir=model_dir,
        engines=EngineConfig(
            hidden=HiddenEngineConfig(
                model_dir=hidden_model_dir,
                providers=cast(tuple[str, ...], values["engines.hidden.providers"]),
            )
        ),
    )


def save_config_value(config_path: Path, key: str, value: str) -> None:
    canonical_key = _canonical_config_key(key)
    if canonical_key not in CONFIG_KEYS:
        supported = ", ".join(sorted({*CONFIG_KEYS, *CONFIG_KEY_ALIASES}))
        raise InvalidInputError(
            f"unknown config key: {key}",
            hint=f"Use one of: {supported}",
        )
    config = _read_config_file(config_path) if config_path.exists() else {}
    config[canonical_key] = _coerce_value(canonical_key, value)
    _write_config_file(config_path, config)


def reset_config_file(config_path: Path) -> None:
    if config_path.exists():
        config_path.unlink()


def _read_config_file(path: Path) -> dict[str, object]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"failed to parse config file: {path}",
            hint="Fix the TOML syntax or remove the invalid config file.",
        ) from exc
    result: dict[str, object] = {}
    for key, expected_type in CONFIG_KEYS.items():
        value = _lookup_config_value(data, key)
        if value is not None:
            result[key] = _normalize_loaded_value(expected_type, value)
    return result


def _write_config_file(path: Path, values: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nested = _to_nested_mapping(values)
    try:
        import tomli_w
    except ImportError:
        rendered = _dump_toml(nested)
    else:
        rendered = tomli_w.dumps(nested)
    path.write_text(rendered, encoding="utf-8")


def _environment_overrides() -> dict[str, object]:
    overrides: dict[str, object] = {}
    for key in CONFIG_KEYS:
        for env_name in _environment_names(key):
            if env_name in os.environ:
                overrides[key] = _coerce_value(key, os.environ[env_name])
                break
    return overrides


def _coerce_value(key: str, raw_value: str) -> object:
    expected_type = CONFIG_KEYS[key]
    if expected_type is Path:
        return Path(raw_value).expanduser()
    if expected_type is int:
        return int(raw_value)
    if expected_type is float:
        return float(raw_value)
    if expected_type is tuple:
        return tuple(item.strip() for item in raw_value.split(",") if item.strip())
    return raw_value


def _normalize_loaded_value(expected_type: type[Any], raw_value: object) -> object:
    if expected_type is Path:
        return Path(str(raw_value)).expanduser()
    if expected_type is int:
        return int(str(raw_value))
    if expected_type is float:
        return float(str(raw_value))
    if expected_type is tuple:
        if isinstance(raw_value, list):
            return tuple(str(item) for item in raw_value)
        return tuple(item.strip() for item in str(raw_value).split(",") if item.strip())
    return str(raw_value)


def _lookup_nested_value(data: dict[str, object], dotted_key: str) -> object | None:
    current: object = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _lookup_config_value(data: dict[str, object], key: str) -> object | None:
    value = _lookup_nested_value(data, key)
    if value is not None:
        return value
    for alias, canonical_key in CONFIG_KEY_ALIASES.items():
        if canonical_key != key:
            continue
        aliased_value = _lookup_nested_value(data, alias)
        if aliased_value is not None:
            return aliased_value
    return None


def _canonical_config_key(key: str) -> str:
    return CONFIG_KEY_ALIASES.get(key, key)


def _to_nested_mapping(values: dict[str, object]) -> dict[str, object]:
    nested: dict[str, object] = {}
    for key, value in values.items():
        current = nested
        parts = key.split(".")
        for part in parts[:-1]:
            current = cast(dict[str, object], current.setdefault(part, {}))
        current[parts[-1]] = _serialize_value(value)
    return nested


def _serialize_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _dump_toml(data: dict[str, object], *, prefix: tuple[str, ...] = ()) -> str:
    lines: list[str] = []
    scalar_items: list[tuple[str, object]] = []
    nested_items: list[tuple[str, dict[str, object]]] = []
    for key, value in data.items():
        if isinstance(value, dict):
            nested_items.append((key, cast(dict[str, object], value)))
        else:
            scalar_items.append((key, value))
    if prefix:
        lines.append(f"[{'.'.join(prefix)}]")
    for key, value in scalar_items:
        lines.append(f"{key} = {_render_toml_value(value)}")
    if scalar_items and nested_items:
        lines.append("")
    for index, (key, value) in enumerate(nested_items):
        lines.append(_dump_toml(value, prefix=(*prefix, key)).rstrip())
        if index != len(nested_items) - 1:
            lines.append("")
    return "\n".join(lines) + ("\n" if lines else "")


def _render_toml_value(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(_render_toml_value(item) for item in value) + "]"
    return str(value)


def _environment_names(key: str) -> tuple[str, ...]:
    base = key.upper().replace(".", "_")
    aliases = [f"STEGMARK_{base}"]
    if key == "engines.hidden.model_dir":
        aliases.append("STEGMARK_HIDDEN_MODEL_DIR")
    if key == "engines.hidden.providers":
        aliases.append("STEGMARK_HIDDEN_PROVIDERS")
    return tuple(aliases)
