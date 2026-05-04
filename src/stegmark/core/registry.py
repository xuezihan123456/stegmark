from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import cache, lru_cache
from importlib import import_module, metadata

from stegmark.core.engine import WatermarkEngine
from stegmark.exceptions import InvalidInputError

ENTRY_POINT_GROUP = "stegmark.engines"
_ALIASES = {"auto": "native"}
_BUILTIN_ENGINES = {
    "native": "stegmark.core.native:NativeEngine",
    "hidden": "stegmark.core.hidden:HiddenEngine",
    "trustmark": "stegmark.core.trustmark:TrustMarkEngine",
}
_REGISTERED_FACTORIES: dict[str, Callable[[], WatermarkEngine] | type[WatermarkEngine] | WatermarkEngine] = {}


def get_engine(name: str) -> WatermarkEngine:
    normalized = _normalize_name(name)
    try:
        return _build_engine(normalized)
    except KeyError as exc:
        available = ", ".join(sorted({"auto", *registered_engines()}))
        raise InvalidInputError(
            f"unknown engine: {name}",
            hint=f"Use one of: {available}.",
        ) from exc


def registered_engines() -> tuple[str, ...]:
    discovered = set(_BUILTIN_ENGINES)
    discovered.update(_REGISTERED_FACTORIES)
    discovered.update(_entry_point_map())
    return tuple(sorted(discovered))


def register_engine(
    name: str,
    factory: Callable[[], WatermarkEngine] | type[WatermarkEngine] | WatermarkEngine,
    *,
    override: bool = False,
) -> None:
    normalized = _normalize_name(name)
    exists = normalized in _BUILTIN_ENGINES or normalized in _REGISTERED_FACTORIES or normalized in _entry_point_map()
    if exists and not override:
        raise InvalidInputError(
            f"engine already registered: {name}",
            hint="Use override=True to replace an existing engine registration.",
        )
    _REGISTERED_FACTORIES[normalized] = factory
    clear_engine_cache()


def clear_engine_cache() -> None:
    _build_engine.cache_clear()
    _entry_point_map.cache_clear()


@cache
def _build_engine(name: str) -> WatermarkEngine:
    return _instantiate_engine(_resolve_factory(name))


def _resolve_factory(
    name: str,
) -> Callable[[], WatermarkEngine] | type[WatermarkEngine] | WatermarkEngine:
    if name in _REGISTERED_FACTORIES:
        return _REGISTERED_FACTORIES[name]
    if name in _BUILTIN_ENGINES:
        return _load_object(_BUILTIN_ENGINES[name])
    entry_point = _entry_point_map().get(name)
    if entry_point is not None:
        loaded = entry_point.load()
        if isinstance(loaded, WatermarkEngine):
            return loaded
        if isinstance(loaded, type) and issubclass(loaded, WatermarkEngine):
            return loaded
        if callable(loaded):
            return loaded
    raise KeyError(name)


def _instantiate_engine(
    factory: Callable[[], WatermarkEngine] | type[WatermarkEngine] | WatermarkEngine,
) -> WatermarkEngine:
    if isinstance(factory, WatermarkEngine):
        return factory
    if isinstance(factory, type) and issubclass(factory, WatermarkEngine):
        return factory()
    engine = factory()
    if isinstance(engine, WatermarkEngine):
        return engine
    raise TypeError("engine factory must return a WatermarkEngine instance")


def _normalize_name(name: str) -> str:
    normalized = name.lower()
    return _ALIASES.get(normalized, normalized)


def _load_object(import_path: str) -> type[WatermarkEngine]:
    module_name, _, attribute = import_path.partition(":")
    module = import_module(module_name)
    loaded = getattr(module, attribute)
    if isinstance(loaded, type) and issubclass(loaded, WatermarkEngine):
        return loaded
    raise TypeError(f"invalid engine import target: {import_path}")


@lru_cache(maxsize=1)
def _entry_point_map() -> dict[str, metadata.EntryPoint]:
    return {entry_point.name.lower(): entry_point for entry_point in _iter_entry_points()}


def _iter_entry_points() -> Iterable[metadata.EntryPoint]:
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        return tuple(entry_points.select(group=ENTRY_POINT_GROUP))
    return tuple(entry_points.get(ENTRY_POINT_GROUP, ()))
