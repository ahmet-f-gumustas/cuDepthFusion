"""Configuration loading: YAML file or mapping -> validated native ``Config``.

The field list and types are read from the native binding, so the C++ structs stay the
single source of truth. Unknown keys and wrongly typed values are rejected, never ignored.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any

import yaml

from cudepthfusion import _core
from cudepthfusion._core import ConfigError

CORE_SECTIONS = ("depth", "spatial", "noise", "fusion", "reset")
BACKENDS = ("cpu", "cuda")
DEFAULT_BACKEND = "cpu"


@dataclass(frozen=True)
class BenchmarkConfig:
    warmup_frames: int = 100
    measured_frames: int = 500
    repeats: int = 5


@dataclass(frozen=True, eq=False)
class LoadedConfig:
    core: _core.Config
    backend: str = DEFAULT_BACKEND
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            **core_to_dict(self.core),
            "benchmark": asdict(self.benchmark),
        }


ConfigSource = LoadedConfig | Mapping[str, Any] | str | PathLike[str] | None


def load_config(source: ConfigSource = None) -> LoadedConfig:
    """Load and validate a configuration. ``None`` gives the built-in defaults."""
    if isinstance(source, LoadedConfig):
        return source
    if source is None:
        return _from_mapping({}, origin=None)
    if isinstance(source, Mapping):
        return _from_mapping(source, origin=None)
    path = Path(source)
    return _from_mapping(_read_yaml(path), origin=str(path))


def core_to_dict(core: _core.Config) -> dict[str, dict[str, Any]]:
    return {
        name: {key: getattr(getattr(core, name), key) for key in _field_types(getattr(core, name))}
        for name in CORE_SECTIONS
    }


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except OSError as error:
        raise ConfigError(f"cannot read config file {path}: {error.strerror or error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: invalid YAML: {error}") from error
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def _from_mapping(raw: Mapping[str, Any], origin: str | None) -> LoadedConfig:
    allowed = {"backend", "benchmark", *CORE_SECTIONS}
    unknown = sorted(str(key) for key in raw if key not in allowed)
    if unknown:
        raise ConfigError(f"unknown config section(s) {unknown}; allowed: {sorted(allowed)}")

    core = _core.Config()
    for name in CORE_SECTIONS:
        if name in raw:
            _apply_section(getattr(core, name), name, raw[name])
    _core.validate_config(core)

    return LoadedConfig(
        core=core,
        backend=_parse_backend(raw.get("backend", DEFAULT_BACKEND)),
        benchmark=_parse_benchmark(raw.get("benchmark") or {}),
        source=origin,
    )


def _field_types(section: object) -> dict[str, type]:
    """Field name -> Python type of its default, for a bound native config struct."""
    cls = type(section)
    return {
        name: type(getattr(section, name))
        for name, attribute in vars(cls).items()
        if isinstance(attribute, property)
    }


def _apply_section(target: object, name: str, values: Any) -> None:
    if not isinstance(values, Mapping):
        raise ConfigError(f"config section '{name}' must be a mapping, got {values!r}")
    types = _field_types(target)
    unknown = sorted(str(key) for key in values if key not in types)
    if unknown:
        raise ConfigError(f"unknown key(s) {unknown} in section '{name}'; allowed: {sorted(types)}")
    for key, value in values.items():
        path = f"{name}.{key}"
        try:
            setattr(target, key, _coerce(path, value, types[key]))
        except TypeError as error:  # e.g. an int too large for the native field
            raise ConfigError(f"config field '{path}' cannot hold {value!r}: {error}") from error


def _coerce(path: str, value: Any, expected: type) -> Any:
    if expected is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"config field '{path}' must be true or false, got {value!r}")
        return value
    if isinstance(value, str):
        hint = " (YAML 1.1 reads 1e-6 as a string; write 1.0e-6)" if _looks_numeric(value) else ""
        raise ConfigError(f"config field '{path}' must be a number, got string {value!r}{hint}")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"config field '{path}' must be a number, got {value!r}")
    if expected is int:
        if not isinstance(value, int):
            raise ConfigError(f"config field '{path}' must be an integer, got {value!r}")
        return value
    return float(value)


def _looks_numeric(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _parse_backend(value: Any) -> str:
    if value not in BACKENDS:
        raise ConfigError(f"config field 'backend' must be one of {list(BACKENDS)}, got {value!r}")
    return str(value)


def _parse_benchmark(values: Any) -> BenchmarkConfig:
    if not isinstance(values, Mapping):
        raise ConfigError(f"config section 'benchmark' must be a mapping, got {values!r}")
    defaults = asdict(BenchmarkConfig())
    unknown = sorted(str(key) for key in values if key not in defaults)
    if unknown:
        raise ConfigError(
            f"unknown key(s) {unknown} in section 'benchmark'; allowed: {sorted(defaults)}"
        )
    parsed = {key: _coerce(f"benchmark.{key}", value, int) for key, value in values.items()}
    merged = {**defaults, **parsed}
    minimums = {"warmup_frames": 0, "measured_frames": 1, "repeats": 1}
    for key, minimum in minimums.items():
        if merged[key] < minimum:
            raise ConfigError(
                f"config field 'benchmark.{key}' must be >= {minimum}, got {merged[key]}"
            )
    return BenchmarkConfig(**merged)
