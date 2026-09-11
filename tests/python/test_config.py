from __future__ import annotations

from pathlib import Path

import pytest

import cudepthfusion as cdf
from cudepthfusion import cli


def test_defaults_load_and_use_cpu_backend() -> None:
    loaded = cdf.load_config()
    assert loaded.backend == "cpu"
    assert loaded.core.fusion.history_decay == pytest.approx(0.95)
    assert loaded.benchmark == cdf.BenchmarkConfig()


def test_default_yaml_matches_native_defaults(default_config_path: Path) -> None:
    # Guards against drift between configs/default.yaml and include/cudepthfusion/config.hpp.
    from_yaml = cdf.load_config(default_config_path)
    assert from_yaml.to_dict() == cdf.load_config().to_dict()
    assert from_yaml.source == str(default_config_path)


def test_partial_mapping_overrides_only_given_fields() -> None:
    loaded = cdf.load_config({"fusion": {"k_sigma": 2}, "spatial": {"enabled": False}})
    assert loaded.core.fusion.k_sigma == 2.0
    assert loaded.core.spatial.enabled is False
    assert loaded.core.fusion.tau_abs_m == pytest.approx(0.015)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"filter": {}}, "unknown config section"),
        ({"fusion": {"k_sigmaa": 3.0}}, "unknown key(s) ['k_sigmaa']"),
        ({"fusion": []}, "must be a mapping"),
        ({"spatial": {"enabled": 1}}, "must be true or false"),
        ({"spatial": {"radius": 2.5}}, "must be an integer"),
        ({"fusion": {"k_sigma": True}}, "must be a number"),
        ({"fusion": {"variance_floor_m2": "1e-6"}}, "write 1.0e-6"),
        ({"fusion": {"max_history_age_frames": 10**12}}, "cannot hold"),
        ({"fusion": {"history_decay": 1.5}}, "fusion.history_decay"),
        ({"depth": {"min_m": 5.0, "max_m": 1.0}}, "depth.max_m"),
        ({"backend": "gpu"}, "backend"),
        ({"benchmark": {"repeats": 0}}, "benchmark.repeats"),
        ({"benchmark": {"warmup": 1}}, "unknown key(s) ['warmup']"),
    ],
)
def test_invalid_values_are_rejected_with_context(raw: dict, message: str) -> None:
    with pytest.raises(cdf.ConfigError) as excinfo:
        cdf.load_config(raw)
    assert message in str(excinfo.value)


def test_config_error_is_a_value_error() -> None:
    assert issubclass(cdf.ConfigError, ValueError)


def test_yaml_top_level_must_be_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(cdf.ConfigError, match="top level must be a mapping"):
        cdf.load_config(path)


def test_unreadable_or_malformed_yaml_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(cdf.ConfigError, match="cannot read config file"):
        cdf.load_config(tmp_path / "missing.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("fusion: [1, 2\n  bad: x\n", encoding="utf-8")
    with pytest.raises(cdf.ConfigError, match="invalid YAML"):
        cdf.load_config(broken)


def test_benchmark_section_must_be_mapping() -> None:
    with pytest.raises(cdf.ConfigError, match="'benchmark' must be a mapping"):
        cdf.load_config({"benchmark": [1, 2]})


def test_engine_config_reflects_what_the_engine_runs_with() -> None:
    loaded = cdf.load_config({"fusion": {"tau_abs_m": 0.05}})
    engine = cdf.DepthFusion(loaded)
    loaded.core.fusion.tau_abs_m = 999.0  # caller edits after construction must not leak in
    assert engine.config.core.fusion.tau_abs_m == pytest.approx(0.05)
    engine.config.core.fusion.tau_abs_m = 123.0  # nor can edits through the returned copy
    assert engine.config.core.fusion.tau_abs_m == pytest.approx(0.05)


def test_cli_reports_bad_config_paths_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["check-config", str(tmp_path / "missing.yaml")]) == cli.EXIT_USAGE_ERROR
    assert "cannot read config file" in capsys.readouterr().err
    assert cli.main(["smoke", "--config", str(tmp_path / "missing.yaml")]) == cli.EXIT_USAGE_ERROR
    assert "Traceback" not in capsys.readouterr().err


def test_check_config_cli(
    tmp_path: Path, default_config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["check-config", str(default_config_path)]) == cli.EXIT_OK
    assert "history_decay: 0.95" in capsys.readouterr().out

    bad = tmp_path / "bad.yaml"
    bad.write_text("fusion:\n  history_decay: 2.0\n", encoding="utf-8")
    assert cli.main(["check-config", str(bad)]) == cli.EXIT_USAGE_ERROR
    assert "fusion.history_decay" in capsys.readouterr().err
