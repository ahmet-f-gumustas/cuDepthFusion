from __future__ import annotations

import json
from importlib.metadata import version

import pytest

import cudepthfusion as cdf
from cudepthfusion import cli


def test_version_matches_distribution_metadata() -> None:
    assert cdf.__version__ == version("cudepthfusion")


def test_public_api_is_exported() -> None:
    for name in cdf.__all__:
        assert hasattr(cdf, name), name


def test_build_info_is_consistent() -> None:
    info = cdf.build_info()
    assert info["version"] == cdf.__version__
    assert info["cuda_compiled"] == cdf.cuda_compiled()
    if not info["cuda_compiled"]:
        assert info["cuda_devices"] == []
        assert info["cuda_runtime_version"] is None


def test_info_cli_prints_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["info"]) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["cudepthfusion"]["version"] == cdf.__version__
    assert payload["platform"]["machine"]


def test_smoke_cli_passes_on_cpu(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["smoke", "--backend", "cpu", "--frames", "3", "--width", "64", "--height", "48"]
    code = cli.main(argv)
    summary = json.loads(capsys.readouterr().out)
    assert code == cli.EXIT_OK, summary["failures"]
    assert summary["passed"] is True
    assert [f["reset_reason"] for f in summary["frames"]] == ["first_frame", "none", "none"]


def test_smoke_cli_reports_unavailable_cuda(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["smoke", "--backend", "cuda", "--frames", "1"]) == cli.EXIT_USAGE_ERROR
    assert "backend 'cuda'" in capsys.readouterr().err


@pytest.mark.gpu
def test_cuda_device_is_reported() -> None:
    info = cdf.build_info()
    assert info["cuda_runtime_version"] > 0
    assert info["cuda_architectures"]
    for device in info["cuda_devices"]:
        major = int(device["compute_capability"].split(".")[0])
        assert major >= 5
        assert device["total_memory_bytes"] > 0
