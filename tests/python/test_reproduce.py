from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

from fake_icl import build_fake_icl

from cudepthfusion import cli
from cudepthfusion.data.fetch import fetch_sequence
from cudepthfusion.data.registry import SEQUENCES, VARIANTS

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "reproduce.py"
ICL = SEQUENCES["icl-nuim"]


def load_script(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while it executes
    spec.loader.exec_module(module)
    return module


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )


def fake_icl_root(tmp_path: Path) -> Path:
    """kt0-kt3 as small offline look-alikes, with the real names and splits."""
    fake = build_fake_icl(tmp_path / "fake")
    root = tmp_path / "data"
    for name, spec in ICL.items():
        fetch_sequence(
            replace(fake.spec, name=name, split=spec.split),
            VARIANTS,
            root,
            command="test",
            downloader=fake.downloader,
            log=lambda _: None,
        )
    return root


def test_every_step_is_a_command_the_real_parsers_accept(tmp_path: Path) -> None:
    reproduce = load_script(SCRIPT)
    demo = load_script(REPO_ROOT / "examples" / "compare_depth.py")
    steps = reproduce.plan_steps(tmp_path / "data", tmp_path / "runs", "cuda", None)

    assert [step.name for step in steps] == [
        *(f"download {name}" for name in ICL),
        *(f"validate {name}" for name in ICL),
        "evaluate validation",
        "ablate validation",
        "evaluate test",
        "demo kt0",
        "benchmark",
    ]
    for step in steps:
        if step.argv[:2] == ("-m", "cudepthfusion.cli"):
            cli._build_parser().parse_args(step.argv[2:])
        elif step.argv[0].endswith("compare_depth.py"):
            demo.build_parser().parse_args(step.argv[1:])
        else:
            assert step.argv[0].endswith("download_dataset.py")


def test_downloaded_sequences_are_not_fetched_again(tmp_path: Path) -> None:
    reproduce = load_script(SCRIPT)
    (tmp_path / "data" / "kt1").mkdir(parents=True)
    (tmp_path / "data" / "kt1" / "manifest.json").write_text("{}", encoding="utf-8")

    steps = reproduce.plan_steps(tmp_path / "data", tmp_path / "runs", "cpu", 3)
    downloads = [step.name for step in steps if step.name.startswith("download")]
    assert downloads == ["download kt0", "download kt2", "download kt3"]
    evaluate = next(step for step in steps if step.name == "evaluate test")
    assert "--frames" in evaluate.argv and "3" in evaluate.argv


def test_dry_run_prints_the_commands_and_writes_nothing(tmp_path: Path) -> None:
    output = tmp_path / "runs"
    result = run("--dry-run", "--data-root", str(tmp_path / "data"), "--output", str(output))
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("\n") == 13
    assert "cudepthfusion.cli evaluate --split test" in result.stdout
    assert not output.exists()


def test_one_command_runs_every_step_end_to_end(tmp_path: Path) -> None:
    root = fake_icl_root(tmp_path)
    output = tmp_path / "runs"
    result = run(
        *("--data-root", str(root), "--output", str(output)),
        *("--backend", "cpu", "--frames", "4"),
    )
    assert result.returncode == 0, result.stderr[-3000:]

    report = json.loads((output / "reproduce.json").read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["frames_limit"] == 4
    assert len(report["steps"]) == 9  # nothing to download
    assert all(step["exit_code"] == 0 for step in report["steps"])
    for folder in ("validation", "ablation", "test", "demo/kt0", "performance"):
        assert (output / folder / "summary.json").exists(), folder
    assert (output / "demo" / "kt0" / "demo.mp4").stat().st_size > 0
    test_summary = json.loads((output / "test" / "summary.json").read_text(encoding="utf-8"))
    assert {entry["sequence"] for entry in test_summary["sequences"]} == {"kt2", "kt3"}


def test_the_first_failing_step_stops_the_run(tmp_path: Path) -> None:
    root = tmp_path / "data"
    for name in ICL:
        (root / name).mkdir(parents=True)
        (root / name / "manifest.json").write_text("not json", encoding="utf-8")
    output = tmp_path / "runs"
    result = run("--data-root", str(root), "--output", str(output), "--backend", "cpu")

    assert result.returncode != 0
    assert "step 'validate kt0' failed" in result.stderr
    report = json.loads((output / "reproduce.json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert [step["name"] for step in report["steps"]] == ["validate kt0"]
