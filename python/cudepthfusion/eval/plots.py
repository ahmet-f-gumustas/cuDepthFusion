"""Figures for a run folder (spec 8: results plus automatic plots)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # a run must not need a display
import matplotlib.pyplot as plt  # noqa: E402

FIGURE_DPI = 140


def write_plots(output: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    written = [
        _quality_bars(output, summary),
        _edge_interior_bars(output, summary),
        _per_frame_curve(output, rows),
    ]
    return [path for path in written if path is not None]


def _baselines(summary: dict[str, Any]) -> list[str]:
    return sorted(summary["overall"]["pooled"])


def _quality_bars(output: Path, summary: dict[str, Any]) -> Path | None:
    names = _baselines(summary)
    pooled = summary["overall"]["pooled"]
    if not names:
        return None
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for axis, (key, label, scale) in zip(
        axes,
        [
            ("rmse_m", "RMSE (mm)", 1000.0),
            ("median_abs_m", "median |error| (mm)", 1000.0),
            ("bad_pixel_rate", "bad pixels (%)", 100.0),
        ],
        strict=True,
    ):
        values = [(pooled[name].get(key) or 0.0) * scale for name in names]
        axis.bar(names, values, color="#4878a8")
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.3)
    figure.suptitle("Pooled over every evaluated pixel")
    figure.tight_layout()
    path = output / "quality.png"
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return path


def _edge_interior_bars(output: Path, summary: dict[str, Any]) -> Path | None:
    names = _baselines(summary)
    pooled = summary["overall"]["pooled"]
    if not names:
        return None
    figure, axis = plt.subplots(figsize=(7, 3.6))
    width = 0.38
    positions = range(len(names))
    edge = [(pooled[name].get("edge_rmse_m") or 0.0) * 1000.0 for name in names]
    interior = [(pooled[name].get("interior_rmse_m") or 0.0) * 1000.0 for name in names]
    axis.bar([p - width / 2 for p in positions], edge, width, label="edge region", color="#c0623f")
    axis.bar([p + width / 2 for p in positions], interior, width, label="interior", color="#4878a8")
    axis.set_xticks(list(positions))
    axis.set_xticklabels(names)
    axis.set_ylabel("RMSE (mm)")
    axis.set_title("Depth edges carry the outliers; the interior is where fusion works")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path = output / "edge_vs_interior.png"
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return path


def _per_frame_curve(output: Path, rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None
    sequence = rows[0]["sequence"]
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row["sequence"] != sequence:
            continue
        series[row["baseline"]].append((int(row["frame_id"]), float(row["rmse_m"]) * 1000.0))
    figure, axis = plt.subplots(figsize=(9, 3.6))
    for name in sorted(series):
        points = sorted(series[name])
        axis.plot([p[0] for p in points], [p[1] for p in points], label=name, linewidth=1.0)
    axis.set_xlabel("frame")
    axis.set_ylabel("RMSE (mm)")
    axis.set_title(f"Per-frame RMSE, {sequence}")
    axis.legend(ncol=5, fontsize=8)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    path = output / "per_frame_rmse.png"
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return path
