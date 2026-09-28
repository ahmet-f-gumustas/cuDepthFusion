"""Panel rendering for the comparison demo (spec 11).

Every depth panel in a run uses the same metre range and the same colormap, and every error
panel the same millimetre range, so two panels side by side can be compared by eye. The
scales are fixed once for the whole run and written into the run's ``summary.json``; they
never adapt per frame, because a per-frame stretch makes a worse result look like a better
one. Invalid pixels are black in every panel.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEPTH_COLORMAP = cv2.COLORMAP_TURBO
ERROR_COLORMAP = cv2.COLORMAP_INFERNO
CONFIDENCE_COLORMAP = cv2.COLORMAP_VIRIDIS
INVALID_COLOR = (0, 0, 0)
TITLE_HEIGHT = 26
FOOTER_HEIGHT = 74
BAR_HEIGHT = 14
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (235, 235, 235)
BACKGROUND = (18, 18, 18)
# source_mask values from the engine, with a colour each (BGR).
SOURCE_LEGEND = (
    (0, "invalid", (0, 0, 0)),
    (1, "current", (235, 180, 60)),
    (2, "fused", (90, 220, 90)),
    (3, "history", (60, 140, 250)),
)


@dataclass(frozen=True)
class Scales:
    """The fixed ranges shared by every panel and every frame of one run."""

    depth_min_m: float
    depth_max_m: float
    error_max_mm: float

    def to_dict(self) -> dict[str, float]:
        return {
            "depth_min_m": self.depth_min_m,
            "depth_max_m": self.depth_max_m,
            "error_max_mm": self.error_max_mm,
        }


@dataclass(frozen=True, eq=False)
class Panel:
    title: str
    image: np.ndarray  # BGR uint8


def auto_depth_range(samples: list[np.ndarray]) -> tuple[float, float]:
    """A range covering the whole run, taken from a few frames spread over it and then frozen.

    Taking it from the first frame alone would saturate every later frame once the camera has
    moved, which hides exactly the differences the demo exists to show. Empty input falls back
    to 0-1 m.
    """
    pool = np.concatenate([sample.ravel() for sample in samples]) if samples else np.empty(0)
    if pool.size == 0:
        return 0.0, 1.0
    low, high = (round(float(value), 2) for value in np.percentile(pool, (0.5, 99.5)))
    # Round first, then widen: rounding a hair-thin range would collapse it to zero width.
    if high - low < 0.01:
        high = low + 0.01
    return low, high


def _colorize(values: np.ndarray, valid: np.ndarray, low: float, high: float, cmap: int):
    span = max(high - low, 1e-9)
    scaled = np.clip((values - low) / span, 0.0, 1.0) * 255.0
    colored = cv2.applyColorMap(scaled.astype(np.uint8), cmap)
    colored[~valid] = INVALID_COLOR
    return colored


def depth_panel(title: str, depth_m: np.ndarray, valid: np.ndarray, scales: Scales) -> Panel:
    image = _colorize(depth_m, valid, scales.depth_min_m, scales.depth_max_m, DEPTH_COLORMAP)
    return Panel(title, image)


def error_panel(title: str, error_m: np.ndarray, mask: np.ndarray, scales: Scales) -> Panel:
    """``error_m`` is a signed or absolute error in metres; only its magnitude is shown."""
    image = _colorize(np.abs(error_m) * 1000.0, mask, 0.0, scales.error_max_mm, ERROR_COLORMAP)
    return Panel(title, image)


def source_panel(title: str, source_mask: np.ndarray) -> Panel:
    image = np.zeros((*source_mask.shape, 3), dtype=np.uint8)
    for value, _, color in SOURCE_LEGEND:
        image[source_mask == value] = color
    return Panel(title, image)


def confidence_panel(title: str, confidence: np.ndarray, valid: np.ndarray) -> Panel:
    return Panel(title, _colorize(confidence, valid, 0.0, 1.0, CONFIDENCE_COLORMAP))


def missing_panel(title: str, shape: tuple[int, int], reason: str) -> Panel:
    """A panel with no data says so; it is never filled with zeros that look like a result."""
    image = np.full((*shape, 3), 40, dtype=np.uint8)
    cv2.putText(image, reason, (8, shape[0] // 2), FONT, 0.4, TEXT_COLOR, 1, cv2.LINE_AA)
    return Panel(title, image)


def _with_title(panel: Panel, width: int) -> np.ndarray:
    height = int(round(panel.image.shape[0] * width / panel.image.shape[1]))
    body = cv2.resize(panel.image, (width, height), interpolation=cv2.INTER_NEAREST)
    bar = np.full((TITLE_HEIGHT, width, 3), BACKGROUND, dtype=np.uint8)
    cv2.putText(bar, panel.title, (8, 18), FONT, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
    return np.vstack([bar, body])


def _legend_strip(width: int) -> np.ndarray:
    strip = np.full((22, width, 3), BACKGROUND, dtype=np.uint8)
    x = 8
    for _, label, color in SOURCE_LEGEND:
        cv2.rectangle(strip, (x, 6), (x + 12, 16), color, -1)
        cv2.putText(strip, label, (x + 16, 16), FONT, 0.36, TEXT_COLOR, 1, cv2.LINE_AA)
        x += 20 + 8 * len(label)
    return strip


def _colorbar(width: int, cmap: int, low_label: str, high_label: str, caption: str) -> np.ndarray:
    ramp = np.linspace(0, 255, width, dtype=np.uint8)[None, :].repeat(BAR_HEIGHT, axis=0)
    bar = cv2.applyColorMap(ramp, cmap)
    block = np.full((BAR_HEIGHT + 22, width, 3), BACKGROUND, dtype=np.uint8)
    block[:BAR_HEIGHT] = bar
    cv2.putText(block, low_label, (2, BAR_HEIGHT + 16), FONT, 0.38, TEXT_COLOR, 1, cv2.LINE_AA)
    (text_w, _), _ = cv2.getTextSize(high_label, FONT, 0.38, 1)
    cv2.putText(
        block,
        high_label,
        (width - text_w - 2, BAR_HEIGHT + 16),
        FONT,
        0.38,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    (caption_w, _), _ = cv2.getTextSize(caption, FONT, 0.38, 1)
    cv2.putText(
        block,
        caption,
        ((width - caption_w) // 2, BAR_HEIGHT + 16),
        FONT,
        0.38,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    return block


def compose(
    panels: list[Panel],
    scales: Scales,
    status: str,
    *,
    columns: int = 3,
    panel_width: int = 420,
) -> np.ndarray:
    """The full demo frame: a grid of titled panels, the shared colorbars and a status line."""
    tiles = [_with_title(panel, panel_width) for panel in panels]
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        while len(row) < columns:
            row.append(np.full_like(tiles[0], BACKGROUND))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    width = grid.shape[1]

    footer = np.full((FOOTER_HEIGHT, width, 3), BACKGROUND, dtype=np.uint8)
    cv2.putText(footer, status, (8, 18), FONT, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
    half = width // 2 - 16
    depth_bar = _colorbar(
        half,
        DEPTH_COLORMAP,
        f"{scales.depth_min_m:.2f} m",
        f"{scales.depth_max_m:.2f} m",
        "depth (invalid = black)",
    )
    error_bar = _colorbar(
        half, ERROR_COLORMAP, "0 mm", f"{scales.error_max_mm:.0f} mm", "absolute error"
    )
    footer[28 : 28 + depth_bar.shape[0], 8 : 8 + half] = depth_bar
    footer[28 : 28 + error_bar.shape[0], width - half - 8 : width - 8] = error_bar
    return np.vstack([grid, footer, _legend_strip(width)])
