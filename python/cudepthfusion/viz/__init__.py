"""Visualisation: the side-by-side comparison demo and its panel renderer."""

from cudepthfusion.viz.demo import DemoError, DemoOptions, run_demo, write_demo_run
from cudepthfusion.viz.render import Panel, Scales, compose

__all__ = [
    "DemoError",
    "DemoOptions",
    "Panel",
    "Scales",
    "compose",
    "run_demo",
    "write_demo_run",
]
