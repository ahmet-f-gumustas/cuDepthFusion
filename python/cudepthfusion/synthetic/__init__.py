"""Analytic synthetic scenes: the oracle for geometry and fusion tests.

Everything is computed in closed form (planar rectangles, exact ray casting, rigid
trajectories) and the sensor noise is seeded per frame, so sequences are reproducible and
their ground truth is exact. No third-party assets are involved.
"""

from cudepthfusion.synthetic.noise import NOISELESS, NoiseModel
from cudepthfusion.synthetic.scene import Rectangle, Scene, render
from cudepthfusion.synthetic.suite import (
    GroundTruthFrame,
    Scenario,
    SyntheticFrame,
    SyntheticSequence,
    get_scenario,
    scenario_names,
)

__all__ = [
    "NOISELESS",
    "GroundTruthFrame",
    "NoiseModel",
    "Rectangle",
    "Scenario",
    "Scene",
    "SyntheticFrame",
    "SyntheticSequence",
    "get_scenario",
    "render",
    "scenario_names",
]
