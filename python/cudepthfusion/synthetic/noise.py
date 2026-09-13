"""Seeded depth-sensor noise: Gaussian with sigma(z) = a + b z^2, plus random dropouts.

This is the same functional form the filter assumes (spec 5.2), which makes synthetic
tests an oracle for the filter's own model. It is not a claim about any real sensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseModel:
    a_m: float = 0.002
    b_per_m: float = 0.001
    dropout_fraction: float = 0.01

    def __post_init__(self) -> None:
        if self.a_m < 0 or self.b_per_m < 0:
            raise ValueError(f"noise coefficients must be >= 0, got a={self.a_m}, b={self.b_per_m}")
        if not 0 <= self.dropout_fraction < 1:
            raise ValueError(f"dropout_fraction must be in [0, 1), got {self.dropout_fraction}")

    def sigma(self, z: np.ndarray) -> np.ndarray:
        return self.a_m + self.b_per_m * z * z

    def apply(self, clean_z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Noisy float32 depth. Both random draws always happen, so results depend only on
        the generator's seed, never on the scene content."""
        gaussian = rng.standard_normal(clean_z.shape)
        dropped = rng.random(clean_z.shape) < self.dropout_fraction
        noisy = clean_z + gaussian * self.sigma(clean_z)
        noisy[(clean_z <= 0) | dropped | (noisy <= 0)] = 0.0
        return noisy.astype(np.float32)


NOISELESS = NoiseModel(a_m=0.0, b_per_m=0.0, dropout_fraction=0.0)
