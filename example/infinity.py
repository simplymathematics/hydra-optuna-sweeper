# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Infinity objective example.

Returns (-inf, +inf) as the two objective values regardless of the input
parameters. This script demonstrates that the Optuna sweeper handles
floating-point infinities gracefully in a [maximize, minimize] study —
the typical sentinel values a function might return on failure.
"""
import math
from typing import Tuple

import hydra
from omegaconf import DictConfig


def compute_infinity(x: float, y: float) -> Tuple[float, float]:
    """Return (-inf, +inf) unconditionally to exercise infinity handling."""
    return (-math.inf, math.inf)


@hydra.main(version_base=None, config_path="infinity-conf", config_name="config")
def infinity_objective(cfg: DictConfig) -> Tuple[float, float]:
    return compute_infinity(cfg.x, cfg.y)


if __name__ == "__main__":
    infinity_objective()
