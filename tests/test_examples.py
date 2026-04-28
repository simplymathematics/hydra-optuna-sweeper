# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Integration tests for the scripts in the example/ folder.

Each test invokes an example script as a subprocess using the Hydra CLI
(``python example/<script>.py --multirun ...``) and validates the
``optimization_results.yaml`` file written to the sweep directory.
"""
import math
import sys
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from hydra.test_utils.test_utils import chdir_plugin_root, run_python_script

# Ensure the working directory is the repository root so that relative
# config paths inside each example script resolve correctly.
chdir_plugin_root()


# ---------------------------------------------------------------------------
# sphere.py
# ---------------------------------------------------------------------------


def test_sphere_example(tmp_path: Path) -> None:
    """sphere.py minimises x²+y²; best value must be non-negative."""
    run_python_script(
        [
            "example/sphere.py",
            "--multirun",
            f"hydra.sweep.dir={tmp_path}",
            "hydra.job.chdir=False",
            "hydra.sweeper.n_trials=10",
            "hydra.sweeper.n_jobs=1",
            "hydra/sweeper/sampler=random",
            "hydra.sweeper.sampler.seed=0",
        ]
    )
    returns = OmegaConf.load(tmp_path / "optimization_results.yaml")
    assert isinstance(returns, DictConfig)
    assert returns.name == "optuna"
    assert "best_params" in returns
    assert "best_value" in returns
    assert returns.best_value >= 0.0


# ---------------------------------------------------------------------------
# multi-objective.py
# ---------------------------------------------------------------------------


def test_multi_objective_example(tmp_path: Path) -> None:
    """multi-objective.py runs the Binh-and-Korn benchmark; sweep must complete."""
    run_python_script(
        [
            "example/multi-objective.py",
            "--multirun",
            f"hydra.sweep.dir={tmp_path}",
            "hydra.job.chdir=False",
            "hydra.sweeper.n_trials=10",
            "hydra.sweeper.n_jobs=1",
            "hydra/sweeper/sampler=random",
            "hydra.sweeper.sampler.seed=0",
        ]
    )
    returns = OmegaConf.load(tmp_path / "optimization_results.yaml")
    assert isinstance(returns, DictConfig)
    assert returns.name == "optuna"
    assert "solutions" in returns
    assert len(returns.solutions) > 0
    for solution in returns.solutions:
        assert "params" in solution
        assert "values" in solution
        assert len(solution.values) == 2


# ---------------------------------------------------------------------------
# custom-search-space-objective.py
# ---------------------------------------------------------------------------


def test_custom_search_space_example(tmp_path: Path) -> None:
    """custom-search-space-objective.py uses a custom search-space callback."""
    run_python_script(
        [
            "example/custom-search-space-objective.py",
            "--multirun",
            f"hydra.sweep.dir={tmp_path}",
            "hydra.job.chdir=True",
            "hydra.sweeper.n_trials=10",
            "hydra.sweeper.n_jobs=1",
            "hydra/sweeper/sampler=random",
            "hydra.sweeper.sampler.seed=0",
        ]
    )
    returns = OmegaConf.load(tmp_path / "optimization_results.yaml")
    assert isinstance(returns, DictConfig)
    assert returns.name == "optuna"
    assert "best_params" in returns
    assert "best_value" in returns


# ---------------------------------------------------------------------------
# infinity.py
# ---------------------------------------------------------------------------


def test_infinity_example(tmp_path: Path) -> None:
    """infinity.py always returns (-inf, +inf).

    The sweep must complete without error for a [maximize, minimize] study,
    demonstrating that the sweeper handles infinite return values gracefully.
    All trials are Pareto-optimal since they all share the same objective
    values, so the solutions list must be non-empty.
    """
    run_python_script(
        [
            "example/infinity.py",
            "--multirun",
            f"hydra.sweep.dir={tmp_path}",
            "hydra.job.chdir=False",
        ]
    )
    returns = OmegaConf.load(tmp_path / "optimization_results.yaml")
    assert isinstance(returns, DictConfig)
    assert returns.name == "optuna"
    assert "solutions" in returns
    assert len(returns.solutions) > 0
    for solution in returns.solutions:
        assert "params" in solution
        assert "values" in solution
        v0, v1 = solution.values
        assert math.isinf(v0) and v0 < 0, f"Expected -inf, got {v0}"
        assert math.isinf(v1) and v1 > 0, f"Expected +inf, got {v1}"
