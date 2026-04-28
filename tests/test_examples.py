# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Integration tests for the scripts in the example/ folder.

Each test runs an example script as a subprocess:
    python example/<script>.py --multirun ...
and validates the ``optimization_results.yaml`` written to the sweep directory.
"""
import math
import subprocess
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

# Repository root — all example scripts are run from here so that their
# relative config_path arguments resolve correctly.
REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(*args: str) -> None:
    """Run ``python <args>`` from the repository root, raising on failure."""
    result = subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Command failed (exit {result.returncode}):\n"
            f"  {' '.join([sys.executable, *args])}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# sphere.py
# ---------------------------------------------------------------------------


def test_sphere_example(tmp_path: Path) -> None:
    """sphere.py minimises x²+y²; best value must be non-negative."""
    _run(
        "example/sphere.py",
        "--multirun",
        "hydra/sweeper=optuna",
        f"hydra.sweep.dir={tmp_path}",
        "hydra.job.chdir=False",
        "hydra.sweeper.n_trials=10",
        "hydra.sweeper.n_jobs=1",
        "hydra/sweeper/sampler=random",
        "hydra.sweeper.sampler.seed=0",
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
    _run(
        "example/multi-objective.py",
        "--multirun",
        "hydra/sweeper=optuna",
        f"hydra.sweep.dir={tmp_path}",
        "hydra.job.chdir=False",
        "hydra.sweeper.n_trials=10",
        "hydra.sweeper.n_jobs=1",
        "hydra/sweeper/sampler=random",
        "hydra.sweeper.sampler.seed=0",
    )
    returns = OmegaConf.load(tmp_path / "optimization_results.yaml")
    assert isinstance(returns, DictConfig)
    assert returns.name == "optuna"
    assert "solutions" in returns
    assert len(returns.solutions) > 0
    for solution in returns.solutions:
        assert "params" in solution
        assert "values" in solution
        assert len(solution["values"]) == 2


# ---------------------------------------------------------------------------
# custom-search-space-objective.py
# ---------------------------------------------------------------------------


def test_custom_search_space_example(tmp_path: Path) -> None:
    """custom-search-space-objective.py uses a custom search-space callback."""
    _run(
        "example/custom-search-space-objective.py",
        "--multirun",
        "hydra/sweeper=optuna",
        f"hydra.sweep.dir={tmp_path}",
        "hydra.job.chdir=True",
        "hydra.sweeper.n_trials=10",
        "hydra.sweeper.n_jobs=1",
        "hydra/sweeper/sampler=random",
        "hydra.sweeper.sampler.seed=0",
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
    All trials share the same objective values so the solutions list must be
    non-empty.
    """
    _run(
        "example/infinity.py",
        "--multirun",
        "hydra/sweeper=optuna",
        f"hydra.sweep.dir={tmp_path}",
        "hydra.job.chdir=False",
    )
    returns = OmegaConf.load(tmp_path / "optimization_results.yaml")
    assert isinstance(returns, DictConfig)
    assert returns.name == "optuna"
    assert "solutions" in returns
    assert len(returns.solutions) > 0
    for solution in returns.solutions:
        assert "params" in solution
        assert "values" in solution
        v0, v1 = solution["values"]
        assert math.isinf(v0) and v0 < 0, f"Expected -inf, got {v0}"
        assert math.isinf(v1) and v1 > 0, f"Expected +inf, got {v1}"
