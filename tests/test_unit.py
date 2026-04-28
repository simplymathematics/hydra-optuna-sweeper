# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Pure unit tests for hydra-optuna-sweeper.

These tests exercise individual functions and class methods in isolation,
without spawning subprocesses or requiring a full Hydra sweep runner.
"""
import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import optuna
import pytest
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import DictConfig, OmegaConf
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.samplers import RandomSampler

from hydra_plugins.hydra_optuna_sweeper import _impl
from hydra_plugins.hydra_optuna_sweeper._impl import (
    OptunaSweeperImpl,
    create_optuna_distribution_from_config,
    create_optuna_distribution_from_override,
    create_params_from_overrides,
)
from hydra_plugins.hydra_optuna_sweeper.config import (
    Direction,
    DistributionConfig,
    DistributionType,
    GridSamplerConfig,
    NSGAIISamplerConfig,
    OptunaSweeperConf,
    RandomSamplerConfig,
    TPESamplerConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sweeper(
    direction: Any = Direction.minimize,
    params: Optional[DictConfig] = None,
    search_space: Optional[DictConfig] = None,
    n_trials: int = 10,
    n_jobs: int = 1,
    max_failure_rate: float = 0.0,
) -> OptunaSweeperImpl:
    """Construct a minimal OptunaSweeperImpl for unit testing."""
    return OptunaSweeperImpl(
        sampler=RandomSampler(),
        direction=direction,
        storage=None,
        study_name="unit-test",
        n_trials=n_trials,
        n_jobs=n_jobs,
        max_failure_rate=max_failure_rate,
        search_space=search_space,
        custom_search_space=None,
        params=params,
    )


# ---------------------------------------------------------------------------
# DistributionType and Direction enums
# ---------------------------------------------------------------------------

class TestEnums:
    def test_distribution_type_values(self) -> None:
        assert DistributionType.int.value == 1
        assert DistributionType.float.value == 2
        assert DistributionType.categorical.value == 3

    def test_distribution_type_membership(self) -> None:
        assert DistributionType["int"] is DistributionType.int
        assert DistributionType["float"] is DistributionType.float
        assert DistributionType["categorical"] is DistributionType.categorical

    def test_direction_values(self) -> None:
        assert Direction.minimize.value == 1
        assert Direction.maximize.value == 2

    def test_direction_names(self) -> None:
        assert Direction.minimize.name == "minimize"
        assert Direction.maximize.name == "maximize"


# ---------------------------------------------------------------------------
# DistributionConfig dataclass
# ---------------------------------------------------------------------------

class TestDistributionConfig:
    def test_categorical_config(self) -> None:
        cfg = DistributionConfig(type=DistributionType.categorical, choices=[1, 2, 3])
        assert cfg.type is DistributionType.categorical
        assert cfg.choices == [1, 2, 3]
        assert cfg.low is None
        assert cfg.high is None
        assert cfg.log is False
        assert cfg.step is None

    def test_int_config_defaults(self) -> None:
        cfg = DistributionConfig(type=DistributionType.int, low=0.0, high=10.0)
        assert cfg.type is DistributionType.int
        assert cfg.low == 0.0
        assert cfg.high == 10.0
        assert cfg.log is False
        assert cfg.step is None

    def test_float_config_with_step(self) -> None:
        cfg = DistributionConfig(type=DistributionType.float, low=0.0, high=5.0, step=0.5)
        assert cfg.step == 0.5

    def test_int_config_log(self) -> None:
        cfg = DistributionConfig(type=DistributionType.int, low=1.0, high=100.0, log=True)
        assert cfg.log is True


# ---------------------------------------------------------------------------
# create_optuna_distribution_from_config
# ---------------------------------------------------------------------------

class TestCreateDistributionFromConfig:
    @pytest.mark.parametrize(
        "config, expected",
        [
            # categorical
            (
                {"type": "categorical", "choices": [1, 2, 3]},
                CategoricalDistribution([1, 2, 3]),
            ),
            # int basic
            (
                {"type": "int", "low": 0, "high": 10},
                IntDistribution(0, 10),
            ),
            # int with step
            (
                {"type": "int", "low": 0, "high": 10, "step": 2},
                IntDistribution(0, 10, step=2),
            ),
            # int log
            (
                {"type": "int", "low": 1, "high": 100, "log": True},
                IntDistribution(1, 100, log=True),
            ),
            # float basic
            (
                {"type": "float", "low": 0.0, "high": 1.0},
                FloatDistribution(0.0, 1.0),
            ),
            # float with step
            (
                {"type": "float", "low": 0.0, "high": 10.0, "step": 2.0},
                FloatDistribution(0.0, 10.0, step=2.0),
            ),
            # float log
            (
                {"type": "float", "low": 1.0, "high": 100.0, "log": True},
                FloatDistribution(1.0, 100.0, log=True),
            ),
        ],
    )
    def test_create_distribution(self, config: Any, expected: BaseDistribution) -> None:
        actual = create_optuna_distribution_from_config(config)
        if isinstance(expected, CategoricalDistribution):
            assert isinstance(actual, CategoricalDistribution)
            assert set(actual.choices) == set(expected.choices)
        else:
            assert actual == expected

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises((NotImplementedError, KeyError)):
            create_optuna_distribution_from_config({"type": "unsupported"})


# ---------------------------------------------------------------------------
# create_optuna_distribution_from_override
# ---------------------------------------------------------------------------

def _parse_override(override_str: str) -> Any:
    parser = OverridesParser.create()
    return parser.parse_overrides([override_str])[0]


class TestCreateDistributionFromOverride:
    @pytest.mark.parametrize(
        "override_str, expected",
        [
            # choice sweep → CategoricalDistribution
            ("key=choice(1,2,3)", CategoricalDistribution([1, 2, 3])),
            ("key=choice(true,false)", CategoricalDistribution([True, False])),
            ("key=choice('a','b')", CategoricalDistribution(["a", "b"])),
            # shuffle range → CategoricalDistribution
            ("key=shuffle(range(1,4))", CategoricalDistribution((1, 2, 3))),
            # integer range → IntDistribution
            ("key=range(1,5)", IntDistribution(1, 5)),
            # float interval → FloatDistribution
            ("key=interval(0,1)", FloatDistribution(0, 1)),
            # int interval
            ("key=int(interval(1,5))", IntDistribution(1, 5)),
            # log interval float
            ("key=tag(log,interval(1,100))", FloatDistribution(1, 100, log=True)),
            # log interval int
            ("key=tag(log,int(interval(1,100)))", IntDistribution(1, 100, log=True)),
            # range with float step → FloatDistribution with step
            ("key=range(0.0,5.0,step=1.0)", FloatDistribution(0.0, 5.0, step=1.0)),
        ],
    )
    def test_create_distribution(self, override_str: str, expected: Any) -> None:
        override = _parse_override(override_str)
        actual = create_optuna_distribution_from_override(override)
        if isinstance(expected, CategoricalDistribution):
            assert isinstance(actual, CategoricalDistribution)
            assert set(actual.choices) == set(expected.choices)
        else:
            assert actual == expected

    def test_fixed_value_returns_string(self) -> None:
        override = _parse_override("key=42")
        result = create_optuna_distribution_from_override(override)
        assert result == "42"

    def test_fixed_string_value(self) -> None:
        override = _parse_override("key=hello")
        result = create_optuna_distribution_from_override(override)
        assert result == "hello"


# ---------------------------------------------------------------------------
# create_params_from_overrides
# ---------------------------------------------------------------------------

class TestCreateParamsFromOverrides:
    def test_single_sweep_param(self) -> None:
        distributions, fixed = create_params_from_overrides(["key=choice(1,2)"])
        assert "key" in distributions
        assert isinstance(distributions["key"], CategoricalDistribution)
        assert fixed == {}

    def test_single_fixed_param(self) -> None:
        distributions, fixed = create_params_from_overrides(["key=5"])
        assert distributions == {}
        assert fixed == {"key": "5"}

    def test_mixed_params(self) -> None:
        distributions, fixed = create_params_from_overrides(
            ["x=choice(0,1)", "y=3", "z=range(1,4)"]
        )
        assert set(distributions.keys()) == {"x", "z"}
        assert isinstance(distributions["x"], CategoricalDistribution)
        assert isinstance(distributions["z"], IntDistribution)
        assert fixed == {"y": "3"}

    def test_empty_overrides(self) -> None:
        distributions, fixed = create_params_from_overrides([])
        assert distributions == {}
        assert fixed == {}

    def test_multiple_sweep_params(self) -> None:
        distributions, fixed = create_params_from_overrides(
            ["a=interval(0,1)", "b=range(0,10)"]
        )
        assert isinstance(distributions["a"], FloatDistribution)
        assert isinstance(distributions["b"], IntDistribution)
        assert fixed == {}


# ---------------------------------------------------------------------------
# OptunaSweeperImpl._get_directions
# ---------------------------------------------------------------------------

class TestGetDirections:
    def test_single_minimize(self) -> None:
        sweeper = _make_sweeper(direction=Direction.minimize)
        assert sweeper._get_directions() == ["minimize"]

    def test_single_maximize(self) -> None:
        sweeper = _make_sweeper(direction=Direction.maximize)
        assert sweeper._get_directions() == ["maximize"]

    def test_direction_as_string(self) -> None:
        sweeper = _make_sweeper(direction="minimize")
        assert sweeper._get_directions() == ["minimize"]

    def test_list_of_directions(self) -> None:
        sweeper = _make_sweeper(direction=[Direction.minimize, Direction.maximize])
        assert sweeper._get_directions() == ["minimize", "maximize"]

    def test_list_of_direction_strings(self) -> None:
        sweeper = _make_sweeper(direction=["minimize", "maximize"])
        assert sweeper._get_directions() == ["minimize", "maximize"]


# ---------------------------------------------------------------------------
# OptunaSweeperImpl._parse_sweeper_params_config
# ---------------------------------------------------------------------------

class TestParseSweepParamsConfig:
    def test_no_params_returns_empty(self) -> None:
        sweeper = _make_sweeper(params=None)
        # params=None → no param list before _process_searchspace_config
        sweeper.params = None
        assert sweeper._parse_sweeper_params_config() == []

    def test_empty_params_returns_empty(self) -> None:
        sweeper = _make_sweeper(params=OmegaConf.create({}))
        assert sweeper._parse_sweeper_params_config() == []

    def test_params_are_formatted_correctly(self) -> None:
        params = OmegaConf.create({"x": "choice(1,2)", "y": "interval(0,1)"})
        sweeper = _make_sweeper(params=params)
        result = sweeper._parse_sweeper_params_config()
        assert "x=choice(1,2)" in result
        assert "y=interval(0,1)" in result

    def test_fixed_param(self) -> None:
        params = OmegaConf.create({"lr": "0.001"})
        sweeper = _make_sweeper(params=params)
        result = sweeper._parse_sweeper_params_config()
        assert result == ["lr=0.001"]


# ---------------------------------------------------------------------------
# OptunaSweeperImpl._to_grid_sampler_choices
# ---------------------------------------------------------------------------

class TestToGridSamplerChoices:
    def test_categorical_distribution(self) -> None:
        sweeper = _make_sweeper()
        dist = CategoricalDistribution(["a", "b", "c"])
        assert sweeper._to_grid_sampler_choices(dist) == ("a", "b", "c")

    def test_int_distribution_with_step(self) -> None:
        sweeper = _make_sweeper()
        dist = IntDistribution(0, 6, step=2)
        choices = sweeper._to_grid_sampler_choices(dist)
        assert list(choices) == [0, 2, 4]

    def test_int_distribution_default_step(self) -> None:
        sweeper = _make_sweeper()
        dist = IntDistribution(0, 3, step=1)
        choices = sweeper._to_grid_sampler_choices(dist)
        assert list(choices) == [0, 1, 2]

    def test_float_distribution_with_step(self) -> None:
        sweeper = _make_sweeper()
        dist = FloatDistribution(0.0, 1.0, step=0.5)
        choices = sweeper._to_grid_sampler_choices(dist)
        assert len(choices) == 2
        assert choices[0] == pytest.approx(0.0)
        assert choices[1] == pytest.approx(0.5)

    def test_float_distribution_without_step_raises(self) -> None:
        sweeper = _make_sweeper()
        dist = FloatDistribution(0.0, 1.0)
        with pytest.raises((AssertionError, ValueError)):
            sweeper._to_grid_sampler_choices(dist)

    def test_unsupported_distribution_raises(self) -> None:
        sweeper = _make_sweeper()
        # FloatDistribution with log has no step → unsupported for grid
        dist = FloatDistribution(1.0, 100.0, log=True)
        with pytest.raises((AssertionError, ValueError)):
            sweeper._to_grid_sampler_choices(dist)


# ---------------------------------------------------------------------------
# OptunaSweeperImpl._process_searchspace_config
# ---------------------------------------------------------------------------

class TestProcessSearchspaceConfig:
    def test_none_params_and_none_search_space_initializes_empty_params(self) -> None:
        sweeper = _make_sweeper(params=None, search_space=None)
        sweeper._process_searchspace_config()
        assert sweeper.params is not None
        assert len(sweeper.params) == 0

    def test_params_only_no_warning(self) -> None:
        sweeper = _make_sweeper(params=OmegaConf.create({"x": "choice(1,2)"}))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sweeper._process_searchspace_config()
        assert len(w) == 0

    def test_search_space_only_emits_deprecation_warning(self) -> None:
        search_space = OmegaConf.create(
            {"x": {"type": "float", "low": 0.0, "high": 1.0}}
        )
        sweeper = _make_sweeper(params=None, search_space=search_space)
        with pytest.warns(UserWarning, match=r".*search_space.*deprecated.*"):
            sweeper._process_searchspace_config()

    def test_both_params_and_search_space_emits_conflict_warning(self) -> None:
        search_space = OmegaConf.create({})
        params = OmegaConf.create({})
        sweeper = _make_sweeper(params=params, search_space=search_space)
        with pytest.warns(UserWarning, match=r"(?i).*both.*"):
            sweeper._process_searchspace_config()

    def test_search_space_only_populates_distributions(self) -> None:
        search_space = OmegaConf.create(
            {"x": {"type": "float", "low": 0.0, "high": 1.0}}
        )
        sweeper = _make_sweeper(params=None, search_space=search_space)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            sweeper._process_searchspace_config()
        assert sweeper.search_space_distributions is not None
        assert "x" in sweeper.search_space_distributions
        assert isinstance(sweeper.search_space_distributions["x"], FloatDistribution)


# ---------------------------------------------------------------------------
# OptunaSweeperImpl constructor validation
# ---------------------------------------------------------------------------

class TestOptunaSweeperImplInit:
    def test_valid_max_failure_rate(self) -> None:
        sweeper = _make_sweeper(max_failure_rate=0.5)
        assert sweeper.max_failure_rate == 0.5

    def test_zero_max_failure_rate(self) -> None:
        sweeper = _make_sweeper(max_failure_rate=0.0)
        assert sweeper.max_failure_rate == 0.0

    def test_one_max_failure_rate(self) -> None:
        sweeper = _make_sweeper(max_failure_rate=1.0)
        assert sweeper.max_failure_rate == 1.0

    def test_negative_max_failure_rate_raises(self) -> None:
        with pytest.raises(AssertionError):
            _make_sweeper(max_failure_rate=-0.1)

    def test_max_failure_rate_above_one_raises(self) -> None:
        with pytest.raises(AssertionError):
            _make_sweeper(max_failure_rate=1.1)

    def test_attributes_stored_correctly(self) -> None:
        sweeper = _make_sweeper(
            direction=Direction.maximize,
            n_trials=50,
            n_jobs=4,
        )
        assert sweeper.direction is Direction.maximize
        assert sweeper.n_trials == 50
        assert sweeper.n_jobs == 4
        assert sweeper.storage is None
        assert sweeper.study_name == "unit-test"


# ---------------------------------------------------------------------------
# Sampler config dataclasses
# ---------------------------------------------------------------------------

class TestSamplerConfigs:
    def test_tpe_sampler_defaults(self) -> None:
        cfg = TPESamplerConfig()
        assert cfg._target_ == "optuna.samplers.TPESampler"
        assert cfg.seed is None
        assert cfg.n_startup_trials == 10
        assert cfg.multivariate is False

    def test_random_sampler_defaults(self) -> None:
        cfg = RandomSamplerConfig()
        assert cfg._target_ == "optuna.samplers.RandomSampler"
        assert cfg.seed is None

    def test_grid_sampler_defaults(self) -> None:
        cfg = GridSamplerConfig()
        assert cfg._target_ == "optuna.samplers.GridSampler"
        assert cfg._partial_ is True

    def test_nsgaii_sampler_defaults(self) -> None:
        cfg = NSGAIISamplerConfig()
        assert cfg._target_ == "optuna.samplers.NSGAIISampler"
        assert cfg.population_size == 50
        assert cfg.mutation_prob is None


# ---------------------------------------------------------------------------
# OptunaSweeperConf dataclass
# ---------------------------------------------------------------------------

class TestOptunaSweeperConf:
    def test_default_target(self) -> None:
        conf = OptunaSweeperConf()
        assert (
            conf._target_
            == "hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper"
        )

    def test_default_direction(self) -> None:
        conf = OptunaSweeperConf()
        assert conf.direction is Direction.minimize

    def test_default_n_trials(self) -> None:
        conf = OptunaSweeperConf()
        assert conf.n_trials == 20

    def test_default_n_jobs(self) -> None:
        conf = OptunaSweeperConf()
        assert conf.n_jobs == 2

    def test_default_max_failure_rate(self) -> None:
        conf = OptunaSweeperConf()
        assert conf.max_failure_rate == 0.0

    def test_default_storage_is_none(self) -> None:
        conf = OptunaSweeperConf()
        assert conf.storage is None

    def test_default_params_is_none(self) -> None:
        conf = OptunaSweeperConf()
        assert conf.params is None

    def test_default_search_space_is_none(self) -> None:
        conf = OptunaSweeperConf()
        assert conf.search_space is None


# ---------------------------------------------------------------------------
# example/infinity.py — pure-function logic
# ---------------------------------------------------------------------------

import importlib.util
import math
import os

_EXAMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "example")
)


def _import_from_file(module_name: str, file_path: str) -> Any:
    """Import a module from an arbitrary file path (handles hyphens in names)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


_inf_mod = _import_from_file("infinity", os.path.join(_EXAMPLE_DIR, "infinity.py"))
compute_infinity = _inf_mod.compute_infinity


class TestExampleInfinity:
    """Unit tests for example/infinity.py — the infinity objective."""

    def test_returns_negative_inf_first(self) -> None:
        v0, _ = compute_infinity(0.0, 0.0)
        assert math.isinf(v0)
        assert v0 < 0  # -inf

    def test_returns_positive_inf_second(self) -> None:
        _, v1 = compute_infinity(0.0, 0.0)
        assert math.isinf(v1)
        assert v1 > 0  # +inf

    def test_returns_tuple_of_two(self) -> None:
        result = compute_infinity(1.0, 2.0)
        assert len(result) == 2

    def test_independent_of_x(self) -> None:
        """Result must not depend on x."""
        assert compute_infinity(0.0, 0.0) == compute_infinity(99.0, 0.0)

    def test_independent_of_y(self) -> None:
        """Result must not depend on y."""
        assert compute_infinity(0.0, 0.0) == compute_infinity(0.0, -42.0)

    def test_first_value_is_neg_inf(self) -> None:
        v0, _ = compute_infinity(0.0, 0.0)
        assert v0 == -math.inf

    def test_second_value_is_pos_inf(self) -> None:
        _, v1 = compute_infinity(0.0, 0.0)
        assert v1 == math.inf

    def test_float_convertible(self) -> None:
        """Both values must be representable as Python floats."""
        v0, v1 = compute_infinity(0.0, 0.0)
        assert isinstance(float(v0), float)
        assert isinstance(float(v1), float)
