# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class DistributionType(Enum):
    int = 1
    float = 2
    categorical = 3


class Direction(Enum):
    minimize = 1
    maximize = 2


@dataclass
class SamplerConfig:
    _target_: str = MISSING


@dataclass
class PrunerConfig:
    _target_: str = MISSING


@dataclass
class GridSamplerConfig(SamplerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.GridSampler.html
    """

    _target_: str = "optuna.samplers.GridSampler"
    # search_space will be populated at run time based on hydra.sweeper.params
    _partial_: bool = True


@dataclass
class TPESamplerConfig(SamplerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html
    """

    _target_: str = "optuna.samplers.TPESampler"
    seed: Optional[int] = None

    consider_prior: bool = True
    prior_weight: float = 1.0
    consider_magic_clip: bool = True
    consider_endpoints: bool = False
    n_startup_trials: int = 10
    n_ei_candidates: int = 24
    multivariate: bool = False
    warn_independent_sampling: bool = True


@dataclass
class RandomSamplerConfig(SamplerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.RandomSampler.html
    """

    _target_: str = "optuna.samplers.RandomSampler"
    seed: Optional[int] = None


@dataclass
class CmaEsSamplerConfig(SamplerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.CmaEsSampler.html
    """

    _target_: str = "optuna.samplers.CmaEsSampler"
    seed: Optional[int] = None

    x0: Optional[Dict[str, Any]] = None
    sigma0: Optional[float] = None
    independent_sampler: Optional[Any] = None
    warn_independent_sampling: bool = True
    consider_pruned_trials: bool = False
    restart_strategy: Optional[Any] = None
    inc_popsize: int = 2
    use_separable_cma: bool = False
    source_trials: Optional[Any] = None


@dataclass
class NSGAIISamplerConfig(SamplerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.NSGAIISampler.html
    """

    _target_: str = "optuna.samplers.NSGAIISampler"
    seed: Optional[int] = None

    population_size: int = 50
    mutation_prob: Optional[float] = None
    crossover_prob: float = 0.9
    swapping_prob: float = 0.5
    constraints_func: Optional[Any] = None


@dataclass
class NopPrunerConfig(PrunerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.NopPruner.html
    """

    _target_: str = "optuna.pruners.NopPruner"


@dataclass
class MedianPrunerConfig(PrunerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html
    """

    _target_: str = "optuna.pruners.MedianPruner"
    n_startup_trials: int = 5
    n_warmup_steps: int = 0
    interval_steps: int = 1
    n_min_trials: int = 1


@dataclass
class PercentilePrunerConfig(PrunerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.PercentilePruner.html
    """

    _target_: str = "optuna.pruners.PercentilePruner"
    percentile: float = 25.0
    n_startup_trials: int = 5
    n_warmup_steps: int = 0
    interval_steps: int = 1
    n_min_trials: int = 1


@dataclass
class SuccessiveHalvingPrunerConfig(PrunerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.SuccessiveHalvingPruner.html
    """

    _target_: str = "optuna.pruners.SuccessiveHalvingPruner"
    min_resource: Any = "auto"
    reduction_factor: int = 4
    min_early_stopping_rate: int = 0
    bootstrap_count: int = 0


@dataclass
class HyperbandPrunerConfig(PrunerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html
    """

    _target_: str = "optuna.pruners.HyperbandPruner"
    min_resource: int = 1
    max_resource: Any = "auto"
    reduction_factor: int = 3
    bootstrap_count: int = 0


@dataclass
class ThresholdPrunerConfig(PrunerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.ThresholdPruner.html
    """

    _target_: str = "optuna.pruners.ThresholdPruner"
    lower: Optional[float] = None
    upper: Optional[float] = None
    n_warmup_steps: int = 0
    interval_steps: int = 1


@dataclass
class PatientPrunerConfig(PrunerConfig):
    """
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.PatientPruner.html
    """

    _target_: str = "optuna.pruners.PatientPruner"
    wrapped_pruner: Optional[Any] = None
    patience: int = 0
    min_delta: float = 0.0


@dataclass
class DistributionConfig:
    # Type of distribution. "int", "float" or "categorical"
    type: DistributionType

    # Choices of categorical distribution
    # List element type should be Union[str, int, float, bool]
    choices: Optional[List[Any]] = None

    # Lower bound of int or float distribution
    low: Optional[float] = None

    # Upper bound of int or float distribution
    high: Optional[float] = None

    # If True, space is converted to the log domain
    # Valid for int or float distribution
    log: bool = False

    # Discritization step
    # Valid for int or float distribution
    step: Optional[float] = None


defaults = [{"sampler": "tpe"}, {"pruner": "nop"}]


@dataclass
class OptunaSweeperConf:
    _target_: str = "hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper"
    defaults: List[Any] = field(default_factory=lambda: defaults)

    # Sampling algorithm
    # Please refer to the reference for further details
    # https://optuna.readthedocs.io/en/stable/reference/samplers.html
    sampler: SamplerConfig = MISSING

    # Pruning algorithm
    # Please refer to the reference for further details
    # https://optuna.readthedocs.io/en/stable/reference/pruners.html
    pruner: PrunerConfig = MISSING

    # Direction of optimization
    # Union[Direction, List[Direction]]
    direction: Any = Direction.minimize

    # Storage URL to persist optimization results
    # For example, you can use SQLite if you set 'sqlite:///example.db'
    # Please refer to the reference for further details
    # https://optuna.readthedocs.io/en/stable/reference/storages.html
    storage: Optional[Any] = None

    # Name of study to persist optimization results
    study_name: Optional[str] = None

    # Total number of function evaluations
    n_trials: int = 20

    # Number of parallel workers
    n_jobs: int = 2

    # Maximum authorized failure rate for a batch of parameters
    max_failure_rate: float = 0.0

    search_space: Optional[Dict[str, Any]] = None

    params: Optional[Dict[str, str]] = None

    # Allow custom trial configuration via Python methods.
    # If given, `custom_search_space` should be a an instantiate-style dotpath targeting
    # a callable with signature Callable[[DictConfig, optuna.trial.Trial], None].
    # https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html
    custom_search_space: Optional[str] = None


ConfigStore.instance().store(
    group="hydra/sweeper",
    name="optuna",
    node=OptunaSweeperConf,
    provider="optuna_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/sampler",
    name="tpe",
    node=TPESamplerConfig,
    provider="optuna_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/sampler",
    name="random",
    node=RandomSamplerConfig,
    provider="optuna_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/sampler",
    name="cmaes",
    node=CmaEsSamplerConfig,
    provider="optuna_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/sampler",
    name="nsgaii",
    node=NSGAIISamplerConfig,
    provider="optuna_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/sampler",
    name="grid",
    node=GridSamplerConfig,
    provider="optuna_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/pruner",
    name="nop",
    node=NopPrunerConfig,
    provider="optuna_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/pruner",
    name="median",
    node=MedianPrunerConfig,
    provider="optuna_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/pruner",
    name="percentile",
    node=PercentilePrunerConfig,
    provider="optuna_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/pruner",
    name="successive_halving",
    node=SuccessiveHalvingPrunerConfig,
    provider="optuna_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/pruner",
    name="hyperband",
    node=HyperbandPrunerConfig,
    provider="optuna_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/pruner",
    name="threshold",
    node=ThresholdPrunerConfig,
    provider="optuna_sweeper",
)

ConfigStore.instance().store(
    group="hydra/sweeper/pruner",
    name="patient",
    node=PatientPrunerConfig,
    provider="optuna_sweeper",
)
