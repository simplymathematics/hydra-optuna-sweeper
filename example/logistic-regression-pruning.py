# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Logistic-regression example with Optuna pruning support.

This example trains a single-layer PyTorch logistic regression model on
synthetic binary-classification data.  It demonstrates how to raise
``optuna.TrialPruned`` from inside a Hydra task function so that the
sweeper can mark the corresponding Optuna trial as PRUNED rather than
FAILED.

How pruning works in this integration
--------------------------------------
Optuna pruners (e.g. MedianPruner) decide *at each training step* whether
an ongoing trial is unlikely to beat the current best.  In the standard
Optuna API you obtain the ``Trial`` object directly, call
``trial.report(intermediate_value, step)`` to log the per-epoch metric, and
then check ``trial.should_prune()`` to see whether to stop early.

Inside a Hydra task function the ``Trial`` object is not passed in directly.
Instead, the Hydra job ID (``HydraConfig.get().job.id``) corresponds to the
Optuna trial number.  We use it to reload the trial from the study — which
requires a persistent storage (configured as
``hydra.sweeper.storage: sqlite:///logistic-regression-pruning.db`` in the
config).  This lets us call the real ``trial.report`` / ``trial.should_prune``
API so that the MedianPruner configured in the sweep actually makes the
pruning decision.

Usage
-----
Run the sweep from the repository root::

    python example/logistic-regression-pruning.py --multirun
"""

import optuna
import torch
import torch.nn as nn
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _make_data(
    n_samples: int,
    n_features: int,
    seed: int,
) -> tuple:
    """Create a linearly separable synthetic binary-classification dataset."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    X = torch.randn(n_samples, n_features, generator=rng)
    w = torch.randn(n_features, generator=rng)
    y = (X @ w > 0).float()
    return X, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LogisticRegression(nn.Module):
    """One-layer logistic regression."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear(x).squeeze(1)


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path="logistic-regression-pruning-conf",
    config_name="config",
)
def logistic_regression(cfg: DictConfig) -> float:
    """Train a logistic regression and return final validation loss.

    Uses ``HydraConfig.get().job.id`` as the Optuna trial number to reload
    the trial from persistent storage and invoke ``trial.report`` /
    ``trial.should_prune`` so the configured MedianPruner can prune
    unpromising trials.  Raises ``optuna.TrialPruned`` when pruned.
    """
    torch.manual_seed(cfg.seed)

    # ------------------------------------------------------------------
    # Retrieve the Optuna trial via the Hydra job ID.
    # HydraConfig.get().job.id is the job index assigned by the sweeper,
    # which matches the trial number in the Optuna study (for a fresh
    # study where trial numbers start from 0).
    # ------------------------------------------------------------------
    hydra_cfg = HydraConfig.get()
    try:
        trial_number = int(hydra_cfg.job.id)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Expected hydra.job.id to be an integer; got {hydra_cfg.job.id!r}."
        ) from exc
    study = optuna.load_study(
        study_name=hydra_cfg.sweeper.study_name,
        storage=hydra_cfg.sweeper.storage,
    )
    trial = study.trials[trial_number] if trial_number < len(study.trials) else None
    if trial is None:
        raise RuntimeError(
            f"Could not find Optuna trial with number {trial_number} in study "
            f"'{hydra_cfg.sweeper.study_name}'. "
            "Ensure the storage backend is configured and the study is fresh."
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    X, y = _make_data(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        seed=cfg.seed,
    )
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # ------------------------------------------------------------------
    # Model / optimizer
    # ------------------------------------------------------------------
    model = LogisticRegression(cfg.n_features)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    # Training loop with Optuna pruning
    # ------------------------------------------------------------------
    best_val_loss = float("inf")

    for epoch in range(cfg.epochs):
        # --- train step ---
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        # --- validation ---
        model.eval()
        with torch.no_grad():
            val_loss: float = criterion(model(X_val), y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Report the intermediate value to Optuna so the pruner can
        # evaluate whether this trial is worth continuing.
        trial.report(val_loss, epoch)

        # Let the configured pruner (e.g. MedianPruner) decide whether
        # to stop this trial early.
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_loss


if __name__ == "__main__":
    logistic_regression()

