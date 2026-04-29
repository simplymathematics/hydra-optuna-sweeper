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

Inside a Hydra task function the ``Trial`` object is not passed in
directly.  To retain a clean, file-system-free example we therefore
implement our *own* lightweight early-stopping rule: after a short warmup
we prune any trial whose running validation loss is worse than the best
loss seen so far across all epochs (patience == 0).  Raising
``optuna.TrialPruned`` signals the sweeper to mark the trial as PRUNED.

For full ``trial.report`` / ``trial.should_prune`` integration (needed when
you want the sweeper's configured pruner to make the pruning decision),
configure a persistent storage (e.g. ``storage: sqlite:///study.db``) and
retrieve the trial inside the task function with::

    study = optuna.load_study(study_name=cfg.hydra.sweeper.study_name,
                              storage=cfg.hydra.sweeper.storage)
    trial  = study.trials[cfg.optuna_trial_number]   # pass via sweeper params

Usage
-----
Run the sweep from the repository root::

    python example/logistic-regression-pruning.py --multirun
"""

import optuna
import torch
import torch.nn as nn
import hydra
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

    Raises ``optuna.TrialPruned`` when the trial is not promising so that
    the sweeper records the trial as PRUNED instead of FAILED.
    """
    torch.manual_seed(cfg.seed)

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
    # Training loop with early-stopping-based pruning
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

        # Track best validation loss seen so far.
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Prune after a short warmup if the current val loss is strictly
        # worse than the best seen in any previous epoch.
        # This mimics the behaviour of ``trial.should_prune()`` without
        # requiring access to the Optuna Trial object.
        if epoch >= 2 and val_loss > best_val_loss:
            raise optuna.TrialPruned()

    return best_val_loss


if __name__ == "__main__":
    logistic_regression()
