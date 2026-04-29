# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Scikit-learn progressive cross-validation example with Optuna pruning.

This example demonstrates pruning using a *progressive* cross-validation
strategy.  The trial starts with a small number of folds (``min_folds``) and
adds one more fold at a time up to ``max_folds``.  After each fold the
running mean validation accuracy is reported to the Optuna trial so the
MedianPruner can stop unpromising trials before they evaluate all folds.

How it works
------------
Standard k-fold cross-validation is not pruning-friendly because all folds
are computed in one shot.  The progressive variant used here instead evaluates
folds one by one:

* After fold *k* we have the mean CV score over the first *k* folds.
* We call ``trial.report(mean_score, step=k)`` to log this intermediate value.
* We call ``trial.should_prune()``; if the pruner says to stop we raise
  ``optuna.TrialPruned`` and skip the remaining ``max_folds - k`` folds.
* If the trial is not pruned it continues to ``max_folds`` folds and returns
  the full cross-validated mean score.

To access the Optuna ``Trial`` object inside the Hydra task function we:

1. Read ``HydraConfig.get().job.id`` — this is the job index assigned by
   the sweeper and matches the trial number in a fresh Optuna study.
2. Load the study from persistent SQLite storage (configured via
   ``hydra.sweeper.storage``) and look up the trial by number.

Usage
-----
Run the sweep from the repository root::

    python example/sklearn-cv-pruning.py --multirun
"""

import numpy as np
import optuna
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path="sklearn-cv-pruning-conf",
    config_name="config",
)
def sklearn_cv(cfg: DictConfig) -> float:
    """Evaluate a LogisticRegression with progressive CV and return mean accuracy.

    Uses ``HydraConfig.get().job.id`` as the Optuna trial number to look up the
    trial from persistent storage and call ``trial.report`` / ``trial.should_prune``
    after each fold so the MedianPruner can stop unpromising trials early.
    """
    # ------------------------------------------------------------------
    # Retrieve the Optuna trial via the Hydra job ID.
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
    X, y = make_classification(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        n_classes=cfg.n_classes,
        n_informative=max(2, cfg.n_features // 2),
        random_state=cfg.seed,
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    clf = LogisticRegression(
        C=cfg.C,
        max_iter=cfg.max_iter,
        random_state=cfg.seed,
        solver="lbfgs",
    )

    # ------------------------------------------------------------------
    # Progressive cross-validation with Optuna pruning.
    # We iterate fold by fold; after each fold we report the running mean
    # accuracy to the pruner so it can decide to stop early.
    # ------------------------------------------------------------------
    kf = StratifiedKFold(n_splits=cfg.max_folds, shuffle=True, random_state=cfg.seed)
    fold_scores: list = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        # Only use folds up to max_folds; stop early if pruned.
        if fold_idx >= cfg.max_folds:
            break

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf.fit(X_train, y_train)
        fold_scores.append(float(clf.score(X_val, y_val)))

        mean_score = float(np.mean(fold_scores))

        # Report the running mean after this fold so the pruner can assess
        # whether the trial is competitive.
        trial.report(mean_score, step=fold_idx)

        # Let the configured pruner (e.g. MedianPruner) decide whether to
        # stop evaluating more folds for this trial.
        if fold_idx + 1 >= cfg.min_folds and trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


if __name__ == "__main__":
    sklearn_cv()
