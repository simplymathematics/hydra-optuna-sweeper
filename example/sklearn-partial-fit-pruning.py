# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Scikit-learn SGDClassifier example with Optuna pruning via ``partial_fit``.

This example demonstrates pruning inside a Hydra task function using
scikit-learn's incremental-learning API (``partial_fit``).  After each
mini-epoch the current validation accuracy is reported to the Optuna trial so
the configured MedianPruner can stop unpromising trials early.

How it works
------------
``SGDClassifier`` supports ``partial_fit``, which updates the model on one
pass through a mini-batch of data.  We treat each call to ``partial_fit`` as
one "step" and report the resulting validation accuracy with
``trial.report(score, step)``.  After each step we call
``trial.should_prune()``; if the pruner decides the trial is unlikely to
beat the current best we raise ``optuna.TrialPruned`` so the sweeper can
mark it as PRUNED.

To access the Optuna ``Trial`` object inside the Hydra task function we:

1. Read ``HydraConfig.get().job.id`` — this is the job index assigned by
   the sweeper and matches the trial number in a fresh Optuna study.
2. Load the study from persistent SQLite storage (configured via
   ``hydra.sweeper.storage``) and look up the trial by number.

Usage
-----
Run the sweep from the repository root::

    python example/sklearn-partial-fit-pruning.py --multirun
"""

import numpy as np
import optuna
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path="sklearn-partial-fit-pruning-conf",
    config_name="config",
)
def sklearn_partial_fit(cfg: DictConfig) -> float:
    """Train an SGDClassifier with ``partial_fit`` and return validation accuracy.

    Uses ``HydraConfig.get().job.id`` as the Optuna trial number to look up the
    trial from persistent storage and call ``trial.report`` / ``trial.should_prune``
    at each epoch so the MedianPruner can prune unpromising trials.
    """
    rng = np.random.RandomState(cfg.seed)

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
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=cfg.seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    classes = np.unique(y)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    clf = SGDClassifier(
        loss="log_loss",
        alpha=cfg.alpha,
        eta0=cfg.eta0,
        learning_rate="constant",
        random_state=cfg.seed,
    )

    # ------------------------------------------------------------------
    # Incremental training loop with Optuna pruning
    # ------------------------------------------------------------------
    batch_size = max(1, len(X_train) // cfg.n_epochs)
    best_val_acc = 0.0

    for epoch in range(cfg.n_epochs):
        # Draw a random mini-batch for this epoch.
        idx = rng.choice(len(X_train), size=batch_size, replace=False)
        clf.partial_fit(X_train[idx], y_train[idx], classes=classes)

        val_acc: float = float(clf.score(X_val, y_val))
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Report the intermediate value to Optuna so the pruner can
        # evaluate whether this trial is worth continuing.
        trial.report(val_acc, epoch)

        # Let the configured pruner (e.g. MedianPruner) decide whether
        # to stop this trial early.
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_acc


if __name__ == "__main__":
    sklearn_partial_fit()
