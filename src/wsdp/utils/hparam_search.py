"""Hyperparameter search utilities for WSDP.

Requires `optuna <https://optuna.org>`_ (``pip install optuna``).
"""

import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def suggest_search_space(trial) -> dict:
    """Default hyperparameter search space for Optuna.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    dict
        Dictionary of suggested hyperparameters.
    """
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "num_epochs": trial.suggest_int("num_epochs", 10, 100),
    }


def run_hparam_search(
    input_path: str,
    output_folder: str,
    dataset: str,
    n_trials: int = 20,
    search_space_fn: Optional[Callable] = None,
    seed: int = 42,
    padding_length: int = 1500,
    test_split: float = 0.3,
    val_split: float = 0.5,
    direction: str = "maximize",
    **pipeline_kwargs,
) -> "optuna.Study":  # type: ignore[name-defined]  # noqa: F821
    """Run hyperparameter search using Optuna.

    Each trial loads the data, trains a model with the suggested
    hyperparameters, and returns the validation accuracy as the
    objective value.

    Parameters
    ----------
    input_path : str
        Path to the input data directory.
    output_folder : str
        Path to the output directory for checkpoints / artifacts.
    dataset : str
        Dataset name recognised by :func:`wsdp.core._load_and_preprocess`.
    n_trials : int, default=20
        Number of Optuna trials to run.
    search_space_fn : callable or None
        Custom ``fn(trial) -> dict`` that returns hyperparameters.
        Uses :func:`suggest_search_space` when *None*.
    seed : int, default=42
        Random seed for the data split.
    padding_length : int, default=1500
        CSI padding length passed to ``_load_and_preprocess``.
    test_split : float, default=0.3
        Fraction of data held out from training.
    val_split : float, default=0.5
        Fraction of held-out data used for validation vs. test.
    direction : str, default='maximize'
        Optimisation direction (``'maximize'`` for accuracy).
    **pipeline_kwargs
        Extra keyword arguments (currently unused, reserved for future use).

    Returns
    -------
    optuna.Study
        The completed Optuna study.

    Raises
    ------
    ImportError
        If ``optuna`` is not installed.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Optuna is required for hyperparameter search. "
            "Install it with: pip install optuna"
        )

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from wsdp.core import _load_and_preprocess, _create_data_split, _evaluate_model
    from wsdp.datasets import CSIDataset
    from wsdp.models import CSIModel
    from wsdp.utils import train_model

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Pre-load data once (expensive I/O) and reuse across trials.
    logger.info("Loading and preprocessing data for hparam search ...")
    processed_data, labels, groups, unique_labels = _load_and_preprocess(
        input_path, dataset, padding_length
    )
    num_classes = len(unique_labels)

    n_groups = len(set(groups.tolist()))
    use_simple_split = n_groups < 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial: "optuna.Trial") -> float:  # type: ignore[name-defined]
        space_fn = search_space_fn or suggest_search_space
        hparams = space_fn(trial)

        lr = hparams.get("lr", 3e-4)
        wd = hparams.get("weight_decay", 1e-3)
        batch_size = hparams.get("batch_size", 32)
        num_epochs = hparams.get("num_epochs", 20)

        # Split data
        train_data, val_data, test_data, train_labels, val_labels, test_labels = (
            _create_data_split(
                processed_data, labels, groups,
                test_split, val_split, seed, use_simple_split,
                dataset=dataset,
            )
        )

        # DataLoaders
        train_loader = DataLoader(
            CSIDataset(train_data, train_labels),
            batch_size=batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            CSIDataset(val_data, val_labels),
            batch_size=batch_size, shuffle=False,
        )
        test_loader = DataLoader(
            CSIDataset(test_data, test_labels),
            batch_size=batch_size, shuffle=False,
        )

        # Model, optimiser, scheduler
        model = CSIModel(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5,
        )

        checkpoint_path = output_path / f"hparam_trial_{trial.number}.pth"

        # Train
        train_model(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            device=device,
            checkpoint_path=checkpoint_path,
            padding_length=padding_length,
        )

        # Reload best checkpoint and evaluate
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

        _, _, accuracy = _evaluate_model(model, test_loader, device)

        logger.info("Trial %d — accuracy: %.4f | lr=%.2e wd=%.2e bs=%d epochs=%d",
                     trial.number, accuracy, lr, wd, batch_size, num_epochs)
        return accuracy

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    logger.info("Best trial: %s", study.best_trial.params)
    logger.info("Best accuracy: %.4f", study.best_value)

    return study
