import os
import random
import torch
import yaml
import pandas as pd
import numpy as np
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from . import readers
from .datasets import CSIDataset
from .utils import load_params, train_model, resize_csi_to_fixed_length, load_custom_model
from .utils.cache import get_cache_key, load_cache, save_cache
from .processors.base_processor import BaseProcessor
from .processors import ConfigurableProcessor
from .models import create_model
from .algorithms import apply_preset, load_config as load_algorithm_config
from .record import SeedRecord, persist_pipeline_record
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Any, Dict, Optional, Tuple, Callable

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Hyperparameters:
    """Resolved training hyperparameters after config and call-site overrides."""

    batch_size: int
    learning_rate: float
    weight_decay: float
    num_epochs: int
    padding_length: int


@dataclass(frozen=True)
class PreprocessedData:
    """CSI arrays and metadata after loading and preprocessing."""

    processed_data: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    unique_labels: list


@dataclass(frozen=True)
class DataSplitBundle:
    """Train/validation/test arrays for one random seed."""

    train_data: np.ndarray
    val_data: np.ndarray
    test_data: np.ndarray
    train_labels: np.ndarray
    val_labels: np.ndarray
    test_labels: np.ndarray


@dataclass(frozen=True)
class LoaderBundle:
    """DataLoaders used by one training/evaluation seed."""

    train: DataLoader
    test: DataLoader
    val: DataLoader


@dataclass(frozen=True)
class SeedRunResult:
    """Metrics collected from one random seed run."""

    top1_accuracy: float
    record: SeedRecord


def _load_and_preprocess(
    input_path: str,
    dataset: str,
    pad_len: int,
    pipeline_steps: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Load CSI data, run processing pipeline, and return arrays ready for splitting.

    Returns:
        (processed_data, zero_indexed_labels, zero_indexed_groups, unique_labels)
    """
    csi_data_list = readers.load_data(input_path, dataset)

    if pipeline_steps is None:
        processor = BaseProcessor()
    else:
        processor = ConfigurableProcessor(pipeline_steps)

    res = processor.process(csi_data_list, dataset=dataset)

    unadjusted_data = res[0]
    processed_data = resize_csi_to_fixed_length(unadjusted_data, target_length=pad_len)
    logger.info(f"processed_data's shape: {processed_data[0].shape}")

    labels = res[1]
    groups = res[2]

    # Normalize labels and groups to stable zero-based integer IDs.
    unique_labels = sorted(list(set(labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    zero_indexed_labels = [label_map[label] for label in labels]

    unique_groups = sorted(list(set(groups)))
    group_map = {group: i for i, group in enumerate(unique_groups)}
    zero_indexed_groups = [group_map[group] for group in groups]

    logger.info(f"all unique labels idx: {list(set(zero_indexed_labels))}")
    logger.info(f"all unique groups idx: {list(set(zero_indexed_groups))}")
    logger.info(f"total sample: {len(processed_data)}, "
                f"total labels: {len(zero_indexed_labels)}, total groups: {len(zero_indexed_groups)}")

    processed_data = np.array(processed_data)
    zero_indexed_labels = np.array(zero_indexed_labels)
    zero_indexed_groups = np.array(zero_indexed_groups)

    return processed_data, zero_indexed_labels, zero_indexed_groups, unique_labels


def _load_pipeline_params(dataset: str, config_file: Optional[str]) -> Dict[str, Any]:
    """Load dataset defaults and optional YAML overrides."""
    params = load_params(dataset)

    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            yaml_params = yaml.safe_load(f)

        if yaml_params and dataset in yaml_params:
            params.update(yaml_params[dataset])

        logger.info(f"Loaded config from {config_file}")

    return params


def _effective_num_workers(num_workers: Optional[int]) -> int:
    """Resolve DataLoader worker count from explicit value or CPU count."""
    return num_workers if num_workers is not None else min(os.cpu_count() or 1, 8)


def _resolve_pipeline_steps(
    pipeline_steps: Optional[Dict[str, Dict[str, Any]]] = None,
    algorithm_config_file: Optional[str] = None,
    algorithm_preset: Optional[str] = None,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Resolve optional algorithm pipeline configuration.

    Priority: explicit pipeline_steps > config file > preset > legacy BaseProcessor.
    Returning None preserves the historical BaseProcessor behavior.
    """
    if pipeline_steps is not None:
        return pipeline_steps

    if algorithm_config_file is not None:
        return load_algorithm_config(algorithm_config_file)

    if algorithm_preset is not None:
        return apply_preset(algorithm_preset)

    return None


def _resolve_hyperparameters(
    params: Dict[str, Any],
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    num_epochs: Optional[int] = None,
    padding_length: Optional[int] = None,
) -> Hyperparameters:
    """Merge dataset defaults with explicit function-level overrides."""
    return Hyperparameters(
        batch_size=batch_size if batch_size is not None else params.get("batch", 32),
        learning_rate=learning_rate if learning_rate is not None else params.get("lr", 3e-4),
        weight_decay=weight_decay if weight_decay is not None else params.get("wd", 1e-3),
        num_epochs=num_epochs if num_epochs is not None else params.get("num_epochs", 20),
        padding_length=padding_length if padding_length is not None else params.get("padding_length", 1500),
    )


def _preprocessed_from_cache(cached_result: Dict[str, Any]) -> PreprocessedData:
    """Convert cache payloads into the same shape returned by preprocessing."""
    return PreprocessedData(
        processed_data=cached_result['processed_data'],
        labels=cached_result['labels'],
        groups=cached_result['groups'],
        unique_labels=cached_result['unique_labels'],
    )


def _load_or_preprocess_data(
    input_path: str,
    output_folder: str,
    dataset: str,
    padding_length: int,
    pipeline_steps: Optional[Dict[str, Dict[str, Any]]],
    use_cache: bool = True,
) -> PreprocessedData:
    """Load preprocessed data from cache when possible, otherwise process and cache it."""
    cache_dir = os.path.join(output_folder, '.wsdp_cache')
    cache_key = None

    if use_cache:
        # Include every preprocessing input that can affect cached output.
        cache_key = get_cache_key(
            input_path,
            dataset,
            padding_length,
            preprocess_config=pipeline_steps,
        )

        cached_result = load_cache(cache_dir, cache_key)
        if cached_result is not None:
            logger.info("Cache hit: loaded preprocessed data from cache")
            return _preprocessed_from_cache(cached_result)

        logger.info("Cache miss: processing data from scratch")

    processed_data, labels, groups, unique_labels = _load_and_preprocess(
        input_path,
        dataset,
        padding_length,
        pipeline_steps=pipeline_steps,
    )

    if use_cache and cache_key is not None:
        save_cache(cache_dir, cache_key, processed_data, labels, groups, unique_labels)

    return PreprocessedData(processed_data, labels, groups, unique_labels)


def _create_data_split(
    processed_data: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    test_split: float,
    val_split: float,
    seed: int,
    use_simple_split: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/val/test sets.

    Returns:
        (train_data, val_data, test_data, train_labels, val_labels, test_labels)
    """
    if use_simple_split:
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            processed_data, labels,
            test_size=test_split, random_state=seed
        )

        test_data, val_data, test_labels, val_labels = train_test_split(
            temp_data, temp_labels,
            test_size=val_split, random_state=seed
        )
    else:
        # Preserve group boundaries when carving out the held-out split.
        splitter_1 = GroupShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
        train_idx, temp_idx = next(
            splitter_1.split(processed_data, labels, groups=groups)
        )

        train_data = processed_data[train_idx]
        train_labels = labels[train_idx]

        temp_data = processed_data[temp_idx]
        temp_labels = labels[temp_idx]
        temp_groups = groups[temp_idx]

        # Split held-out groups again to keep test and validation disjoint.
        splitter_2 = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
        test_idx, val_idx = next(splitter_2.split(temp_data, temp_labels, groups=temp_groups))

        test_data = temp_data[test_idx]
        test_labels = temp_labels[test_idx]

        val_data = temp_data[val_idx]
        val_labels = temp_labels[val_idx]

    train_data = np.stack(train_data, axis=0)
    val_data = np.stack(val_data, axis=0)
    test_data = np.stack(test_data, axis=0)

    return train_data, val_data, test_data, train_labels, val_labels, test_labels


def _create_split_bundle(
    processed_data: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    test_split: float,
    val_split: float,
    seed: int,
    use_simple_split: bool,
) -> DataSplitBundle:
    """Create a named train/validation/test split for one seed."""
    return DataSplitBundle(
        *_create_data_split(
            processed_data,
            labels,
            groups,
            test_split,
            val_split,
            seed,
            use_simple_split,
        )
    )


def _create_loaders(
    split: DataSplitBundle,
    batch_size: int,
    num_workers: int,
) -> LoaderBundle:
    """Build train/test/validation DataLoaders from a split bundle."""
    train_dataset = CSIDataset(split.train_data, split.train_labels)
    test_dataset = CSIDataset(split.test_data, split.test_labels)
    val_dataset = CSIDataset(split.val_data, split.val_labels)

    return LoaderBundle(
        train=DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True),
        test=DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False),
        val=DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False),
    )


def _create_pipeline_model(
    model_path: Optional[str],
    model_name: str,
    model_kwargs: Dict[str, Any],
    num_classes: int,
    input_shape: Tuple[int, ...],
) -> nn.Module:
    """Create the configured model while preserving registered/custom behavior."""
    if model_path is None:
        return create_model(
            model_name,
            num_classes=num_classes,
            input_shape=input_shape,
            **model_kwargs,
        )

    return load_custom_model(
        model_path,
        num_classes,
        input_shape=input_shape,
        model_kwargs=model_kwargs,
    )


def _save_training_history(training_history, output_path: Path, seed: int) -> None:
    """Persist per-seed training history using the historical CSV format."""
    history_path = output_path / f"training_history_{seed}.csv"
    logger.info(f"training complete, save training_history to: {history_path}")
    pd.DataFrame(training_history).to_csv(history_path, index_label='epoch')


def _load_best_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    """Load the best checkpoint or raise the historical missing-file error."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f" no model in file path: {checkpoint_path}")
    logger.info(f"loading model from {checkpoint_path} ...")
    return torch.load(checkpoint_path, map_location=device)


def _plot_confusion_matrix(all_labels, all_predictions, output_path: Path, seed: int) -> None:
    """Write the per-seed confusion matrix image."""
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix (Random State: {seed})", fontsize=16)
    plt.ylabel("Actual Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / f"cm_rs_{seed}.png")
    plt.close()


def _seed_record_from_history(
    seed: int,
    training_history,
    checkpoint: Dict[str, Any],
    test_accuracy: float,
) -> SeedRecord:
    """Build the persisted per-seed record from training/evaluation outputs."""
    if isinstance(training_history, dict) and training_history.get('train_acc'):
        train_acc = training_history['train_acc'][-1] / 100.0
    else:
        train_acc = 0.0

    val_acc = checkpoint.get('best_val_acc', 0.0) / 100.0

    return SeedRecord(
        seed=seed,
        train_acc=train_acc,
        val_acc=val_acc,
        test_acc=test_accuracy,
    )


def _evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[list, list, float]:
    """Evaluate model on test set.

    Returns:
        (predictions, labels, accuracy)
    """
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch_idx, (csi_data_batch, test_labels_batch) in enumerate(
            tqdm(test_loader, desc="Evaluating", leave=False)
        ):
            csi_data_batch = csi_data_batch.to(device)
            test_labels_batch = test_labels_batch.to(device)

            outputs = model(csi_data_batch)

            _, predicted_classes = torch.max(outputs.data, 1)
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_labels.extend(test_labels_batch.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    return all_predictions, all_labels, accuracy


def _run_seed_training(
    seed_index: int,
    total_seeds: int,
    current_seed: int,
    preprocessed: PreprocessedData,
    output_path: Path,
    hyperparameters: Hyperparameters,
    test_split: float,
    val_split: float,
    use_simple_split: bool,
    model_path: Optional[str],
    model_name: str,
    model_kwargs: Dict[str, Any],
    device: torch.device,
    num_workers: int,
    progress_callback: Optional[Callable],
) -> SeedRunResult:
    """Train, evaluate, and persist artifacts for one random seed."""
    print(f"\n{'=' * 25} epoch {seed_index + 1}/{total_seeds} "
          f"begin (Random State: {current_seed}) {'=' * 25}\n")

    split = _create_split_bundle(
        preprocessed.processed_data,
        preprocessed.labels,
        preprocessed.groups,
        test_split,
        val_split,
        current_seed,
        use_simple_split,
    )

    logger.info(f"num of samples in train_data: {len(split.train_data)}, "
                f"num of samples in test_data: {len(split.test_data)}, num of samples in val_data: {len(split.val_data)}")
    logger.info(f"shape of first sample of train_data: {split.train_data[0].shape}, "
                f"shape of last sample of train_data: {split.train_data[-1].shape}")

    loaders = _create_loaders(split, hyperparameters.batch_size, num_workers)
    model = _create_pipeline_model(
        model_path=model_path,
        model_name=model_name,
        model_kwargs=model_kwargs,
        num_classes=len(preprocessed.unique_labels),
        input_shape=split.train_data[0].shape,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    checkpoint_path = output_path / f"best_checkpoint_{current_seed}.pth"

    logger.info("begin training")

    # Train with a per-seed checkpoint so evaluation always uses the best epoch.
    training_history = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loaders.train,
        val_loader=loaders.val,
        num_epochs=hyperparameters.num_epochs,
        device=device,
        checkpoint_path=checkpoint_path,
        padding_length=hyperparameters.padding_length,
        progress_callback=progress_callback,
    )

    _save_training_history(training_history, output_path, current_seed)
    logger.info("save successfully, begin to evaluate model")

    checkpoint = _load_best_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Keep evaluation outputs together because they feed logs, plots, and records.
    all_predictions, all_labels, current_top1_acc = _evaluate_model(model, loaders.test, device)
    logger.info("eval complete")
    logger.info(f"Top-1 acc of current epoch: {current_top1_acc:.4f}")
    logger.info("classification report:\n" + classification_report(all_labels, all_predictions))

    _plot_confusion_matrix(all_labels, all_predictions, output_path, current_seed)

    return SeedRunResult(
        top1_accuracy=current_top1_acc,
        record=_seed_record_from_history(
            seed=current_seed,
            training_history=training_history,
            checkpoint=checkpoint,
            test_accuracy=current_top1_acc,
        ),
    )


def _log_accuracy_summary(top1_accuracies: list) -> None:
    """Log aggregate accuracy metrics for all seed runs."""
    accuracies_np = np.array(top1_accuracies)
    mean_accuracy = np.mean(accuracies_np)
    variance_accuracy = np.var(accuracies_np)
    logger.info(f"All {len(top1_accuracies)} Top-1 acc: {[f'{acc:.4f}' for acc in top1_accuracies]}")
    logger.info(f"Avg Top-1 acc: {mean_accuracy:.4f}")
    logger.info(f"Variance of Top-1 acc: {variance_accuracy:.6f}")


def _processor_record(pipeline_steps: Optional[Dict[str, Dict[str, Any]]]) -> Tuple[str, Dict[str, Any]]:
    """Return processor metadata persisted in the pipeline record."""
    if pipeline_steps is None:
        return "BaseProcessor", {"phase_calibration": "default", "wavelet_denoise_csi": "default"}

    return "ConfigurableProcessor", pipeline_steps


def _persist_pipeline_summary(
    output_folder: str,
    dataset: str,
    total_samples: int,
    pipeline_steps: Optional[Dict[str, Dict[str, Any]]],
    model_path: Optional[str],
    model_name: str,
    seed_records: list,
) -> None:
    """Persist the aggregate JSON record for a pipeline run."""
    processor_type, processor_steps = _processor_record(pipeline_steps)
    model_str = f"custom:{model_path}" if model_path is not None else model_name

    persist_pipeline_record(
        output_folder=output_folder,
        dataset=dataset,
        total_samples=total_samples,
        reader_name=readers.get_reader_class(dataset).__name__,
        processor_type=processor_type,
        processor_steps=processor_steps,
        model=model_str,
        seed_records=seed_records,
    )


def pipeline(
    input_path: str,
    output_folder: str,
    dataset: str,
    model_path: Optional[str] = None,
    model_name: str = "CSIModel",
    model_kwargs: Optional[Dict[str, Any]] = None,
    pipeline_steps: Optional[Dict[str, Dict[str, Any]]] = None,
    algorithm_config_file: Optional[str] = None,
    algorithm_preset: Optional[str] = None,
    # Hyperparameter overrides
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    num_epochs: Optional[int] = None,
    padding_length: Optional[int] = None,
    test_split: float = 0.3,
    val_split: float = 0.5,
    num_seeds: int = 5,
    config_file: Optional[str] = None,
    num_workers: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
    use_cache: bool = True,
) -> None:
    """
    Run the full CSI classification pipeline.

    Args:
        input_path: Path to input data directory
        output_folder: Path to output directory
        dataset: Dataset name
        model_path: Optional path to custom model file
        model_name: Registered model name used when model_path is not provided
        model_kwargs: Extra keyword arguments passed to the registered/custom model
        pipeline_steps: Explicit algorithm pipeline steps for ConfigurableProcessor
        algorithm_config_file: YAML/JSON algorithm config file loaded by wsdp.algorithms.load_config
        algorithm_preset: Algorithm preset name loaded by wsdp.algorithms.apply_preset
        batch_size: Override default batch size
        learning_rate: Override default learning rate
        weight_decay: Override default weight decay
        num_epochs: Override default number of epochs
        padding_length: Override default padding length
        test_split: Fraction of data held out (not used for training) (default 0.3)
        val_split: Fraction of held-out data used for validation (default 0.5)
        num_seeds: Number of random seeds to run (default 5)
        config_file: Optional YAML config file to load parameters from
        num_workers: Number of DataLoader workers. When None, auto-detects: min(cpu_count, 8)
        progress_callback: Optional callable invoked after each training epoch with a dict of metrics
        use_cache: If True, cache preprocessed data to avoid re-processing on repeated runs (default True)
    """
    ipath = input_path
    os.makedirs(output_folder, exist_ok=True)
    opath = Path(output_folder)
    dataset_name = dataset

    effective_num_workers = _effective_num_workers(num_workers)

    model_kwargs = model_kwargs or {}

    # Resolve all runtime configuration before touching input data.
    resolved_pipeline_steps = _resolve_pipeline_steps(
        pipeline_steps=pipeline_steps,
        algorithm_config_file=algorithm_config_file,
        algorithm_preset=algorithm_preset,
    )

    if model_path is not None:
        logger.info(f"Loading model from {model_path}")
    else:
        logger.info(f"Loading registered model: {model_name}")

    if resolved_pipeline_steps is None:
        logger.info("Using default BaseProcessor preprocessing pipeline")
    else:
        logger.info(f"Using configurable preprocessing pipeline: {resolved_pipeline_steps}")

    try:
        params = _load_pipeline_params(dataset_name, config_file)
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"{e}")
        return

    hyperparameters = _resolve_hyperparameters(
        params,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        padding_length=padding_length,
    )

    pad_len = hyperparameters.padding_length

    random_seeds = [random.randint(0, 999) for _ in range(num_seeds)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(
        f"Hyperparameters: batch={hyperparameters.batch_size}, "
        f"lr={hyperparameters.learning_rate}, wd={hyperparameters.weight_decay}, "
        f"epochs={hyperparameters.num_epochs}, pad={pad_len}"
    )

    # Load preprocessing output once, then reuse it across all seed runs.
    preprocessed = _load_or_preprocess_data(
        input_path=ipath,
        output_folder=output_folder,
        dataset=dataset_name,
        padding_length=pad_len,
        pipeline_steps=resolved_pipeline_steps,
        use_cache=use_cache,
    )
    processed_data = preprocessed.processed_data
    zero_indexed_groups = preprocessed.groups

    logger.info(f"the following {num_seeds} seeds will be used: {random_seeds}")

    top1_accuracies = []
    seed_records = []

    # Fall back when grouped splitting cannot produce train/test/val partitions.
    n_groups = len(set(zero_indexed_groups))
    use_simple_split = n_groups < 3
    if use_simple_split:
        logger.warning(f"Only {n_groups} group(s) found (< 3). "
                       f"Using simple train_test_split instead of GroupShuffleSplit.")

    for i, current_seed in enumerate(random_seeds):
        seed_result = _run_seed_training(
            seed_index=i,
            total_seeds=len(random_seeds),
            current_seed=current_seed,
            preprocessed=preprocessed,
            output_path=opath,
            hyperparameters=hyperparameters,
            test_split=test_split,
            val_split=val_split,
            use_simple_split=use_simple_split,
            model_path=model_path,
            model_name=model_name,
            model_kwargs=model_kwargs,
            device=device,
            num_workers=effective_num_workers,
            progress_callback=progress_callback,
        )
        top1_accuracies.append(seed_result.top1_accuracy)
        seed_records.append(seed_result.record)

    _log_accuracy_summary(top1_accuracies)
    _persist_pipeline_summary(
        output_folder=output_folder,
        dataset=dataset_name,
        total_samples=len(processed_data),
        pipeline_steps=resolved_pipeline_steps,
        model_path=model_path,
        model_name=model_name,
        seed_records=seed_records,
    )

    logger.info("All pipeline complete")
