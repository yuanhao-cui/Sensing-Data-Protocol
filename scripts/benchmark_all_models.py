#!/usr/bin/env python3
"""Benchmark all registered models on a dataset.

Usage:
    python scripts/benchmark_all_models.py <input_path> <output_base> <dataset> [--epochs N] [--seeds N]
"""
import sys
import os
import json
import time
import traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from wsdp.models.registry import MODEL_REGISTRY
from wsdp.core import _load_and_preprocess, _create_data_split, _evaluate_model
from wsdp.utils import load_params, resize_csi_to_fixed_length
from wsdp.datasets import CSIDataset
from wsdp.utils.train_func import train_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


def benchmark_model(model_name, model_class, processed_data, labels, groups,
                    unique_labels, num_classes, dataset, pad_len,
                    epochs=5, seeds=None, batch_size=32, output_dir=None):
    """Benchmark a single model, return mean/std accuracy."""
    if seeds is None:
        seeds = [42, 123, 456]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracies = []
    n_groups = len(set(groups))
    use_simple = n_groups < 3

    for seed in seeds:
        try:
            train_data, val_data, test_data, train_labels, val_labels, test_labels = \
                _create_data_split(
                    processed_data, labels, groups, 0.3, 0.5, seed, use_simple,
                    dataset=dataset,
                )

            train_data = np.stack(train_data, axis=0)
            val_data = np.stack(val_data, axis=0)
            test_data = np.stack(test_data, axis=0)

            train_dataset = CSIDataset(train_data, train_labels)
            val_dataset = CSIDataset(val_data, val_labels)
            test_dataset = CSIDataset(test_data, test_labels)

            nw = min(os.cpu_count() or 1, 8)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=nw, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=nw, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=nw, shuffle=False)

            # Get input shape from data
            sample_shape = train_data[0].shape  # (T, F*A) after abs
            T = sample_shape[0]
            if len(sample_shape) == 2:
                F_dim = sample_shape[1]
                A_dim = 1
            else:
                F_dim = sample_shape[1]
                A_dim = sample_shape[2]
            input_shape = (T, F_dim, A_dim)

            # Skip models that need special handling
            if model_name in ('ei', 'fewsense'):
                # EI needs domain labels, FewSense needs support set
                # Use standard classifier mode
                pass

            model = model_class(num_classes=num_classes, input_shape=input_shape)
            model = model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

            cp_path = os.path.join(output_dir, f"{model_name}_seed{seed}.pth") if output_dir else None

            train_model(
                model=model, criterion=criterion, optimizer=optimizer,
                scheduler=scheduler, train_loader=train_loader, val_loader=val_loader,
                num_epochs=epochs, device=device, checkpoint_path=cp_path,
                padding_length=pad_len,
            )

            # Load best checkpoint
            if cp_path and os.path.exists(cp_path):
                ckpt = torch.load(cp_path, map_location=device)
                model.load_state_dict(ckpt['model_state_dict'])

            preds, true_labels, acc = _evaluate_model(model, test_loader, device)
            accuracies.append(acc)
            print(f"  [{model_name}] seed={seed}: acc={acc:.4f}")

        except Exception as e:
            print(f"  [{model_name}] seed={seed}: FAILED - {e}")
            traceback.print_exc()
            continue

    if not accuracies:
        return None, None

    return float(np.mean(accuracies)), float(np.std(accuracies))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Path to dataset files')
    parser.add_argument('output_base', help='Output directory')
    parser.add_argument('dataset', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--models', nargs='*', help='Specific models to test (default: all)')
    args = parser.parse_args()

    os.makedirs(args.output_base, exist_ok=True)

    # Load and preprocess once
    params = load_params(args.dataset)
    pad_len = params.get("padding_length", 1500)
    print(f"Loading and preprocessing {args.dataset} data (pad_len={pad_len})...")

    processed_data, labels, groups, unique_labels = _load_and_preprocess(
        args.input_path, args.dataset, pad_len
    )
    num_classes = len(unique_labels)
    print(f"Loaded: {len(processed_data)} samples, {num_classes} classes, "
          f"{len(set(groups))} groups")

    seeds = list(range(42, 42 + args.seeds))
    results = {}

    models_to_test = args.models or sorted(MODEL_REGISTRY.keys())

    for model_name in models_to_test:
        if model_name not in MODEL_REGISTRY:
            print(f"Skipping unknown model: {model_name}")
            continue

        cat, model_class = MODEL_REGISTRY[model_name]
        n_params = None
        try:
            sample = processed_data[0]
            T = sample.shape[0]
            F_dim = sample.shape[1] if len(sample.shape) >= 2 else 1
            A_dim = sample.shape[2] if len(sample.shape) >= 3 else 1
            model_for_count = model_class(num_classes=num_classes, input_shape=(T, F_dim, A_dim))
            n_params = sum(p.numel() for p in model_for_count.parameters()) / 1e6
            del model_for_count
        except:
            pass

        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({cat}) | Params: {n_params:.2f}M" if n_params else
              f"Model: {model_name} ({cat})")
        print(f"{'='*60}")

        model_dir = os.path.join(args.output_base, model_name)
        os.makedirs(model_dir, exist_ok=True)

        mean_acc, std_acc = benchmark_model(
            model_name, model_class, processed_data, labels, groups,
            unique_labels, num_classes, args.dataset, pad_len,
            epochs=args.epochs, seeds=seeds, batch_size=args.batch_size,
            output_dir=model_dir,
        )

        if mean_acc is not None:
            results[model_name] = {
                'model': model_name,
                'dataset': args.dataset,
                'accuracy_mean': round(mean_acc, 4),
                'accuracy_std': round(std_acc, 4),
                'seeds': seeds,
                'params_M': round(n_params, 3) if n_params else None,
                'training_config': {
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'lr': 3e-4,
                    'weight_decay': 1e-3,
                },
                'submitter': 'Official',
                'date': time.strftime('%Y-%m-%d'),
            }
            print(f"\n>>> {model_name}: {mean_acc:.4f} +/- {std_acc:.4f}")

    # Save results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    for name, r in sorted(results.items(), key=lambda x: -x[1]['accuracy_mean']):
        print(f"  {name:25s} {r['accuracy_mean']:.4f} +/- {r['accuracy_std']:.4f}")

    # Save individual submission JSONs
    submissions_dir = os.path.join(os.path.dirname(__file__), '..', 'benchmarks', 'submissions')
    os.makedirs(submissions_dir, exist_ok=True)
    for name, r in results.items():
        fname = f"{args.dataset}_{name}_official.json"
        with open(os.path.join(submissions_dir, fname), 'w') as f:
            json.dump(r, f, indent=2)
    print(f"\nSubmission JSONs written to {submissions_dir}")

    # Save summary
    summary_path = os.path.join(args.output_base, 'benchmark_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == '__main__':
    main()
