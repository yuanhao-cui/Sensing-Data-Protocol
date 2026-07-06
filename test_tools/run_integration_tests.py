#!/usr/bin/env python3
"""
Manual integration runner.

Runs four datasets through four execution paths:
1. pipeline direct call without optional model/preset arguments
2. pipeline direct call with model and algorithm preset arguments
3. CLI call without optional model/preset arguments
4. CLI call with model and algorithm preset arguments

This file is intentionally a script, not a pytest unit test. Run it manually:
    python test_tools/run_integration_tests.py

Any failure is allowed to raise directly so the user can inspect and handle it.
"""

from pathlib import Path
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")

# Make local src/ importable when the script is executed from the repository root.
SRC_PATH = str(Path(__file__).parent.parent / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

DATASETS = ["widar", "gait", "xrf55", "elderAL"]
DATA_ROOT = "./data"
OUTPUT_ROOT = "./test_script_output"
MODEL_NAME = "cnn1dmodel"
ALGORITHM_PRESET = "robust"
NUM_EPOCHS = 3
NUM_SEEDS = 2
TIMEOUT_SECONDS = 600


def _validate_env() -> None:
    """Fail early if the configured model or preset name is invalid."""
    from wsdp.algorithms import list_presets
    from wsdp.models import list_models

    available_models = list_models()
    if MODEL_NAME.lower() not in available_models:
        raise ValueError(
            f"Unknown model '{MODEL_NAME}'. Available models: "
            f"{list(available_models.keys())}"
        )

    available_presets = list_presets()
    if ALGORITHM_PRESET not in available_presets:
        raise ValueError(
            f"Unknown preset '{ALGORITHM_PRESET}'. Available presets: "
            f"{list(available_presets.keys())}"
        )

    print(f"[INFO] Model: {MODEL_NAME}")
    print(f"[INFO] Preset: {ALGORITHM_PRESET}")
    print(f"[INFO] Data root: {DATA_ROOT}")
    print(f"[INFO] Output root: {OUTPUT_ROOT}")
    print(f"[INFO] Epochs: {NUM_EPOCHS}")
    print(f"[INFO] Seeds: {NUM_SEEDS}")


def _dataset_path(dataset: str) -> str:
    return str(Path(DATA_ROOT) / dataset)


def _run_pipeline_basic(dataset: str, output_dir: str) -> None:
    from wsdp import pipeline

    pipeline(
        input_path=_dataset_path(dataset),
        output_folder=output_dir,
        dataset=dataset,
        num_epochs=NUM_EPOCHS,
        num_seeds=NUM_SEEDS,
        use_cache=True,
    )


def _run_pipeline_with_options(dataset: str, output_dir: str) -> None:
    from wsdp import pipeline

    pipeline(
        input_path=_dataset_path(dataset),
        output_folder=output_dir,
        dataset=dataset,
        model_name=MODEL_NAME,
        algorithm_preset=ALGORITHM_PRESET,
        num_epochs=NUM_EPOCHS,
        num_seeds=NUM_SEEDS,
        use_cache=True,
    )


def _build_cli_subprocess_code(argv: list[str]) -> str:
    """Build code that runs the real CLI parser while injecting NUM_SEEDS."""
    return f"""
import sys
sys.path.insert(0, {SRC_PATH!r})
sys.argv = {argv!r}

import matplotlib
matplotlib.use("Agg")

from unittest.mock import patch
from wsdp.cli import main_cli
from wsdp.core import pipeline as real_pipeline


def _patched_pipeline(*args, **kwargs):
    kwargs["num_seeds"] = {NUM_SEEDS!r}
    return real_pipeline(*args, **kwargs)


with patch("wsdp.cli.pipeline", side_effect=_patched_pipeline):
    main_cli()
"""


def _run_cli_basic(dataset: str, output_dir: str) -> None:
    argv = [
        "wsdp",
        "run",
        _dataset_path(dataset),
        output_dir,
        dataset,
        "--epochs",
        str(NUM_EPOCHS),
    ]
    subprocess.run(
        [sys.executable, "-c", _build_cli_subprocess_code(argv)],
        check=True,
        timeout=TIMEOUT_SECONDS,
    )


def _run_cli_with_options(dataset: str, output_dir: str) -> None:
    argv = [
        "wsdp",
        "run",
        _dataset_path(dataset),
        output_dir,
        dataset,
        "--model",
        MODEL_NAME,
        "--algorithm-preset",
        ALGORITHM_PRESET,
        "--epochs",
        str(NUM_EPOCHS),
    ]
    subprocess.run(
        [sys.executable, "-c", _build_cli_subprocess_code(argv)],
        check=True,
        timeout=TIMEOUT_SECONDS,
    )


def main() -> None:
    _validate_env()
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    test_plan = [
        ("pipeline_basic", _run_pipeline_basic),
        ("pipeline_options", _run_pipeline_with_options),
        ("cli_basic", _run_cli_basic),
        ("cli_options", _run_cli_with_options),
    ]

    total = len(DATASETS) * len(test_plan)
    current = 0

    for dataset in DATASETS:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset}")
        print(f"{'=' * 60}")

        for name, func in test_plan:
            current += 1
            out_dir = str(Path(OUTPUT_ROOT) / dataset / name)
            start = time.time()
            print(f"[{current}/{total}] {dataset} / {name} ...", flush=True)
            func(dataset, out_dir)
            elapsed = time.time() - start
            print(f"[{current}/{total}] completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
