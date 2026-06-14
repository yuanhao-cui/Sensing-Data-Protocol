"""Lightweight pipeline-run record data structure.

Provides pure-data classes and JSON persistence for capturing a single
pipeline() execution: dataset, reader, processor configuration, model,
and per-seed train/val/test accuracies.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


CONFIGURABLE_ALGORITHM_ORDER = [
    "denoise",
    "outliers",
    "calibrate",
    "normalize",
    "interpolate",
    "extract_features",
    "detect",
]


@dataclass
class SeedRecord:
    """Metrics for a single random seed."""

    seed: int
    train_acc: float  # 0-1
    val_acc: float    # 0-1
    test_acc: float   # 0-1


@dataclass
class PipelineRecord:
    """Complete record of one pipeline() invocation."""

    dataset: str
    total_samples: int
    reader: str
    processor_type: str  # "BaseProcessor" | "ConfigurableProcessor"
    algorithm_modules: List[str]
    model: str
    seeds: List[SeedRecord]
    best_test_acc: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert to a plain dictionary suitable for JSON."""
        return asdict(self)

    def save_json(self, path: str) -> None:
        """Persist as a pretty-printed JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            f.write("\n")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineRecord":
        """Reconstruct from a plain dictionary."""
        record_data = dict(data)
        seeds_data = record_data.pop("seeds", [])
        seeds = [SeedRecord(**s) for s in seeds_data]
        if "algorithm_modules" not in record_data and "processor_steps" in record_data:
            record_data["algorithm_modules"] = algorithm_modules_from_steps(
                record_data.pop("processor_steps")
            )
        else:
            record_data.pop("processor_steps", None)
        return cls(seeds=seeds, **record_data)

    @classmethod
    def load_json(cls, path: str) -> "PipelineRecord":
        """Load from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def persist_pipeline_record(
    output_folder: str,
    dataset: str,
    total_samples: int,
    reader_name: str,
    processor_type: str,
    algorithm_modules: List[str],
    model: str,
    seed_records: List[SeedRecord],
) -> Path:
    """Build a PipelineRecord from raw fields and save it to JSON.

    This thin wrapper keeps core.py decoupled from PipelineRecord internals.
    """
    best = max(r.test_acc for r in seed_records) if seed_records else 0.0
    record = PipelineRecord(
        dataset=dataset,
        total_samples=total_samples,
        reader=reader_name,
        processor_type=processor_type,
        algorithm_modules=algorithm_modules,
        model=model,
        seeds=seed_records,
        best_test_acc=best,
    )
    target = Path(output_folder) / "pipeline_record.json"
    record.save_json(str(target))
    return target


def algorithm_modules_from_steps(processor_steps: Optional[Dict[str, Any]]) -> List[str]:
    """Return configured algorithm module names with their method, e.g. denoise:butterworth."""
    if processor_steps is None:
        return []
    ordered = [name for name in CONFIGURABLE_ALGORITHM_ORDER if name in processor_steps]
    extras = [name for name in processor_steps if name not in ordered]

    def _format(name: str) -> str:
        step = processor_steps.get(name)
        if isinstance(step, dict):
            method = step.get("method")
            if method:
                return f"{name}:{method}"
        return name

    return [_format(name) for name in ordered + extras]
