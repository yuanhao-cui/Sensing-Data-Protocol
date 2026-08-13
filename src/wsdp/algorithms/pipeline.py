"""Configurable algorithm pipeline execution.

This module provides ``AlgorithmStep``, a small dataclass describing one
processing step, plus helpers that turn the user-facing category-dict config
into an ordered step list and execute it as a simple chain::

    result = step_n(... step_2(step_1(csi)))

Dependency direction is one-way: this module depends on ``registry``
(for ``get_algorithm`` and ``CATEGORY_ORDER``), never the reverse.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from wsdp.dataset_policy import real_if_negligible_imaginary
from .registry import CATEGORY_ORDER, check_algorithm_compatibility, get_algorithm


@dataclass(frozen=True)
class AlgorithmStep:
    """One configurable algorithm step in a modular pipeline.

    Args:
        category: Algorithm category, e.g. ``denoise`` or ``calibrate``.
        method: Method name within the category, e.g. ``wavelet``.
        params: Method-specific keyword arguments.
    """

    category: str
    method: str
    params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Canonical step identifier used in records and logs."""
        return f"{self.category}:{self.method}"

    @classmethod
    def from_config(cls, config: AlgorithmStep | Mapping[str, Any]) -> AlgorithmStep:
        """Create a step from an ``AlgorithmStep`` or a mapping config.

        Mapping keys ``category`` and ``method`` are required; ``params`` is
        optional. Any extra keys are merged into ``params`` for convenience.
        """
        if isinstance(config, cls):
            return config

        if isinstance(config, Mapping):
            params: dict[str, Any] = dict(config.get("params", {}))
            for key, value in config.items():
                if key not in {"category", "method", "params"}:
                    params[key] = value
            return cls(
                category=str(config["category"]),
                method=str(config["method"]),
                params=params,
            )

        raise TypeError(f"unsupported step config: {type(config)!r}")


def build_steps_from_config(steps: Dict[str, Dict[str, Any]]) -> List[AlgorithmStep]:
    """Convert a category-dict pipeline config into an ordered step list.

    Args:
        steps: Mapping from category name to ``{"method": ..., **params}``.

    Returns:
        Ordered list of ``AlgorithmStep`` instances following
        ``CATEGORY_ORDER``; user-defined categories are appended in
        insertion order.
    """
    ordered_categories = CATEGORY_ORDER + [
        category for category in steps if category not in CATEGORY_ORDER
    ]
    ordered = []
    for category in ordered_categories:
        if category not in steps:
            continue
        params = steps[category].copy()
        method = params.pop("method")
        ordered.append(AlgorithmStep(category=category, method=method, params=params))

    return ordered


def execute_algorithm_steps(
    csi: np.ndarray,
    steps: Sequence[AlgorithmStep | Mapping[str, Any]],
    *,
    dataset: str = "",
) -> np.ndarray:
    """Execute algorithm steps in order, chaining each output to the next input.

    Args:
        csi: Input CSI array.
        steps: Ordered sequence of ``AlgorithmStep`` objects or mapping configs.
        dataset: Dataset name, passed to steps that request it.

    Returns:
        The output of the last step (or ``csi`` unchanged when ``steps`` is
        empty).
    """
    result = csi
    for raw_step in steps:
        step = AlgorithmStep.from_config(raw_step)
        check_algorithm_compatibility(step.category, step.method, dataset)
        func = get_algorithm(step.category, step.method)
        result = func(result, dataset=dataset, method=step.method, **dict(step.params))
    return result


def execute_pipeline(csi, steps: Dict[str, Dict[str, Any]],
                     dataset: Optional[str] = None) -> Any:
    """
    Execute a processing pipeline on CSI data.

    Applies each processing step in order (denoise → outliers → calibrate → ...).

    Args:
        csi: Input CSI array of shape (T, F, A)
        steps: Pipeline steps from apply_preset() or config file
        dataset: Optional dataset name for dataset-aware steps and
            amplitude-primary cleanup.

    Returns:
        Processed CSI array

    Examples:
        >>> from wsdp.algorithms import apply_preset, execute_pipeline
        >>> steps = apply_preset('high_quality')
        >>> processed = execute_pipeline(csi, steps)

        >>> # Or with custom steps
        >>> steps = {
        ...     'denoise': {'method': 'butterworth', 'order': 5},
        ...     'calibrate': {'method': 'stc'},
        ... }
        >>> processed = execute_pipeline(csi, steps)
    """
    result = execute_algorithm_steps(
        csi, build_steps_from_config(steps), dataset=dataset or ""
    )
    return real_if_negligible_imaginary(result, dataset or "")
