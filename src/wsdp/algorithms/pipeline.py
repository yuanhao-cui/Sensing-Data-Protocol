"""Ordered, state-driven algorithm pipeline execution.

This module introduces ``AlgorithmStep``, a small dataclass that describes one
processing step, and ``execute_algorithm_steps``, which runs steps in order
while maintaining a mutable state dictionary. Steps can read from and write to
named state keys, making it easy to route data between heterogeneous stages
without hard-coding a global execution order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .registry import get_algorithm


@dataclass(frozen=True)
class AlgorithmStep:
    """One configurable algorithm step in a modular pipeline.

    Args:
        category: Algorithm category, e.g. ``denoise`` or ``calibrate``.
        method: Method name within the category, e.g. ``wavelet``.
        params: Method-specific keyword arguments.
        input_key: Name of the state entry to use as input (default ``csi``).
        output_key: Name(s) of the state entry to write. Can be a single
            string or a sequence of strings when the step returns multiple
            values.
        enabled: If False, the step is skipped.
    """

    category: str
    method: str
    params: Mapping[str, Any] = field(default_factory=dict)
    input_key: str = "csi"
    output_key: str | Sequence[str] = "csi"
    enabled: bool = True

    @property
    def name(self) -> str:
        """Canonical step identifier used in records and logs."""
        return f"{self.category}:{self.method}"

    @classmethod
    def from_config(
        cls,
        config: AlgorithmStep | Mapping[str, Any] | Sequence[Any],
    ) -> AlgorithmStep:
        """Create a step from a dataclass, mapping, or tuple-like config."""
        if isinstance(config, cls):
            return config

        if isinstance(config, Mapping):
            params: dict[str, Any] = dict(config.get("params", {}))
            for key, value in config.items():
                if key not in {
                    "category",
                    "method",
                    "params",
                    "input_key",
                    "output_key",
                    "enabled",
                }:
                    params[key] = value
            return cls(
                category=str(config["category"]),
                method=str(config["method"]),
                params=params,
                input_key=str(config.get("input_key", "csi")),
                output_key=config.get("output_key", "csi"),
                enabled=bool(config.get("enabled", True)),
            )

        if isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
            if len(config) < 2:
                raise ValueError("step tuple must contain at least category and method")
            tuple_params = config[2] if len(config) > 2 else {}
            return cls(str(config[0]), str(config[1]), dict(tuple_params))

        raise TypeError(f"unsupported step config: {type(config)!r}")


def _assign_output(
    state: dict[str, Any],
    output_key: str | Sequence[str],
    value: Any,
) -> None:
    """Write a step result into the state dictionary."""
    if isinstance(output_key, str):
        state[output_key] = value
        return

    keys = list(output_key)
    if not isinstance(value, (tuple, list)):
        raise ValueError(
            f"step returned a single value but output_key expects {keys}"
        )
    if len(keys) != len(value):
        raise ValueError(
            f"output_key count {len(keys)} does not match result count {len(value)}"
        )
    for key, item in zip(keys, value):
        state[str(key)] = item


def execute_algorithm_steps(
    csi: np.ndarray,
    steps: Sequence[AlgorithmStep | Mapping[str, Any] | Sequence[Any]],
    *,
    dataset: str = "",
    initial_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute configurable algorithm steps and return the full state dict.

    Args:
        csi: Input CSI array.
        steps: Ordered sequence of steps. Each step may be an ``AlgorithmStep``,
            a mapping, or a ``(category, method, params?)`` tuple.
        dataset: Dataset name, passed to steps that request it.
        initial_state: Optional initial state entries (e.g. file metadata).

    Returns:
        State dictionary containing at least ``csi`` plus any named outputs
        produced by the steps.
    """
    state: dict[str, Any] = dict(initial_state or {})
    state.setdefault("csi", csi)

    for raw_step in steps:
        step = AlgorithmStep.from_config(raw_step)
        if not step.enabled:
            continue

        if step.input_key not in state:
            raise KeyError(
                f"step {step.name} missing input_key: {step.input_key}"
            )

        func = get_algorithm(step.category, step.method)
        result = func(
            state[step.input_key],
            dataset=dataset,
            method=step.method,
            **dict(step.params),
        )
        _assign_output(state, step.output_key, result)

    return state
