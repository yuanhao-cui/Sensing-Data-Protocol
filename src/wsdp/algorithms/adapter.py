"""Adapters that wrap existing function-style algorithms as standard callables.

The existing algorithm implementations are plain functions with varying
signatures. These adapters let us expose them through a uniform
``(csi, *, dataset, **params)`` interface used by ``AlgorithmStep``.
"""

import inspect
import logging
from typing import Any, Callable, Dict

import numpy as np

logger = logging.getLogger(__name__)


class FunctionAlgorithm:
    """Wrap a plain function so it matches the unified algorithm interface.

    Args:
        func: The algorithm function to wrap.
        method: Default method name to inject when ``pass_method`` is True and
            the caller does not provide a ``method`` argument.
        pass_dataset: If True, inject ``dataset`` into the call kwargs.
        pass_method: If True, inject the registered ``method`` name as
            ``method`` into the call kwargs. This is useful for shared
            implementations such as ``normalize_amplitude`` that dispatch
            internally on the method name.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        method: str = "",
        pass_dataset: bool = False,
        pass_method: bool = False,
    ):
        if not callable(func):
            raise TypeError("func must be callable")
        self.func = func
        self.method = method
        self.pass_dataset = pass_dataset
        self.pass_method = pass_method

    def __call__(
        self,
        csi: np.ndarray,
        *,
        dataset: str = "",
        method: str = "",
        **params: Any,
    ) -> Any:
        kwargs = dict(params)
        if self.pass_dataset:
            kwargs["dataset"] = dataset
        if self.pass_method:
            kwargs["method"] = method or self.method
        kwargs = filter_kwargs(self.func, kwargs)
        return self.func(csi, **kwargs)

    @property
    def __module__(self) -> str:
        return self.func.__module__

    @property
    def __name__(self) -> str:
        return self.func.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.func.__module__}:{self.func.__name__})"


def filter_kwargs(func: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop kwargs that the callable cannot accept unless it has **kwargs."""
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return kwargs

    accepts_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_var_keyword:
        return kwargs

    valid = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    dropped = [k for k in kwargs if k not in valid]
    if dropped:
        logger.debug(
            "Dropped parameters not accepted by %s.%s: %s",
            func.__module__,
            func.__name__,
            dropped,
        )
    return filtered
