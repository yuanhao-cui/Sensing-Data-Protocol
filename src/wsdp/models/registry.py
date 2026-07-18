"""Pluggable model registry for WSDP."""

from typing import Any, Dict, Type
import torch.nn as nn

from wsdp.interfaces import ModelBuilder
from .builders import ClassModelBuilder

# Global registry: {lowercase_name: (category, builder)}
MODEL_REGISTRY: Dict[str, tuple] = {}


def register_model(category: str, name: str, model_class: Type[nn.Module]) -> None:
    """Register a model class in the global registry.

    Args:
        category: Model category (baseline, mainstream, sota).
        name: Human-readable model name.
        model_class: nn.Module subclass.
    """
    key = name.lower()
    if key in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' already registered.")
    builder = ClassModelBuilder(name, model_class)
    MODEL_REGISTRY[key] = (category.lower(), builder)


def register_model_builder(category: str, name: str, builder: ModelBuilder) -> None:
    """Register a custom model builder.

    Args:
        category: Model category.
        name: Human-readable model name.
        builder: ``ModelBuilder`` instance.
    """
    key = name.lower()
    if key in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' already registered.")
    MODEL_REGISTRY[key] = (category.lower(), builder)


def unregister_model(name: str) -> bool:
    """Remove a model registration, returning whether it existed."""
    key = name.lower()
    return MODEL_REGISTRY.pop(key, None) is not None


def get_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a registered model by name.

    Args:
        name: Model name (case-insensitive).
        **kwargs: Passed to model constructor (must include num_classes, input_shape).

    Returns:
        Instantiated nn.Module.

    Raises:
        KeyError: If model name not found.
    """
    key = name.lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    _, builder = MODEL_REGISTRY[key]
    return builder.build(**kwargs)


def create_model(name: str, num_classes: int, input_shape: tuple, **kwargs) -> nn.Module:
    """Create a model by name with unified interface.

    Args:
        name: Model name from registry (case-insensitive).
        num_classes: Number of output classes.
        input_shape: (T, F, A) tuple — time steps, frequency bins, antennas.
        **kwargs: Extra model-specific hyperparameters.

    Returns:
        nn.Module instance.
    """
    return get_model(name, num_classes=num_classes, input_shape=input_shape, **kwargs)


def list_models(category: str | None = None) -> Dict[str, str]:
    """List all registered models, optionally filtered by category.

    Args:
        category: Filter by category name (baseline, mainstream, sota).

    Returns:
        Dict mapping model names to their categories.
    """
    result = {}
    for name, (cat, _) in MODEL_REGISTRY.items():
        if category is None or cat == category.lower():
            result[name] = cat
    return result


def get_model_builder(name: str) -> ModelBuilder:
    """Return the builder registered under ``name``."""
    key = name.lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    _, builder = MODEL_REGISTRY[key]
    return builder
