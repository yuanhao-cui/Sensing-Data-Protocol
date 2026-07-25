"""Pipeline configuration parsing and normalization.

This module converts the user-facing category-dict configuration format into
an ordered list of ``AlgorithmStep`` objects. Keeping the conversion in one
place avoids duplication between the registry, processors, and CLI.
"""

from typing import Any, Dict, List

from wsdp.algorithms.pipeline import AlgorithmStep


# Default execution order for built-in algorithm categories. User-defined
# categories are appended in insertion order.
CATEGORY_ORDER = [
    "denoise",
    "outliers",
    "calibrate",
    "normalize",
    "interpolate",
    "extract_features",
    "detect",
]


def build_steps_from_config(steps: Dict[str, Dict[str, Any]]) -> List[AlgorithmStep]:
    """Convert a category-dict pipeline config into an ordered step list.

    Args:
        steps: Mapping from category name to ``{"method": ..., **params}``.

    Returns:
        Ordered list of ``AlgorithmStep`` instances.
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
        ordered.append(
            AlgorithmStep(category=category, method=method, params=params)
        )

    return ordered
