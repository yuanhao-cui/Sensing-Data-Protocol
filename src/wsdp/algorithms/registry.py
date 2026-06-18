"""
Algorithm Registry for WSDP — Pluggable Architecture

Provides a registry pattern that allows:
1. Easy switching between algorithms
2. Custom algorithm parameters
3. User-defined algorithm implementations
4. Configuration file-based algorithm selection
5. Pipeline presets for common processing chains

Usage:
    # List available algorithms
    >>> from wsdp.algorithms import list_algorithms
    >>> list_algorithms('denoise')
    {'wavelet': <function ...>, 'butterworth': <function ...>, 'savgol': <function ...>}

    # Register a custom algorithm
    >>> from wsdp.algorithms import register_algorithm
    >>> register_algorithm('denoise', 'my_method', my_denoise_func)

    # Load from config file
    >>> from wsdp.algorithms import load_config
    >>> config = load_config('examples/configs/algorithms_config.yaml')

    # Apply a preset
    >>> from wsdp.algorithms import apply_preset
    >>> steps = apply_preset('high_quality')
"""
import json
import importlib
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np


# ============================================================================
# Algorithm Registry
# ============================================================================

# Lazy-loaded function cache
_algorithm_cache: Dict[str, Dict[str, Callable]] = {}

# Registry of algorithm string references (module:function)
_ALGORITHM_REGISTRY: Dict[str, Dict[str, str]] = {
    'denoise': {
        'wavelet': 'wsdp.algorithms.denoising:wavelet_denoise_csi',
        'butterworth': 'wsdp.algorithms.denoising_butterworth:butterworth_denoise',
        'savgol': 'wsdp.algorithms.denoising_butterworth:savgol_denoise',
        'bandpass': 'wsdp.algorithms.denoising_butterworth:butterworth_bandpass',
        'hampel': 'wsdp.algorithms.amplitude:hampel_filter',
    },
    'calibrate': {
        'linear': 'wsdp.algorithms.phase_calibration:phase_calibration',
        'polynomial': 'wsdp.algorithms.phase:polynomial_calibration',
        'stc': 'wsdp.algorithms.phase:stc_calibration',
        'robust': 'wsdp.algorithms.phase:robust_phase_sanitization',
    },
    'normalize': {
        'z-score': 'wsdp.algorithms.amplitude:normalize_amplitude',
        'min-max': 'wsdp.algorithms.amplitude:normalize_amplitude',
        'agc': 'wsdp.algorithms.amplitude:agc_compensate',
    },
    'interpolate': {
        'linear': 'wsdp.algorithms.interpolation:interpolate_grid',
        'cubic': 'wsdp.algorithms.interpolation:interpolate_grid',
        'nearest': 'wsdp.algorithms.interpolation:interpolate_grid',
        'decimate': 'wsdp.algorithms.interpolation:decimate_antialias',
    },
    'extract_features': {
        'doppler': 'wsdp.algorithms.features:doppler_spectrum',
        'entropy': 'wsdp.algorithms.features:entropy_features',
        'ratio': 'wsdp.algorithms.features:csi_ratio',
        'decomposition': 'wsdp.algorithms.features:tensor_decomposition',
        'conjugate_multiply': 'wsdp.algorithms.features:conjugate_multiply',
        'pca_fusion': 'wsdp.algorithms.features:pca_subcarrier_fusion',
    },
    'detect': {
        'activity': 'wsdp.algorithms.detection:detect_activity',
        'change_point': 'wsdp.algorithms.detection:change_point_detection',
    },
    'outliers': {
        'iqr': 'wsdp.algorithms.amplitude:remove_outliers',
        'z-score': 'wsdp.algorithms.amplitude:remove_outliers',
    },
}

# Direct function references for built-in algorithms (populated lazily)
_custom_algorithms: Dict[str, Dict[str, Callable]] = {}


CONFIG_CATEGORIES = frozenset(_ALGORITHM_REGISTRY.keys())
EXECUTION_ORDER = (
    'denoise',
    'outliers',
    'calibrate',
    'normalize',
    'interpolate',
    'extract_features',
    'detect',
)


# ============================================================================
# Core Registry Functions
# ============================================================================

def _resolve_algorithm(ref: str) -> Callable:
    """Resolve a 'module:function' string reference to a callable."""
    module_path, func_name = ref.rsplit(':', 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def _category_exists(category: str) -> bool:
    """Return True when a category has built-in or custom algorithms."""
    return category in _ALGORITHM_REGISTRY or category in _custom_algorithms


def _ensure_category(category: str) -> None:
    """Ensure a category exists in the registry."""
    if not _category_exists(category):
        raise ValueError(
            f"Unknown algorithm category '{category}'. "
            f"Available: {list_algorithms().keys()}"
        )


def _custom_algorithm_refs(category: str) -> Dict[str, str]:
    """Return display references for custom algorithms in one category."""
    return {
        name: f"{func.__module__}:{func.__name__}"
        for name, func in _custom_algorithms.get(category, {}).items()
    }


def _available_algorithms(category: str) -> Dict[str, str]:
    """Return built-in and custom algorithm references for one category."""
    result = {}
    result.update(_ALGORITHM_REGISTRY.get(category, {}))
    result.update(_custom_algorithm_refs(category))
    return result


def register_algorithm(
    category: str,
    name: str,
    func: Callable,
) -> None:
    """
    Register a custom algorithm.

    Allows users to add their own algorithm implementations to the registry,
    making them available through the unified API.

    Args:
        category: Algorithm category ('denoise', 'calibrate', 'normalize',
            'interpolate', 'extract_features', 'detect', 'outliers')
        name: Algorithm name (used as method= parameter)
        func: Callable that implements the algorithm

    Raises:
        ValueError: If category is not a known category

    Examples:
        >>> def my_denoise(csi, strength=1.0, **kwargs):
        ...     # Custom denoising logic
        ...     return denoised_csi
        >>> register_algorithm('denoise', 'my_method', my_denoise)

        >>> # Now usable via unified API
        >>> from wsdp.algorithms import denoise
        >>> denoised = denoise(csi, method='my_method', strength=2.0)
    """
    if category not in _custom_algorithms:
        _custom_algorithms[category] = {}
    _custom_algorithms[category][name] = func


def unregister_algorithm(category: str, name: str) -> bool:
    """
    Unregister a custom algorithm.

    Only custom (user-registered) algorithms can be unregistered.
    Built-in algorithms cannot be removed.

    Args:
        category: Algorithm category
        name: Algorithm name

    Returns:
        True if algorithm was removed, False if not found

    Raises:
        ValueError: If trying to unregister a built-in algorithm
    """
    # Check if it's a built-in algorithm
    if category in _ALGORITHM_REGISTRY and name in _ALGORITHM_REGISTRY[category]:
        raise ValueError(
            f"Cannot unregister built-in algorithm '{category}:{name}'. "
            f"Use override instead by registering a replacement."
        )

    if category in _custom_algorithms and name in _custom_algorithms[category]:
        del _custom_algorithms[category][name]
        return True
    return False


def get_algorithm(category: str, name: str) -> Callable:
    """
    Get an algorithm function by category and name.

    Searches custom algorithms first, then built-in registry.

    Args:
        category: Algorithm category
        name: Algorithm name

    Returns:
        The algorithm function

    Raises:
        ValueError: If category or name not found

    Examples:
        >>> func = get_algorithm('denoise', 'butterworth')
        >>> result = func(csi, order=5, cutoff=0.3)
    """
    # Check custom algorithms first (allows overriding built-ins)
    if category in _custom_algorithms and name in _custom_algorithms[category]:
        return _custom_algorithms[category][name]

    # Check built-in registry
    if category in _ALGORITHM_REGISTRY and name in _ALGORITHM_REGISTRY[category]:
        ref = _ALGORITHM_REGISTRY[category][name]
        return _resolve_algorithm(ref)

    # Build error message with available options
    available = list_algorithms(category) if _category_exists(category) else {}
    raise ValueError(
        f"Unknown algorithm '{name}' in category '{category}'. "
        f"Available: {list(available.keys())}"
    )


def list_algorithms(category: Optional[str] = None) -> Dict[str, Any]:
    """
    List available algorithms.

    Args:
        category: If provided, list algorithms in this category only.
            If None, return all categories with their algorithms.

    Returns:
        If category is None: dict mapping category names to list of algorithm names
        If category is provided: dict mapping algorithm names to their string references

    Examples:
        >>> list_algorithms()
        {'denoise': ['wavelet', 'butterworth', 'savgol'], 'calibrate': [...]}

        >>> list_algorithms('denoise')
        {'wavelet': 'wsdp.algorithms.denoising:wavelet_denoise_csi', ...}
    """
    if category is not None:
        return _available_algorithms(category)

    all_categories = set(_ALGORITHM_REGISTRY.keys()) | set(_custom_algorithms.keys())
    return {cat: list(list_algorithms(cat).keys()) for cat in sorted(all_categories)}


def is_registered(category: str, name: str) -> bool:
    """
    Check if an algorithm is registered.

    Args:
        category: Algorithm category
        name: Algorithm name

    Returns:
        True if the algorithm exists
    """
    if category in _custom_algorithms and name in _custom_algorithms[category]:
        return True
    if category in _ALGORITHM_REGISTRY and name in _ALGORITHM_REGISTRY[category]:
        return True
    return False


# ============================================================================
# Pipeline Presets
# ============================================================================

PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    'high_quality': {
        'denoise': {'method': 'butterworth', 'order': 5, 'cutoff': 0.3},
        'calibrate': {'method': 'stc'},
        'normalize': {'method': 'z-score'},
    },
    'fast': {
        'denoise': {'method': 'savgol', 'window_length': 7, 'polyorder': 3},
        'calibrate': {'method': 'linear'},
        'normalize': {'method': 'min-max'},
    },
    'robust': {
        'denoise': {'method': 'wavelet'},
        'calibrate': {'method': 'robust'},
        'normalize': {'method': 'z-score'},
    },
    'gesture_recognition': {
        'denoise': {'method': 'butterworth', 'order': 4, 'cutoff': 0.25},
        'calibrate': {'method': 'stc'},
        'normalize': {'method': 'z-score'},
        'interpolate': {'method': 'cubic', 'target_K': 30},
    },
    'activity_detection': {
        'denoise': {'method': 'savgol', 'window_length': 11, 'polyorder': 3},
        'calibrate': {'method': 'polynomial', 'degree': 2},
        'normalize': {'method': 'z-score'},
    },
    'localization': {
        'denoise': {'method': 'wavelet'},
        'calibrate': {'method': 'robust'},
        'normalize': {'method': 'z-score'},
        'interpolate': {'method': 'cubic', 'target_K': 64},
    },
}


def register_preset(name: str, steps: Dict[str, Dict[str, Any]]) -> None:
    """
    Register a custom pipeline preset.

    Args:
        name: Preset name
        steps: Dictionary mapping category names to parameter dicts.
            Each dict must contain 'method' key.

    Examples:
        >>> register_preset('my_preset', {
        ...     'denoise': {'method': 'butterworth', 'order': 3},
        ...     'calibrate': {'method': 'linear'},
        ... })
    """
    for category, params in steps.items():
        if 'method' not in params:
            raise ValueError(
                f"Preset step '{category}' must include 'method' key. Got: {params}"
            )
    PRESETS[name] = steps


def apply_preset(name: str) -> Dict[str, Dict[str, Any]]:
    """
    Get a pipeline preset configuration.

    Args:
        name: Preset name (see PRESETS for available presets)

    Returns:
        Dictionary mapping category names to their parameter dicts

    Raises:
        ValueError: If preset name not found

    Examples:
        >>> steps = apply_preset('high_quality')
        >>> steps
        {
            'denoise': {'method': 'butterworth', 'order': 5, 'cutoff': 0.3},
            'calibrate': {'method': 'stc'},
            'normalize': {'method': 'z-score'},
        }
    """
    if name not in PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. "
            f"Available: {list(PRESETS.keys())}"
        )
    return PRESETS[name].copy()


def list_presets() -> Dict[str, list]:
    """
    List all available presets and their processing steps.

    Returns:
        Dictionary mapping preset names to list of processing step names

    Examples:
        >>> list_presets()
        {'high_quality': ['denoise', 'calibrate', 'normalize'], ...}
    """
    return {name: list(steps.keys()) for name, steps in PRESETS.items()}


def execute_pipeline(csi, steps: Dict[str, Dict[str, Any]]) -> 'np.ndarray':
    """
    Execute a processing pipeline on CSI data.

    Applies each processing step in order (denoise → calibrate → normalize → ...).

    Args:
        csi: Input CSI array of shape (T, F, A)
        steps: Pipeline steps from apply_preset() or config file

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

    result = csi.copy()

    for category in EXECUTION_ORDER:
        if category in steps:
            params = steps[category].copy()
            method = params.pop('method')
            func = get_algorithm(category, method)

            if category == 'extract_features':
                # extract_features returns a dict
                features_result = func(result, **params)
                result = features_result  # Pass through for chaining
            elif category == 'detect':
                # detect returns boolean arrays
                detection_result = func(result, **params)
                result = detection_result
            else:
                result = func(result, **params)

    return result


# ============================================================================
# Configuration File Support
# ============================================================================

def _load_raw_config(config_path: Path) -> Dict[str, Any]:
    """Read a YAML or JSON config file into a raw dictionary."""
    suffix = config_path.suffix.lower()
    if suffix in ('.yaml', '.yml'):
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config files. "
                "Install with: pip install pyyaml"
            )
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    if suffix == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    raise ValueError(
        f"Unsupported config format '{suffix}'. "
        f"Supported: .yaml, .yml, .json"
    )


def _parse_step_config(category: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one category config into the internal step format."""
    if not isinstance(config, dict):
        raise ValueError(
            f"Category '{category}' config must be a dict, got {type(config)}"
        )

    method = config.get('method')
    if not method:
        raise ValueError(
            f"Category '{category}' must specify 'method'. Got: {config}"
        )

    if 'params' in config:
        params = config['params'] or {}
    else:
        params = {k: v for k, v in config.items() if k != 'method'}
    return {'method': method, **params}


def _config_output_payload(steps: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Convert internal steps into the persisted config file shape."""
    output = {}
    for category, params in steps.items():
        step_params = params.copy()
        method = step_params.pop('method')
        output[category] = {'method': method}
        if step_params:
            output[category]['params'] = step_params
    return output


def load_config(config_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    Load algorithm configuration from a YAML or JSON file.

    Supports two formats:
    1. Algorithm pipeline config (denoise, calibrate, etc.)
    2. Named preset config with 'preset' key

    Args:
        config_path: Path to configuration file (.yaml, .yml, or .json)

    Returns:
        Dictionary of pipeline steps

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config format is invalid

    Examples:
        YAML format:
        ```yaml
        denoise:
          method: butterworth
          params:
            order: 5
            cutoff: 0.3
        calibrate:
          method: polynomial
          params:
            degree: 3
        normalize:
          method: z-score
        ```

        JSON format:
        ```json
        {
          "denoise": {
            "method": "butterworth",
            "params": {"order": 5, "cutoff": 0.3}
          }
        }
        ```

        Preset reference:
        ```yaml
        preset: high_quality
        ```
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return _parse_config(_load_raw_config(config_path))


def _parse_config(raw_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Parse raw config dict into pipeline steps format."""
    if not isinstance(raw_config, dict):
        raise ValueError("Config must be a dictionary")

    # Handle preset reference
    if 'preset' in raw_config:
        preset_name = raw_config['preset']
        steps = apply_preset(preset_name)
        # Allow overrides
        overrides = {k: v for k, v in raw_config.items()
                     if k not in ('preset',) and isinstance(v, dict)}
        steps.update(overrides)
        return steps

    steps = {}
    for category, config in raw_config.items():
        if category not in CONFIG_CATEGORIES:
            # Skip unknown keys silently (could be metadata)
            continue
        steps[category] = _parse_step_config(category, config)
    return steps


def save_config(
    steps: Dict[str, Dict[str, Any]],
    config_path: Union[str, Path],
    format: str = 'yaml',
) -> None:
    """
    Save algorithm configuration to a file.

    Args:
        steps: Pipeline steps dictionary
        config_path: Output file path
        format: Output format ('yaml' or 'json')

    Examples:
        >>> from wsdp.algorithms import save_config, apply_preset
        >>> steps = apply_preset('high_quality')
        >>> save_config(steps, 'my_config.yaml')
    """
    config_path = Path(config_path)

    output = _config_output_payload(steps)

    if format == 'yaml':
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required: pip install pyyaml")
        with open(config_path, 'w') as f:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False)
    elif format == 'json':
        with open(config_path, 'w') as f:
            json.dump(output, f, indent=2)
    else:
        raise ValueError(f"Unsupported format '{format}'. Use 'yaml' or 'json'.")


# ============================================================================
# Convenience: Algorithm Info
# ============================================================================

def algorithm_info(category: str, name: str) -> Dict[str, Any]:
    """
    Get detailed information about an algorithm.

    Args:
        category: Algorithm category
        name: Algorithm name

    Returns:
        Dictionary with algorithm metadata (docstring, module, signature)
    """
    func = get_algorithm(category, name)
    import inspect

    return {
        'name': name,
        'category': category,
        'module': func.__module__,
        'function': func.__name__,
        'docstring': (func.__doc__ or '').strip(),
        'signature': str(inspect.signature(func)),
        'is_custom': (category in _custom_algorithms and
                      name in _custom_algorithms.get(category, {})),
    }
