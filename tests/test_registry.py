"""Tests for the WSDP algorithm registry — pluggable architecture."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from wsdp.algorithms import (
    # Registry API
    register_algorithm,
    unregister_algorithm,
    get_algorithm,
    list_algorithms,
    is_registered,
    algorithm_info,
    load_config,
    save_config,
    apply_preset,
    register_preset,
    list_presets,
    execute_pipeline,
    PRESETS,
    # Unified API
    denoise,
    calibrate,
    normalize,
    interpolate,
    extract_features,
    _call_algorithm,
)
from wsdp.algorithms.registry import _custom_algorithms


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def csi_complex():
    """Generate synthetic complex CSI data."""
    np.random.seed(42)
    T, F, A = 100, 30, 3
    return np.random.randn(T, F, A) + 1j * np.random.randn(T, F, A)


# ============================================================================
# Registry Tests
# ============================================================================
class TestRegistry:
    def test_list_all_algorithms(self):
        all_algos = list_algorithms()
        assert 'denoise' in all_algos
        assert 'calibrate' in all_algos
        assert 'normalize' in all_algos
        assert 'interpolate' in all_algos
        assert 'extract_features' in all_algos
        assert 'detect' in all_algos

    def test_list_category(self):
        denoise_algos = list_algorithms('denoise')
        assert 'wavelet' in denoise_algos
        assert 'butterworth' in denoise_algos
        assert 'savgol' in denoise_algos

    def test_list_calibrate(self):
        calib_algos = list_algorithms('calibrate')
        assert 'linear' in calib_algos
        assert 'polynomial' in calib_algos
        assert 'stc' in calib_algos
        assert 'robust' in calib_algos

    def test_get_algorithm_denoise(self):
        func = get_algorithm('denoise', 'wavelet')
        assert callable(func)

    def test_get_algorithm_calibrate(self):
        func = get_algorithm('calibrate', 'stc')
        assert callable(func)

    def test_get_algorithm_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            get_algorithm('denoise', 'nonexistent')

    def test_is_registered_builtin(self):
        assert is_registered('denoise', 'wavelet')
        assert is_registered('calibrate', 'linear')
        assert not is_registered('denoise', 'nonexistent')


# ============================================================================
# Custom Algorithm Registration Tests
# ============================================================================
class TestCustomAlgorithms:
    def test_register_custom_denoise(self, csi_complex):
        """Register a custom denoise algorithm and use it."""
        def my_denoise(csi, strength=1.0, **kwargs):
            return csi * strength

        register_algorithm('denoise', 'my_test_method', my_denoise)

        assert is_registered('denoise', 'my_test_method')

        # Use via unified API
        result = denoise(csi_complex, method='my_test_method', strength=0.5)
        expected = csi_complex * 0.5

        np.testing.assert_allclose(result, expected)

        # Cleanup
        unregister_algorithm('denoise', 'my_test_method')

        assert not is_registered('denoise', 'my_test_method')

    def test_register_custom_calibrate(self, csi_complex):
        """Register a custom calibrate algorithm."""
        def identity_calibrate(csi, **kwargs):
            return csi.copy()

        register_algorithm('calibrate', 'identity', identity_calibrate)
        result = calibrate(csi_complex, method='identity')
        np.testing.assert_array_equal(result, csi_complex)

        unregister_algorithm('calibrate', 'identity')

    def test_custom_overrides_builtin(self, csi_complex):
        """Custom algorithm can override built-in via priority."""
        original_func = get_algorithm('denoise', 'wavelet')

        def custom_wavelet(csi, **kwargs):
            return csi * 0.99

        register_algorithm('denoise', 'wavelet', custom_wavelet)

        # Should now get custom version (custom takes priority)
        result = denoise(csi_complex, method='wavelet')

        np.testing.assert_allclose(result, csi_complex * 0.99)

        # Cleanup - custom is in _custom_algorithms, so can remove it
        # (it shadows built-in but doesn't replace it)
        del _custom_algorithms['denoise']['wavelet']

        # After cleanup, should get built-in again
        result2 = denoise(csi_complex, method='wavelet')

        assert result2.shape == csi_complex.shape

        # Should NOT be 0.99 * csi anymore
        assert not np.allclose(result2, csi_complex * 0.99)

    def test_unregister_builtin_raises(self):
        """Cannot unregister built-in algorithms."""
        with pytest.raises(ValueError, match="Cannot unregister built-in"):
            unregister_algorithm('denoise', 'wavelet')

    def test_unregister_nonexistent_returns_false(self):
        result = unregister_algorithm('denoise', 'never_registered')
        assert result is False

    def test_algorithm_info(self):
        info = algorithm_info('denoise', 'butterworth')
        assert info['name'] == 'butterworth'
        assert info['category'] == 'denoise'
        assert 'order' in info['signature']
        assert info['is_custom'] is False


# ============================================================================
# Config File Tests
# ============================================================================
class TestConfigFiles:
    def test_load_yaml_config(self):
        """Load algorithm config from YAML file."""
        yaml_content = """
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
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_config(f.name)

        os.unlink(f.name)

        assert config['denoise']['method'] == 'butterworth'
        assert config['denoise']['order'] == 5
        assert config['denoise']['cutoff'] == 0.3
        assert config['calibrate']['method'] == 'polynomial'
        assert config['calibrate']['degree'] == 3

    def test_load_json_config(self):
        """Load algorithm config from JSON file."""
        config_data = {
            "denoise": {
                "method": "savgol",
                "params": {"window_length": 11, "polyorder": 3}
            },
            "calibrate": {
                "method": "linear"
            }
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            f.flush()
            config = load_config(f.name)

        os.unlink(f.name)

        assert config['denoise']['method'] == 'savgol'
        assert config['denoise']['window_length'] == 11

    def test_load_preset_from_config(self):
        """Load a preset reference from config file."""
        yaml_content = """
preset: high_quality
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_config(f.name)

        os.unlink(f.name)

        assert config['denoise']['method'] == 'butterworth'
        assert config['calibrate']['method'] == 'stc'

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config('/nonexistent/path/config.yaml')

    def test_load_unsupported_format_raises(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("some text")
            f.flush()
            with pytest.raises(ValueError, match="Unsupported config format"):
                load_config(f.name)
        os.unlink(f.name)

    def test_save_config_yaml(self):
        """Save config to YAML file."""
        steps = {
            'denoise': {'method': 'butterworth', 'order': 5},
            'calibrate': {'method': 'linear'},
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            save_config(steps, f.name, format='yaml')

            # Verify it can be loaded back
            loaded = load_config(f.name)

        os.unlink(f.name)

        assert loaded['denoise']['method'] == 'butterworth'

    def test_save_config_json(self):
        """Save config to JSON file."""
        steps = {
            'denoise': {'method': 'savgol', 'window_length': 7},
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            save_config(steps, f.name, format='json')

            with open(f.name, 'r') as rf:
                loaded_raw = json.load(rf)

        os.unlink(f.name)

        assert loaded_raw['denoise']['method'] == 'savgol'
        assert loaded_raw['denoise']['params']['window_length'] == 7


# ============================================================================
# Preset Tests
# ============================================================================
class TestPresets:
    def test_list_presets(self):
        presets = list_presets()
        assert 'high_quality' in presets
        assert 'fast' in presets
        assert 'robust' in presets
        assert 'gesture_recognition' in presets
        assert 'activity_detection' in presets
        assert 'localization' in presets

    def test_apply_high_quality_preset(self):
        steps = apply_preset('high_quality')
        assert steps['denoise']['method'] == 'butterworth'
        assert steps['calibrate']['method'] == 'stc'
        assert steps['normalize']['method'] == 'z-score'

    def test_apply_fast_preset(self):
        steps = apply_preset('fast')
        assert steps['denoise']['method'] == 'savgol'
        assert steps['calibrate']['method'] == 'linear'
        assert steps['normalize']['method'] == 'min-max'

    def test_apply_robust_preset(self):
        steps = apply_preset('robust')
        assert steps['denoise']['method'] == 'wavelet'
        assert steps['calibrate']['method'] == 'robust'

    def test_apply_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            apply_preset('nonexistent')

    def test_register_custom_preset(self):
        register_preset('my_custom', {
            'denoise': {'method': 'wavelet'},
            'calibrate': {'method': 'linear'},
        })

        steps = apply_preset('my_custom')

        assert steps['denoise']['method'] == 'wavelet'

        # Cleanup
        del PRESETS['my_custom']

    def test_register_preset_without_method_raises(self):
        with pytest.raises(ValueError, match="must include 'method'"):
            register_preset('bad_preset', {
                'denoise': {'order': 5},  # missing 'method'
            })


# ============================================================================
# Pipeline Execution Tests
# ============================================================================
class TestPipelineExecution:
    def test_execute_simple_pipeline(self, csi_complex):
        """Execute a 2-step pipeline."""
        steps = {
            'denoise': {'method': 'savgol', 'window_length': 11, 'polyorder': 3},
            'calibrate': {'method': 'linear'},
        }
        result = execute_pipeline(csi_complex, steps)
        assert result.shape == csi_complex.shape

    def test_execute_full_pipeline(self, csi_complex):
        """Execute a 3-step pipeline: denoise → calibrate → normalize."""
        steps = {
            'denoise': {'method': 'butterworth', 'order': 4, 'cutoff': 0.3},
            'calibrate': {'method': 'linear'},
            'normalize': {'method': 'z-score'},
        }
        result = execute_pipeline(csi_complex, steps)
        assert result.shape == csi_complex.shape

    def test_execute_preset_pipeline(self, csi_complex):
        """Execute a preset pipeline."""
        steps = apply_preset('fast')
        result = execute_pipeline(csi_complex, steps)
        assert result.shape == csi_complex.shape

    def test_execute_with_custom_algorithm(self, csi_complex):
        """Execute pipeline with a custom algorithm."""
        def simple_denoise(csi, factor=1.0, **kwargs):
            return csi * factor

        register_algorithm('denoise', 'simple', simple_denoise)

        steps = {
            'denoise': {'method': 'simple', 'factor': 0.5},
        }
        result = execute_pipeline(csi_complex, steps)

        np.testing.assert_allclose(result, csi_complex * 0.5)

        unregister_algorithm('denoise', 'simple')


# ============================================================================
# Integration: Registry + Unified API Tests
# ============================================================================
class TestRegistryIntegration:
    def test_call_algorithm_filters_unsupported_kwargs(self, csi_complex):
        def scale_only(csi, scale=1.0):
            return csi * scale

        result = _call_algorithm(scale_only, csi_complex, scale=0.5, ignored=True)

        np.testing.assert_allclose(result, csi_complex * 0.5)

    def test_denose_via_registry(self, csi_complex):
        """denoise() should resolve through registry."""
        result = denoise(csi_complex, method='butterworth', order=4, cutoff=0.3)
        assert result.shape == csi_complex.shape

    def test_calibrate_via_registry(self, csi_complex):
        result = calibrate(csi_complex, method='polynomial', degree=2)
        assert result.shape == csi_complex.shape

    def test_normalize_via_registry(self, csi_complex):
        result = normalize(csi_complex, method='z-score')
        assert result.shape == csi_complex.shape

    def test_all_builtin_denoise_methods(self, csi_complex):
        """All built-in denoise methods should work via unified API."""
        for method in ['wavelet', 'butterworth', 'savgol']:
            result = denoise(csi_complex, method=method)
            assert result.shape == csi_complex.shape, f"Failed for method={method}"

    def test_all_builtin_calibrate_methods(self, csi_complex):
        """All built-in calibrate methods should work via unified API."""
        for method in ['linear', 'polynomial', 'stc', 'robust']:
            result = calibrate(csi_complex, method=method)
            assert result.shape == csi_complex.shape, f"Failed for method={method}"
