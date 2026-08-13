"""Tests for pipeline configuration parsing."""

from wsdp.algorithms import AlgorithmStep, build_steps_from_config
from wsdp.algorithms.registry import CATEGORY_ORDER


class TestBuildStepsFromConfig:
    def test_default_order(self):
        config = {
            "normalize": {"method": "z-score"},
            "denoise": {"method": "wavelet"},
            "calibrate": {"method": "linear"},
        }
        steps = build_steps_from_config(config)
        assert [s.category for s in steps] == ["denoise", "calibrate", "normalize"]

    def test_omitted_steps_are_skipped(self):
        """A2C1 combination: denoise + normalize, no calibrate."""
        config = {
            "denoise": {"method": "wavelet"},
            "normalize": {"method": "z-score"},
        }
        steps = build_steps_from_config(config)
        assert len(steps) == 2
        assert steps[0].category == "denoise"
        assert steps[0].method == "wavelet"
        assert steps[1].category == "normalize"
        assert steps[1].method == "z-score"

    def test_params_are_preserved(self):
        config = {
            "denoise": {"method": "butterworth", "order": 5, "cutoff": 0.3},
        }
        steps = build_steps_from_config(config)
        assert steps[0].params == {"order": 5, "cutoff": 0.3}

    def test_user_defined_category_is_appended(self):
        config = {
            "denoise": {"method": "wavelet"},
            "my_custom": {"method": "special"},
        }
        steps = build_steps_from_config(config)
        assert steps[0].category == "denoise"
        assert steps[1].category == "my_custom"

    def test_returns_algorithm_steps(self):
        config = {"denoise": {"method": "wavelet"}}
        steps = build_steps_from_config(config)
        assert isinstance(steps[0], AlgorithmStep)

    def test_category_order_constant(self):
        assert "denoise" in CATEGORY_ORDER
        assert "calibrate" in CATEGORY_ORDER
        assert "normalize" in CATEGORY_ORDER
