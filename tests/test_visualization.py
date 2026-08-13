"""Tests for visualization module."""
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


class TestVisualizationImport:
    def test_import(self):
        from wsdp.algorithms.visualization import (
            plot_csi_heatmap,
            plot_denoising_comparison,
            plot_phase_calibration,
        )
        assert callable(plot_csi_heatmap)
        assert callable(plot_denoising_comparison)
        assert callable(plot_phase_calibration)


class TestPlotCSIHeatmap:
    def test_basic(self, tmp_path):
        from wsdp.algorithms.visualization import plot_csi_heatmap
        data = np.random.randn(100, 30, 2) + 1j * np.random.randn(100, 30, 2)
        fig = plot_csi_heatmap(data, antenna_idx=0)
        assert fig is not None
        fig.savefig(tmp_path / "test_plot.png")
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_2d_input(self):
        from wsdp.algorithms.visualization import plot_csi_heatmap
        data = np.random.randn(100, 30)
        fig = plot_csi_heatmap(data)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_invalid_dim(self):
        from wsdp.algorithms.visualization import plot_csi_heatmap
        with pytest.raises(ValueError):
            plot_csi_heatmap(np.random.randn(30))


class TestPlotDenoisingComparison:
    def test_basic(self):
        from wsdp.algorithms.visualization import plot_denoising_comparison
        orig = np.random.randn(100, 30, 2) + 1j * np.random.randn(100, 30, 2)
        denoised = orig * 0.9  # simulated denoised
        fig = plot_denoising_comparison(orig, denoised)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotPhaseCalibration:
    def test_basic(self):
        from wsdp.algorithms.visualization import plot_phase_calibration
        t = np.arange(30)
        orig = np.zeros((5, 30, 2), dtype=complex)
        cal = np.zeros((5, 30, 2), dtype=complex)
        for i in range(5):
            for a in range(2):
                phase_orig = 0.5 * t + 0.3 * a
                phase_cal = 0.1 * t + 0.05 * a  # flatter
                orig[i, :, a] = np.exp(1j * phase_orig)
                cal[i, :, a] = np.exp(1j * phase_cal)
        fig = plot_phase_calibration(orig, cal)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
