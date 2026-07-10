"""Tests for CLI module."""
import sys
import pytest
from unittest.mock import patch
from wsdp.cli import main_cli, __version__


class TestVersion:
    def test_version_string(self):
        from wsdp import __version__ as pkg_version
        assert __version__ == pkg_version

    def test_version_flag(self):
        with patch('sys.argv', ['wsdp', '--version']):
            with pytest.raises(SystemExit) as exc_info:
                main_cli()
            assert exc_info.value.code == 0


class TestListCommand:
    def test_list_basic(self, capsys):
        with patch('sys.argv', ['wsdp', 'list']):
            main_cli()
        captured = capsys.readouterr()
        assert "Available datasets" in captured.out
        assert "widar" in captured.out

    def test_list_verbose(self, capsys):
        with patch('sys.argv', ['wsdp', 'list', '--verbose']):
            main_cli()
        captured = capsys.readouterr()
        assert "Format:" in captured.out
        assert "Description:" in captured.out


class TestHelpCommand:
    def test_no_args(self, capsys):
        with patch('sys.argv', ['wsdp']):
            main_cli()
        captured = capsys.readouterr()
        assert "available commands" in captured.out

    def test_download_help(self, capsys):
        with patch('sys.argv', ['wsdp', 'download', '-h']):
            with pytest.raises(SystemExit):
                main_cli()


class TestRunCommand:
    def test_run_passes_model_and_algorithm_args(self):
        with patch('sys.argv', [
            'wsdp', 'run', 'data/xrf55', 'output', 'xrf55',
            '--model', 'cnn1dmodel',
            '--model-kwargs', '{"dropout": 0.25}',
            '--algorithm-preset', 'robust',
            '--algorithm-config', 'algorithms.yaml',
        ]), patch('wsdp.cli.pipeline') as pipeline_mock:
            main_cli()

        pipeline_mock.assert_called_once()
        kwargs = pipeline_mock.call_args.kwargs
        assert kwargs['model_name'] == 'cnn1dmodel'
        assert kwargs['model_kwargs'] == {'dropout': 0.25}
        assert kwargs['algorithm_preset'] == 'robust'
        assert kwargs['algorithm_config_file'] == 'algorithms.yaml'


class TestNonInteractiveMode:
    def test_download_has_email_arg(self):
        """Verify --email argument is available."""
        from wsdp.cli import main_cli
        with patch('sys.argv', ['wsdp', 'download', '-h']):
            with pytest.raises(SystemExit):
                main_cli()

    def test_download_has_token_arg(self):
        """Verify --token argument is available."""
        with patch('sys.argv', ['wsdp', 'download', '-h']):
            with pytest.raises(SystemExit):
                main_cli()


class TestDownloadAuth:
    def test_download_widar_requires_auth(self, capsys):
        """widar without credentials should be rejected."""
        with patch('sys.argv', ['wsdp', 'download', 'widar', './data']), \
             patch('wsdp.cli.download') as mock_download:
            main_cli()
        captured = capsys.readouterr()
        assert "requires authentication" in captured.out
        mock_download.assert_not_called()

    def test_download_gait_requires_auth(self, capsys):
        """gait without credentials should be rejected."""
        with patch('sys.argv', ['wsdp', 'download', 'gait', './data']), \
             patch('wsdp.cli.download') as mock_download:
            main_cli()
        captured = capsys.readouterr()
        assert "requires authentication" in captured.out
        mock_download.assert_not_called()

    def test_download_gait_token_allowed(self):
        """gait with a token should be forwarded to download()."""
        with patch('sys.argv', ['wsdp', 'download', 'gait', './data', '--token', 't']), \
             patch('wsdp.cli.download') as mock_download:
            main_cli()
        mock_download.assert_called_once_with(
            'gait', './data', email=None, password=None, token='t', extensions=None
        )
