"""Tests for download module."""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from wsdp.download import download

# wsdp/__init__.py exports a `download` function that shadows the wsdp.download
# module object. In Python < 3.13 patch('wsdp.download.xxx') resolves to the
# function, causing AttributeError. We use the real module from sys.modules.
_download_module = sys.modules['wsdp.download']


class TestDownloadAuth:
    def test_download_with_token(self):
        """download() should use token when provided."""
        with patch.object(_download_module, 'load_mapping', return_value="test.zip"), \
             patch.object(_download_module, '_download_without_aws', side_effect=Exception("force SDP")), \
             patch.object(_download_module, 'load_api', return_value="https://api.test/auth"), \
             patch.object(_download_module, 'requests') as mock_req:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = "https://download.url/file.zip"
            mock_req.post.return_value = mock_resp

            with patch.object(_download_module, '_download_file_from_url') as mock_dl:
                download("widar", "/tmp", token="my-jwt-token")
                # Verify token was used
                call_args = mock_req.post.call_args
                headers = call_args[1]['headers']
                assert 'Authorization' in headers
                assert headers['Authorization'] == 'Bearer my-jwt-token'

    def test_download_with_email_password(self):
        """download() should use email/password when no token."""
        with patch.object(_download_module, 'load_mapping', return_value="test.zip"), \
             patch.object(_download_module, '_download_without_aws', side_effect=Exception("force SDP")), \
             patch.object(_download_module, 'load_api', return_value="https://api.test/auth"), \
             patch.object(_download_module, 'requests') as mock_req:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = "https://download.url/file.zip"
            mock_req.post.return_value = mock_resp

            with patch.object(_download_module, '_download_file_from_url'):
                download("widar", "/tmp", email="user@test.com", password="secret")
                call_args = mock_req.post.call_args
                payload = call_args[1]['json']
                assert payload['email'] == "user@test.com"
                assert payload['password'] == "secret"

    def test_widar_uses_auth_api_not_ftp(self):
        """widar should skip the FTP path and use the auth API directly."""
        with patch.object(_download_module, 'load_mapping', return_value="Widar3.0-part.zip"), \
             patch.object(_download_module, '_download_without_aws') as mock_no_aws, \
             patch.object(_download_module, 'load_api', return_value="https://api.test/auth"), \
             patch.object(_download_module, 'requests') as mock_req:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = "https://bucket.url/Widar3.0-part.zip"
            mock_req.post.return_value = mock_resp

            with patch.object(_download_module, '_download_file_from_url') as mock_dl:
                download("widar", "/tmp", token="my-jwt-token")
                mock_no_aws.assert_not_called()
                mock_dl.assert_called_once_with("https://bucket.url/Widar3.0-part.zip", "/tmp", "Widar3.0-part.zip")

    def test_gait_uses_auth_api_not_ftp(self):
        """gait should skip the FTP path and use the auth API directly."""
        with patch.object(_download_module, 'load_mapping', return_value="GaitID.zip"), \
             patch.object(_download_module, '_download_without_aws') as mock_no_aws, \
             patch.object(_download_module, 'load_api', return_value="https://api.test/auth"), \
             patch.object(_download_module, 'requests') as mock_req:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = "https://bucket.url/GaitID.zip"
            mock_req.post.return_value = mock_resp

            with patch.object(_download_module, '_download_file_from_url') as mock_dl:
                download("gait", "/tmp", token="my-jwt-token")
                mock_no_aws.assert_not_called()
                mock_dl.assert_called_once_with("https://bucket.url/GaitID.zip", "/tmp", "GaitID.zip")
