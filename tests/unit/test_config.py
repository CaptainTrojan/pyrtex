# tests/unit/test_config.py

import os
from unittest.mock import patch

import pytest

from pyrtex.config import GenerationConfig, InfrastructureConfig, ThinkingConfig


class TestInfrastructureConfig:
    """Test InfrastructureConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        # Clear any environment variables that might interfere
        with patch.dict(os.environ, {}, clear=True):
            config = InfrastructureConfig()

            assert config.project_id is None
            assert config.location == "us-central1"
            assert config.gcs_bucket_name is None
            assert config.bq_dataset_id is None

    def test_explicit_config(self):
        """Test explicit configuration values."""
        # Clear any environment variables that might interfere
        with patch.dict(os.environ, {}, clear=True):
            config = InfrastructureConfig(
                project_id="test-project",
                location="us-west1",
                gcs_bucket_name="test-bucket",
                bq_dataset_id="test_dataset",
            )

            assert config.project_id == "test-project"
            assert config.location == "us-west1"
            assert config.gcs_bucket_name == "test-bucket"
            assert config.bq_dataset_id == "test_dataset"

    def test_environment_variables(self):
        """Test configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "GOOGLE_PROJECT_ID": "env-project",
                "GOOGLE_LOCATION": "europe-west1",
                "PYRTEX_GCS_BUCKET_NAME": "env-bucket",
                "PYRTEX_BQ_DATASET_ID": "env_dataset",
            },
        ):
            config = InfrastructureConfig()

            assert config.project_id == "env-project"
            assert config.location == "europe-west1"
            assert config.gcs_bucket_name == "env-bucket"
            assert config.bq_dataset_id == "env_dataset"

    def test_env_var_precedence(self):
        """Test that explicit values override environment variables."""
        with patch.dict(
            os.environ,
            {"GOOGLE_PROJECT_ID": "env-project", "GOOGLE_LOCATION": "asia-east1"},
        ):
            config = InfrastructureConfig(
                project_id="explicit-project", location="us-west1"
            )

            # Explicit values should override env vars
            assert config.project_id == "explicit-project"
            assert config.location == "us-west1"

    def test_service_account_key_json_env_var(self):
        """Test service account JSON from environment variable."""
        json_key = '{"type": "service_account", "client_email": "test@test.com"}'

        with patch.dict(
            os.environ, {"PYRTEX_SERVICE_ACCOUNT_KEY_JSON": json_key}, clear=True
        ):
            config = InfrastructureConfig()
            assert config.service_account_key_json == json_key

    def test_service_account_key_path_env_var(self, tmp_path):
        """Test service account file path from environment variable."""
        # Create a valid service account file
        sa_file = tmp_path / "service_account.json"
        sa_file.write_text(
            '{"type": "service_account", "client_email": "test@test.com", '
            '"private_key": "key", "token_uri": "uri"}'
        )

        with patch.dict(
            os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": str(sa_file)}, clear=True
        ):
            config = InfrastructureConfig()
            assert config.service_account_key_path == str(sa_file)

    def test_service_account_key_path_env_var_not_service_account(self, tmp_path):
        """Test that user ADC file is not set as service account path."""
        # Create a user ADC file (different format)
        adc_file = tmp_path / "application_default_credentials.json"
        adc_file.write_text(
            '{"client_id": "123", "client_secret": "secret", "refresh_token": "token"}'
        )

        with patch.dict(
            os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": str(adc_file)}, clear=True
        ):
            config = InfrastructureConfig()
            assert config.service_account_key_path is None

    def test_explicit_auth_overrides_env(self, tmp_path):
        """Test that explicit auth config overrides environment variables."""
        sa_file = tmp_path / "service_account.json"
        sa_file.write_text(
            '{"type": "service_account", "client_email": "test@test.com", '
            '"private_key": "key", "token_uri": "uri"}'
        )

        json_key = '{"type": "service_account", "client_email": "explicit@test.com"}'

        with patch.dict(
            os.environ,
            {
                "PYRTEX_SERVICE_ACCOUNT_KEY_JSON": "env_json_key",
                "GOOGLE_APPLICATION_CREDENTIALS": str(sa_file),
            },
            clear=True,
        ):
            config = InfrastructureConfig(
                service_account_key_json=json_key,
                service_account_key_path="/explicit/path",
            )

            # Explicit values should override env vars
            assert config.service_account_key_json == json_key
            assert config.service_account_key_path == "/explicit/path"

    def test_is_service_account_file_valid(self, tmp_path):
        """Test _is_service_account_file with valid service account file."""
        config = InfrastructureConfig()

        # Create valid service account file
        sa_file = tmp_path / "service_account.json"
        sa_file.write_text(
            '{"type": "service_account", "client_email": "test@test.com", '
            '"private_key": "key", "token_uri": "uri"}'
        )

        assert config._is_service_account_file(str(sa_file)) is True

    def test_is_service_account_file_invalid(self, tmp_path):
        """Test _is_service_account_file with invalid file."""
        config = InfrastructureConfig()

        # Create user ADC file (different format)
        adc_file = tmp_path / "application_default_credentials.json"
        adc_file.write_text(
            '{"client_id": "123", "client_secret": "secret", "refresh_token": "token"}'
        )

        assert config._is_service_account_file(str(adc_file)) is False

    def test_is_service_account_file_nonexistent(self):
        """Test _is_service_account_file with non-existent file."""
        config = InfrastructureConfig()

        assert config._is_service_account_file("/nonexistent/file.json") is False

    def test_is_service_account_file_invalid_json(self, tmp_path):
        """Test _is_service_account_file with invalid JSON."""
        config = InfrastructureConfig()

        # Create file with invalid JSON
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json")

        assert config._is_service_account_file(str(bad_file)) is False


class TestGenerationConfig:
    """Test GenerationConfig class."""

    def test_default_config(self):
        """Test default generation configuration."""
        config = GenerationConfig()

        assert config.temperature == 0.0
        assert config.max_output_tokens == None
        assert config.top_p is None
        assert config.top_k is None

    def test_explicit_config(self):
        """Test explicit generation configuration."""
        config = GenerationConfig(
            temperature=0.7, max_output_tokens=1024, top_p=0.9, top_k=40
        )

        assert config.temperature == 0.7
        assert config.max_output_tokens == 1024
        assert config.top_p == 0.9
        assert config.top_k == 40

    def test_temperature_validation(self):
        """Test temperature validation."""
        # Valid temperatures
        GenerationConfig(temperature=0.0)
        GenerationConfig(temperature=1.0)
        GenerationConfig(temperature=2.0)

        # Invalid temperatures
        with pytest.raises(ValueError):
            GenerationConfig(temperature=-0.1)

        with pytest.raises(ValueError):
            GenerationConfig(temperature=2.1)

    def test_max_output_tokens_validation(self):
        """Test max_output_tokens validation."""
        # Valid values
        GenerationConfig(max_output_tokens=1)
        GenerationConfig(max_output_tokens=8192)

        # Invalid values
        with pytest.raises(ValueError):
            GenerationConfig(max_output_tokens=0)

        with pytest.raises(ValueError):
            GenerationConfig(max_output_tokens=-1)

    def test_top_p_validation(self):
        """Test top_p validation."""
        # Valid values
        GenerationConfig(top_p=0.0)
        GenerationConfig(top_p=0.5)
        GenerationConfig(top_p=1.0)
        GenerationConfig(top_p=None)

        # Invalid values
        with pytest.raises(ValueError):
            GenerationConfig(top_p=-0.1)

        with pytest.raises(ValueError):
            GenerationConfig(top_p=1.1)

    def test_top_k_validation(self):
        """Test top_k validation."""
        # Valid values
        GenerationConfig(top_k=1)
        GenerationConfig(top_k=100)
        GenerationConfig(top_k=None)

        # Invalid values
        with pytest.raises(ValueError):
            GenerationConfig(top_k=0)

        with pytest.raises(ValueError):
            GenerationConfig(top_k=-1)

    def test_model_dump(self):
        """Test serialization to dict."""
        config = GenerationConfig(temperature=0.7, max_output_tokens=1024, top_p=0.9)

        data = config.model_dump(exclude_none=True)

        assert data == {"temperature": 0.7, "max_output_tokens": 1024, "top_p": 0.9, "thinking_config": {"thinking_budget": -1}}
        
    def test_model_dump_with_thinking_config(self):
        """Test serialization includes thinking_config."""
        config = GenerationConfig(
            temperature=0.5,
            max_output_tokens=512,
            top_p=0.9,
            thinking_config=ThinkingConfig(thinking_budget=100)
        )

        data = config.model_dump(exclude_none=True)

        assert data == {"temperature": 0.5, "max_output_tokens": 512, "top_p": 0.9, "thinking_config": {"thinking_budget": 100}}

    def test_model_dump_exclude_none(self):
        """Test serialization excludes None values."""
        config = GenerationConfig(
            temperature=0.5,
            max_output_tokens=512,
            # top_p and top_k are None by default
        )

        data = config.model_dump(exclude_none=True)

        assert data == {"temperature": 0.5, "max_output_tokens": 512, "thinking_config": {"thinking_budget": -1}}
        assert "top_p" not in data
        assert "top_k" not in data
