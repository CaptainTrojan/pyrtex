# tests/unit/test_config.py

import os
from unittest.mock import patch

import pytest

from pyrtex.config import GenerationConfig, InfrastructureConfig


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


class TestGenerationConfig:
    """Test GenerationConfig class."""

    def test_default_config(self):
        """Test default generation configuration."""
        config = GenerationConfig()

        assert config.temperature == 0.0
        assert config.max_output_tokens == 2048
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

        assert data == {"temperature": 0.7, "max_output_tokens": 1024, "top_p": 0.9}

    def test_model_dump_exclude_none(self):
        """Test serialization excludes None values."""
        config = GenerationConfig(
            temperature=0.5,
            max_output_tokens=512,
            # top_p and top_k are None by default
        )

        data = config.model_dump(exclude_none=True)

        assert data == {"temperature": 0.5, "max_output_tokens": 512}
        assert "top_p" not in data
        assert "top_k" not in data
