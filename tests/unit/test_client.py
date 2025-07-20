# tests/unit/test_client.py

import json
import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from google.api_core.exceptions import NotFound

from pyrtex.client import Job
from pyrtex.config import GenerationConfig, InfrastructureConfig
from pyrtex.exceptions import ConfigurationError, JobFailedError
from tests.conftest import FileInput, SimpleInput, SimpleOutput


class TestJobInitialization:
    """Test Job class initialization and configuration."""

    def test_job_initialization_with_defaults(self, mock_gcp_clients):
        """Test that Job initializes correctly with default config."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        assert job.model == "gemini-2.0-flash-lite-001"
        assert job.output_schema == SimpleOutput
        assert job.prompt_template == "Test: {{ word }}"
        assert job.simulation_mode is False
        assert isinstance(job.generation_config, GenerationConfig)
        assert isinstance(job.config, InfrastructureConfig)

    def test_job_initialization_with_simulation_mode(self, mock_gcp_clients):
        """Test that Job initializes correctly with simulation mode."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True,
        )

        assert job.simulation_mode is True

    def test_job_initialization_configuration_error(self, mocker):
        """Test that Job raises ConfigurationError when GCP clients fail."""
        mocker.patch(
            "google.cloud.storage.Client", side_effect=Exception("Auth failed")
        )

        with pytest.raises(ConfigurationError) as exc_info:
            Job(
                model="gemini-2.0-flash-lite-001",
                output_schema=SimpleOutput,
                prompt_template="Test: {{ word }}",
            )

        assert "Failed to initialize GCP clients" in str(exc_info.value)
        assert "gcloud auth application-default login" in str(exc_info.value)

    def test_project_id_discovery_failure(self, mocker):
        """Test ConfigurationError when project ID cannot be discovered."""
        # Mock successful authentication but with no project ID discovery
        mock_credentials = Mock()
        mocker.patch("google.auth.default", return_value=(mock_credentials, None))
        
        # Mock storage client with no project ID
        mock_storage_client = Mock()
        mock_storage_client.project = None
        mocker.patch("google.cloud.storage.Client", return_value=mock_storage_client)
        mocker.patch("google.cloud.bigquery.Client")
        mocker.patch("google.cloud.aiplatform.init")

        # Clear the environment variable to simulate no project ID available
        mocker.patch.dict("os.environ", {}, clear=True)

        with pytest.raises(ConfigurationError) as exc_info:
            Job(
                model="gemini-2.0-flash-lite-001",
                output_schema=SimpleOutput,
                prompt_template="Test: {{ word }}",
            )

        assert "Could not automatically discover GCP Project ID" in str(exc_info.value)


class TestAuthentication:
    """Test authentication methods and credential handling."""
    
    def test_service_account_json_string(self, mock_gcp_clients_no_auth, mocker):
        """Test authentication with service account JSON string."""
        mock_credentials = Mock()
        mock_credentials.project_id = "test-project"
        
        mock_sa = mocker.patch("google.oauth2.service_account.Credentials.from_service_account_info")
        mock_sa.return_value = mock_credentials
        
        json_key = '{"type": "service_account", "client_email": "test@test.com", "private_key": "key", "token_uri": "uri"}'
        
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            config=InfrastructureConfig(
                service_account_key_json=json_key
            ),
        )
        
        assert job.config.project_id == "test-project"
        mock_sa.assert_called_once()
    
    def test_service_account_json_string_invalid_json(self, mock_gcp_clients_no_auth):
        """Test error handling for invalid JSON string."""
        with pytest.raises(ConfigurationError) as exc_info:
            Job(
                model="gemini-2.0-flash-lite-001", 
                output_schema=SimpleOutput,
                prompt_template="Test: {{ word }}",
                config=InfrastructureConfig(
                    service_account_key_json="invalid json"
                ),
            )
        
        assert "Invalid JSON in service account key" in str(exc_info.value)
    
    def test_service_account_json_string_auth_error(self, mock_gcp_clients_no_auth, mocker):
        """Test error handling for service account auth failure."""
        mocker.patch("google.oauth2.service_account.Credentials.from_service_account_info", 
                    side_effect=Exception("Auth failed"))
        
        json_key = '{"type": "service_account", "client_email": "test@test.com"}'
        
        with pytest.raises(ConfigurationError) as exc_info:
            Job(
                model="gemini-2.0-flash-lite-001",
                output_schema=SimpleOutput,
                prompt_template="Test: {{ word }}",
                config=InfrastructureConfig(
                    service_account_key_json=json_key
                ),
            )
        
        assert "Failed to load service account from JSON string" in str(exc_info.value)
    
    def test_service_account_file_path(self, mock_gcp_clients_no_auth, mocker, tmp_path):
        """Test authentication with service account file."""
        mock_credentials = Mock()
        mock_credentials.project_id = "file-project"
        
        mock_sa = mocker.patch("google.oauth2.service_account.Credentials.from_service_account_file")
        mock_sa.return_value = mock_credentials
        
        # Create a temporary service account file
        sa_file = tmp_path / "service_account.json"
        sa_file.write_text('{"type": "service_account", "client_email": "test@test.com", "private_key": "key", "token_uri": "uri"}')
        
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            config=InfrastructureConfig(
                service_account_key_path=str(sa_file)
            ),
        )
        
        assert job.config.project_id == "file-project"
        mock_sa.assert_called_once_with(str(sa_file), scopes=['https://www.googleapis.com/auth/cloud-platform'])
    
    def test_service_account_file_path_error(self, mock_gcp_clients_no_auth, mocker, tmp_path):
        """Test error handling for service account file failure."""
        mocker.patch("google.oauth2.service_account.Credentials.from_service_account_file",
                    side_effect=Exception("File error"))
        
        sa_file = tmp_path / "service_account.json"
        sa_file.write_text('{"type": "service_account", "client_email": "test@test.com", "private_key": "key", "token_uri": "uri"}')
        
        with pytest.raises(ConfigurationError) as exc_info:
            Job(
                model="gemini-2.0-flash-lite-001",
                output_schema=SimpleOutput,
                prompt_template="Test: {{ word }}",
                config=InfrastructureConfig(
                    service_account_key_path=str(sa_file)
                ),
            )
        
        assert f"Failed to load service account from file '{sa_file}'" in str(exc_info.value)
    
    def test_is_service_account_file_valid(self, tmp_path):
        """Test _is_service_account_file with valid service account file."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True
        )
        
        # Create valid service account file
        sa_file = tmp_path / "service_account.json"
        sa_file.write_text('{"type": "service_account", "client_email": "test@test.com", "private_key": "key", "token_uri": "uri"}')
        
        assert job._is_service_account_file(str(sa_file)) is True
    
    def test_is_service_account_file_invalid(self, tmp_path):
        """Test _is_service_account_file with invalid file."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True
        )
        
        # Create user ADC file (different format)
        adc_file = tmp_path / "application_default_credentials.json"
        adc_file.write_text('{"client_id": "123", "client_secret": "secret", "refresh_token": "token"}')
        
        assert job._is_service_account_file(str(adc_file)) is False
    
    def test_is_service_account_file_nonexistent(self):
        """Test _is_service_account_file with non-existent file."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True
        )
        
        assert job._is_service_account_file("/nonexistent/file.json") is False
    
    def test_is_service_account_file_invalid_json(self, tmp_path):
        """Test _is_service_account_file with invalid JSON."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True
        )
        
        # Create file with invalid JSON
        bad_file = tmp_path / "bad.json"
        bad_file.write_text('not valid json')
        
        assert job._is_service_account_file(str(bad_file)) is False

    def test_credentials_from_adc_with_project_discovery(self, mocker):
        """Test ADC with automatic project discovery."""
        mock_credentials = Mock()
        
        mock_adc = mocker.patch("google.auth.default")
        mock_adc.return_value = (mock_credentials, "discovered-project")
        
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True
        )
        
        # Test the method directly
        credentials = job._credentials_from_adc()
        
        assert credentials == mock_credentials
        # The project ID should be auto-discovered and set
        assert job.config.project_id == "discovered-project"
        mock_adc.assert_called_once_with(scopes=['https://www.googleapis.com/auth/cloud-platform'])

    def test_handle_authentication_error_adc_specific(self, mocker):
        """Test authentication error handling with ADC-specific help."""
        mocker.patch("google.auth.default", side_effect=Exception("Application Default Credentials not found"))
        
        with pytest.raises(ConfigurationError) as exc_info:
            Job(
                model="gemini-2.0-flash-lite-001",
                output_schema=SimpleOutput,
                prompt_template="Test: {{ word }}",
            )
        
        error_msg = str(exc_info.value)
        assert "Failed to initialize GCP clients" in error_msg
        assert "💡 ADC Troubleshooting:" in error_msg
        assert "gcloud auth application-default login" in error_msg
        assert "PYRTEX_PROJECT_ID or GOOGLE_PROJECT_ID" in error_msg
        assert "required permissions in your GCP project" in error_msg


class TestEnumValidation:
    """Test enum value validation to prevent boolean interpretation conflicts."""

    def test_problematic_enum_values_rejected(self, mock_gcp_clients):
        """Test that problematic enum values are rejected during job initialization."""
        from enum import Enum

        from pydantic import BaseModel, Field

        class ProblematicEnum(str, Enum):
            YES = "yes"
            NO = "no"

        class ProblematicOutput(BaseModel):
            choice: ProblematicEnum = Field(description="Test enum")

        with pytest.raises(ValueError) as exc_info:
            Job(
                model="gemini-2.0-flash-lite-001",
                output_schema=ProblematicOutput,
                prompt_template="Test",
            )

        error_msg = str(exc_info.value)
        assert "Enum value 'yes'" in error_msg
        assert "conflicts with JSON boolean interpretation" in error_msg
        assert "'recommend'/'not_recommend'" in error_msg

    def test_various_problematic_enum_values(self, mock_gcp_clients):
        """Test that various problematic enum values are all caught."""
        from enum import Enum

        from pydantic import BaseModel, Field

        problematic_values = [
            ("yes", "no"),
            ("true", "false"),
            ("YES", "NO"),
            ("True", "False"),
            ("y", "n"),
            ("Y", "N"),
            ("1", "0"),
        ]

        for val1, val2 in problematic_values:

            class TestEnum(str, Enum):
                OPTION1 = val1
                OPTION2 = val2

            class TestOutput(BaseModel):
                choice: TestEnum = Field(description="Test enum")

            with pytest.raises(ValueError) as exc_info:
                Job(
                    model="gemini-2.0-flash-lite-001",
                    output_schema=TestOutput,
                    prompt_template="Test",
                )

            error_msg = str(exc_info.value)
            assert (
                f"Enum value '{val1}'" in error_msg
                or f"Enum value '{val2}'" in error_msg
            )

    def test_safe_enum_values_accepted(self, mock_gcp_clients):
        """Test that safe enum values are accepted."""
        from enum import Enum

        from pydantic import BaseModel, Field

        class SafeEnum(str, Enum):
            RECOMMEND = "recommend"
            NOT_RECOMMEND = "not_recommend"
            EXCELLENT = "excellent"
            GOOD = "good"
            APPROVED = "approved"
            DENIED = "denied"

        class SafeOutput(BaseModel):
            choice: SafeEnum = Field(description="Safe enum")

        # This should not raise an exception
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SafeOutput,
            prompt_template="Test",
        )
        assert job is not None

    def test_optional_enum_validation(self, mock_gcp_clients):
        """Test that Optional[Enum] types are also validated."""
        from enum import Enum
        from typing import Optional

        from pydantic import BaseModel, Field

        class ProblematicEnum(str, Enum):
            TRUE = "true"
            FALSE = "false"

        class OptionalEnumOutput(BaseModel):
            choice: Optional[ProblematicEnum] = Field(description="Optional enum")

        with pytest.raises(ValueError) as exc_info:
            Job(
                model="gemini-2.0-flash-lite-001",
                output_schema=OptionalEnumOutput,
                prompt_template="Test",
            )

        error_msg = str(exc_info.value)
        assert "Enum value" in error_msg
        assert "conflicts with JSON boolean interpretation" in error_msg

    def test_non_enum_fields_ignored(self, mock_gcp_clients):
        """Test that non-enum fields are not affected by validation."""
        from pydantic import BaseModel, Field

        class MixedOutput(BaseModel):
            text_field: str = Field(description="Regular string")
            bool_field: bool = Field(description="Regular boolean")
            int_field: int = Field(description="Regular integer")

        # This should not raise an exception
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=MixedOutput,
            prompt_template="Test",
        )
        assert job is not None

    def test_nested_enum_validation(self, mock_gcp_clients):
        """Test that the validation works for direct enum fields."""
        from enum import Enum
        from typing import List

        from pydantic import BaseModel, Field

        class ProblematicEnum(str, Enum):
            YES = "yes"
            NO = "no"

        class OutputWithDirectEnum(BaseModel):
            choice: ProblematicEnum = Field(description="Direct enum field")
            text: str = Field(description="Regular text field")

        with pytest.raises(ValueError) as exc_info:
            Job(
                model="gemini-2.0-flash-lite-001",
                output_schema=OutputWithDirectEnum,
                prompt_template="Test",
            )

        error_msg = str(exc_info.value)
        assert "Enum value 'yes'" in error_msg


class TestJobRequestManagement:
    """Test adding requests to jobs."""

    def test_add_request_success(self, mock_gcp_clients):
        """Test successfully adding a request to a job."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        input_data = SimpleInput(word="hello")
        result = job.add_request("key1", input_data)

        assert result is job  # Should return self for chaining
        assert len(job._requests) == 1
        assert job._requests[0][0] == "key1"
        assert job._requests[0][1] == input_data

    def test_add_request_after_submission_fails(self, mock_gcp_clients):
        """Test that adding requests after submission raises an error."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        # Simulate job submission
        job._batch_job = Mock()

        with pytest.raises(RuntimeError) as exc_info:
            job.add_request("key1", SimpleInput(word="hello"))

        assert "Cannot add requests after job has been submitted" in str(exc_info.value)


class TestJobSubmission:
    """Test job submission logic."""

    def test_submit_without_requests_fails(self, mock_gcp_clients):
        """Test that submitting without requests raises an error."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        with pytest.raises(RuntimeError) as exc_info:
            job.submit()

        assert "Cannot submit a job with no requests" in str(exc_info.value)

    def test_submit_simulation_mode(self, mock_gcp_clients):
        """Test that simulation mode skips actual submission."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True,
        )

        job.add_request("key1", SimpleInput(word="hello"))
        result = job.submit()

        assert result is job
        assert job._batch_job is not None
        assert hasattr(job._batch_job, "state")  # Should be a mock object now

    def test_add_request_duplicate_key_validation(self, mock_gcp_clients):
        """Test that adding duplicate request keys raises an error."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        # Add first request
        job.add_request("duplicate_key", SimpleInput(word="hello"))

        # Adding same key should raise ValueError
        with pytest.raises(
            ValueError, match="Request key 'duplicate_key' already exists"
        ):
            job.add_request("duplicate_key", SimpleInput(word="world"))

    def test_submit_dry_run(self, mock_gcp_clients, capsys):
        """Test that dry run shows payload without submitting."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        job.add_request("key1", SimpleInput(word="hello"))
        result = job.submit(dry_run=True)

        captured = capsys.readouterr()
        assert "--- DRY RUN OUTPUT ---" in captured.out
        assert "Generated JSONL Payload" in captured.out
        assert result is job
        assert job._batch_job is None  # Should not be set in dry run

    def test_submit_success(self, mock_gcp_clients):
        """Test successful job submission."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        job.add_request("key1", SimpleInput(word="hello"))
        result = job.submit()

        assert result is job
        assert job._batch_job is not None
        mock_gcp_clients["storage"].bucket.assert_called()


class TestJobWaiting:
    """Test job waiting logic."""

    def test_wait_simulation_mode(self, mock_gcp_clients):
        """Test that wait() skips in simulation mode."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True,
        )

        result = job.wait()
        assert result is job

    def test_wait_without_submission(self, mock_gcp_clients, caplog):
        """Test wait() behavior when no job was submitted."""
        import logging

        caplog.set_level(logging.WARNING)

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        result = job.wait()

        assert result is job
        assert "No job submitted, nothing to wait for" in caplog.text
        assert result is job

    def test_wait_method_with_job(self, mock_gcp_clients, caplog):
        """Test wait method when job has been submitted."""
        import logging

        caplog.set_level(logging.INFO)

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        job.add_request("test1", SimpleInput(word="hello"))
        job.submit()  # This creates a mock batch job

        # Test that wait returns self and logs messages
        result = job.wait()
        assert result is job

        # Check that logging messages were printed
        assert "Waiting for job to complete" in caplog.text
        assert "Job completed!" in caplog.text


class TestJobResults:
    """Test job results retrieval."""

    def test_results_simulation_mode(self, mock_gcp_clients):
        """Test that results() returns dummy data in simulation mode."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True,
        )

        job.add_request("key1", SimpleInput(word="hello"))
        job.add_request("key2", SimpleInput(word="world"))

        results = list(job.results())

        assert len(results) == 2
        assert results[0].request_key == "key1"
        assert results[1].request_key == "key2"
        assert all(r.was_successful for r in results)
        assert all(isinstance(r.output, SimpleOutput) for r in results)
        assert all("dummy response" in r.raw_response["note"] for r in results)

    def test_results_without_submission_fails(self, mock_gcp_clients):
        """Test that results() fails when no job was submitted."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        with pytest.raises(RuntimeError) as exc_info:
            list(job.results())

        assert "Cannot get results for a job that has not been submitted" in str(
            exc_info.value
        )


class TestDummyResultsGeneration:
    """Test the _generate_dummy_results method."""

    def test_generate_dummy_results_simple_schema(self, mock_gcp_clients):
        """Test dummy results generation for simple schema."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True,
        )

        job.add_request("key1", SimpleInput(word="hello"))
        job.add_request("key2", SimpleInput(word="world"))

        results = list(job._generate_dummy_results())

        assert len(results) == 2
        assert results[0].request_key == "key1"
        assert results[1].request_key == "key2"
        assert all(r.was_successful for r in results)
        assert all(isinstance(r.output, SimpleOutput) for r in results)
        assert all(r.usage_metadata["promptTokenCount"] == 0 for r in results)

    def test_generate_dummy_results_with_required_fields(self, mock_gcp_clients):
        """Test dummy results generation for schema with required fields."""
        from pydantic import BaseModel

        class RequiredFieldsOutput(BaseModel):
            required_field: str
            optional_field: str = "default"

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=RequiredFieldsOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True,
        )

        job.add_request("key1", SimpleInput(word="hello"))

        results = list(job._generate_dummy_results())

        assert len(results) == 1
        assert results[0].was_successful


class TestCloudResourceSetup:
    """Test cloud resource setup methods."""

    def test_setup_cloud_resources_bucket_exists(self, mock_gcp_clients):
        """Test _setup_cloud_resources when bucket already exists."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        # Mock bucket exists
        mock_bucket = Mock()
        mock_gcp_clients["storage"].get_bucket.return_value = mock_bucket

        # Mock dataset exists
        mock_dataset = Mock()
        mock_dataset.default_table_expiration_ms = None
        mock_gcp_clients["bigquery"].get_dataset.return_value = mock_dataset

        job._setup_cloud_resources()

        # Verify bucket lifecycle rules were set
        mock_bucket.clear_lifecyle_rules.assert_called_once()
        mock_bucket.add_lifecycle_delete_rule.assert_called_once_with(age=1)
        mock_bucket.patch.assert_called_once()

        # Verify dataset expiration was set
        mock_gcp_clients["bigquery"].update_dataset.assert_called_once()

    def test_setup_cloud_resources_bucket_not_found(self, mock_gcp_clients):
        """Test _setup_cloud_resources when bucket doesn't exist."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        # Mock bucket doesn't exist
        mock_gcp_clients["storage"].get_bucket.side_effect = NotFound(
            "Bucket not found"
        )
        mock_bucket = Mock()
        mock_gcp_clients["storage"].create_bucket.return_value = mock_bucket

        # Mock dataset exists
        mock_dataset = Mock()
        mock_dataset.default_table_expiration_ms = None
        mock_gcp_clients["bigquery"].get_dataset.return_value = mock_dataset

        job._setup_cloud_resources()

        # Verify bucket was created
        mock_gcp_clients["storage"].create_bucket.assert_called_once_with(
            job.config.gcs_bucket_name, location=job.config.location
        )

        # Verify lifecycle rules were set
        mock_bucket.clear_lifecyle_rules.assert_called_once()
        mock_bucket.add_lifecycle_delete_rule.assert_called_once_with(age=1)
        mock_bucket.patch.assert_called_once()

    def test_setup_cloud_resources_dataset_not_found(self, mock_gcp_clients):
        """Test _setup_cloud_resources when dataset doesn't exist."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        # Mock bucket exists
        mock_bucket = Mock()
        mock_gcp_clients["storage"].get_bucket.return_value = mock_bucket

        # Mock dataset doesn't exist
        mock_gcp_clients["bigquery"].get_dataset.side_effect = NotFound(
            "Dataset not found"
        )
        mock_dataset = Mock()
        mock_dataset.default_table_expiration_ms = None
        mock_gcp_clients["bigquery"].create_dataset.return_value = mock_dataset

        job._setup_cloud_resources()

        # Verify dataset was created
        mock_gcp_clients["bigquery"].create_dataset.assert_called_once()

        # Verify dataset expiration was set
        mock_gcp_clients["bigquery"].update_dataset.assert_called_once()


class TestFileUpload:
    """Test file upload functionality."""

    def test_upload_file_to_gcs_bytes(self, mock_gcp_clients):
        """Test uploading bytes to GCS."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        # Mock bucket and blob
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_gcp_clients["storage"].bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        test_data = b"test content"
        gcs_uri, mime_type = job._upload_file_to_gcs(test_data, "test/path.txt")

        # Verify blob upload
        mock_blob.upload_from_string.assert_called_once_with(
            test_data, content_type="text/plain"
        )

        expected_uri = f"gs://{job.config.gcs_bucket_name}/test/path.txt"
        assert gcs_uri == expected_uri
        assert mime_type == "text/plain"

    def test_upload_file_to_gcs_path(self, mock_gcp_clients, tmp_path):
        """Test uploading file path to GCS."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        # Mock bucket and blob
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_gcp_clients["storage"].bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        # Create a temporary file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        gcs_uri, mime_type = job._upload_file_to_gcs(test_file, "test/path.txt")

        # Verify blob upload
        mock_blob.upload_from_filename.assert_called_once_with(
            str(test_file), content_type="text/plain"
        )

        expected_uri = f"gs://{job.config.gcs_bucket_name}/test/path.txt"
        assert gcs_uri == expected_uri
        assert mime_type == "text/plain"

    def test_string_file_path_upload(self, mock_gcp_clients, tmp_path):
        """Test file upload with string file path."""
        # Create a temporary file
        temp_file = tmp_path / "test.txt"
        temp_file.write_text("test content")

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process {{ image }}",
        )

        # Use string path instead of Path object
        file_input = FileInput(image=str(temp_file))
        job.add_request("file_key", file_input)

        payload = job._create_jsonl_payload()

        lines = payload.split("\n")
        assert len(lines) == 1

        data = json.loads(lines[0])
        parts = data["request"]["contents"][0]["parts"]

        # Should have file_data part and text part
        assert len(parts) == 2
        assert "file_data" in parts[0]
        assert "file_uri" in parts[0]["file_data"]
        assert parts[0]["file_data"]["file_uri"].startswith("gs://")
        assert "text" in parts[1]


class TestJsonlPayload:
    """Test JSONL payload creation."""

    def test_create_jsonl_payload_simple(self, mock_gcp_clients):
        """Test creating JSONL payload with simple text data."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        job.add_request("key1", SimpleInput(word="hello"))
        job.add_request("key2", SimpleInput(word="world"))

        jsonl_payload = job._create_jsonl_payload()

        # Parse the JSONL
        lines = jsonl_payload.strip().split("\n")
        assert len(lines) == 2

        # Check first line
        data1 = json.loads(lines[0])
        assert "id" in data1
        assert data1["request"]["contents"][0]["role"] == "user"
        assert data1["request"]["contents"][0]["parts"][0]["text"] == "Test: hello"

        # Check second line
        data2 = json.loads(lines[1])
        assert "id" in data2
        assert data2["request"]["contents"][0]["parts"][0]["text"] == "Test: world"

    def test_create_jsonl_payload_with_file(self, mock_gcp_clients, tmp_path):
        """Test creating JSONL payload with file data."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ image }}",
        )

        # Mock file upload
        mock_gcp_clients[
            "storage"
        ].bucket.return_value.blob.return_value.upload_from_filename = Mock()

        # Create test file
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake image data")

        job.add_request("key1", FileInput(image=str(test_file)))

        jsonl_payload = job._create_jsonl_payload()

        # Should have uploaded the file
        mock_gcp_clients["storage"].bucket.assert_called()

        # Check JSONL structure
        lines = jsonl_payload.strip().split("\n")
        assert len(lines) == 1

        data = json.loads(lines[0])
        assert "id" in data
        assert data["request"]["contents"][0]["role"] == "user"


class TestPayloadGeneration:
    """Test comprehensive JSONL payload generation."""

    def test_create_jsonl_payload_text_only(self, mock_gcp_clients):
        """Test payload generation with text-only input."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process this word: {{ word }}",
        )

        job.add_request("key1", SimpleInput(word="hello"))
        job.add_request("key2", SimpleInput(word="world"))

        payload = job._create_jsonl_payload()

        lines = payload.split("\n")
        assert len(lines) == 2

        # Parse first line
        first_line = json.loads(lines[0])
        assert first_line["id"].startswith("req_00000_")
        assert first_line["request"]["contents"][0]["role"] == "user"

        # Check parts - should only have text part
        parts = first_line["request"]["contents"][0]["parts"]
        assert len(parts) == 1
        assert parts[0]["text"] == "Process this word: hello"

        # Check generation config
        gen_config = first_line["request"]["generation_config"]
        assert gen_config["temperature"] == 0.0
        assert gen_config["max_output_tokens"] == 2048

        # Check tools configuration
        tools = first_line["request"]["tools"]
        assert len(tools) == 1
        assert tools[0]["function_declarations"][0]["name"] == "extract_info"
        assert "parameters" in tools[0]["function_declarations"][0]

    def test_create_jsonl_payload_with_custom_generation_config(self, mock_gcp_clients):
        """Test payload generation with custom generation config."""
        custom_config = GenerationConfig(
            temperature=0.7, max_output_tokens=1024, top_p=0.9
        )

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Custom: {{ word }}",
            generation_config=custom_config,
        )

        job.add_request("test_key", SimpleInput(word="custom"))

        payload = job._create_jsonl_payload()

        lines = payload.split("\n")
        assert len(lines) == 1

        data = json.loads(lines[0])
        gen_config = data["request"]["generation_config"]
        assert gen_config["temperature"] == 0.7
        assert gen_config["max_output_tokens"] == 1024
        assert gen_config["top_p"] == 0.9

    def test_create_jsonl_payload_with_file_path(self, mock_gcp_clients, tmp_path):
        """Test payload generation with file path input."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            temp_file_path = f.name

        try:
            job = Job(
                model="gemini-2.0-flash-lite-001",
                output_schema=SimpleOutput,
                prompt_template="Process {{ text }} from file",
            )

            file_input = FileInput(image=temp_file_path)
            job.add_request("file_key", file_input)

            payload = job._create_jsonl_payload()

            lines = payload.split("\n")
            assert len(lines) == 1

            data = json.loads(lines[0])
            parts = data["request"]["contents"][0]["parts"]

            # Should have file_data part
            assert len(parts) == 2
            assert "file_data" in parts[0]
            assert "file_uri" in parts[0]["file_data"]
            assert parts[0]["file_data"]["file_uri"].startswith("gs://")

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def test_create_jsonl_payload_mixed_input(self, mock_gcp_clients):
        """Test payload generation with mixed text and file inputs."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process {{ word }} and {{ image }}",
        )

        # Add text-only request
        job.add_request("text_key", SimpleInput(word="hello"))

        # Add file request
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("file content")
            temp_file_path = f.name

        try:
            file_input = FileInput(image=temp_file_path)
            job.add_request("file_key", file_input)

            payload = job._create_jsonl_payload()

            lines = payload.split("\n")
            assert len(lines) == 2

            # Check text-only request
            text_data = json.loads(lines[0])
            text_parts = text_data["request"]["contents"][0]["parts"]
            assert len(text_parts) == 1
            assert "text" in text_parts[0]

            # Check file request
            file_data = json.loads(lines[1])
            file_parts = file_data["request"]["contents"][0]["parts"]
            assert len(file_parts) == 2
            assert "file_data" in file_parts[0]

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def test_instance_id_generation(self, mock_gcp_clients):
        """Test that instance IDs are generated correctly."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        job.add_request("key1", SimpleInput(word="test1"))
        job.add_request("key2", SimpleInput(word="test2"))

        payload = job._create_jsonl_payload()

        lines = payload.split("\n")
        assert len(lines) == 2

        # Parse both lines
        data1 = json.loads(lines[0])
        data2 = json.loads(lines[1])

        # Check instance IDs are different
        id1 = data1["id"]
        id2 = data2["id"]
        assert id1 != id2
        assert id1.startswith("req_00000_")
        assert id2.startswith("req_00001_")

        # Check that instance map is populated
        assert "key1" in job._instance_map.values()
        assert "key2" in job._instance_map.values()


class TestJobEdgeCases:
    """Test edge cases and error handling in Job class."""

    def test_wait_method_without_job(self, mock_gcp_clients):
        """Test wait method when no job has been submitted."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        # Test that wait returns self when no job
        result = job.wait()
        assert result is job

    def test_results_method_real_job(self, mock_gcp_clients):
        """Test results method with real job (not simulation mode)."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        job.add_request("test1", SimpleInput(word="hello"))
        job.submit()  # This creates a mock batch job

        # Test that results method works
        results = list(job.results())
        assert isinstance(results, list)
        # In mock mode, this should return empty list

    def test_dummy_output_with_default_factory(self, mock_gcp_clients):
        """Test dummy output creation with default factory."""
        from pydantic import BaseModel, Field

        class OutputWithFactory(BaseModel):
            data: list[str] = Field(default_factory=list)
            count: int = Field(default_factory=lambda: 5)

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=OutputWithFactory,
            prompt_template="Test",
            simulation_mode=True,
        )

        job.add_request("test", SimpleInput(word="hello"))
        results = list(job.submit().wait().results())

        assert len(results) == 1
        result = results[0]
        assert isinstance(result.output.data, list)
        assert result.output.count == 5

    def test_dummy_output_with_union_types(self, mock_gcp_clients):
        """Test dummy output creation with union types."""
        from typing import Optional, Union

        from pydantic import BaseModel

        class OutputWithUnion(BaseModel):
            value: Union[str, int]
            optional_value: Optional[str] = None

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=OutputWithUnion,
            prompt_template="Test",
            simulation_mode=True,
        )

        job.add_request("test", SimpleInput(word="hello"))
        results = list(job.submit().wait().results())

        assert len(results) == 1
        result = results[0]
        assert isinstance(result.output.value, str)
        assert result.output.value == "dummy_value"

    def test_dummy_output_with_different_types(self, mock_gcp_clients):
        """Test dummy output creation with different field types."""
        from pydantic import BaseModel

        class OutputWithManyTypes(BaseModel):
            string_field: str
            int_field: int
            float_field: float
            bool_field: bool
            list_field: list[str]
            dict_field: dict[str, str]

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=OutputWithManyTypes,
            prompt_template="Test",
            simulation_mode=True,
        )

        job.add_request("test", SimpleInput(word="hello"))
        results = list(job.submit().wait().results())

        assert len(results) == 1
        result = results[0]
        assert isinstance(result.output.string_field, str)
        assert result.output.string_field == "dummy_string_field"
        assert isinstance(result.output.int_field, int)
        assert result.output.int_field == 42
        assert isinstance(result.output.float_field, float)
        assert result.output.float_field == 3.14
        assert isinstance(result.output.bool_field, bool)
        assert result.output.bool_field is True
        assert isinstance(result.output.list_field, list)
        assert result.output.list_field == ["dummy_list_field_item"]
        assert isinstance(result.output.dict_field, dict)
        assert result.output.dict_field == {
            "dummy_dict_field_key": "dummy_dict_field_value"
        }

    def test_dummy_output_with_complex_type(self, mock_gcp_clients):
        """Test dummy output creation with complex/unknown types."""
        from datetime import datetime

        from pydantic import BaseModel

        class OutputWithComplexType(BaseModel):
            timestamp: datetime
            custom_field: str

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=OutputWithComplexType,
            prompt_template="Test",
            simulation_mode=True,
        )

        job.add_request("test", SimpleInput(word="hello"))
        results = list(job.submit().wait().results())

        assert len(results) == 1
        result = results[0]
        # Complex types should now generate appropriate types
        assert isinstance(result.output.timestamp, datetime)
        assert isinstance(result.output.custom_field, str)
        assert result.output.custom_field == "dummy_custom_field"

    def test_dummy_output_with_existing_default_values(self, mock_gcp_clients):
        """Test dummy output creation with existing default values."""
        from pydantic import BaseModel, Field

        class OutputWithDefaults(BaseModel):
            name: str = "default_name"
            score: float = 0.5
            enabled: bool = False

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=OutputWithDefaults,
            prompt_template="Test",
            simulation_mode=True,
        )

        job.add_request("test", SimpleInput(word="hello"))
        results = list(job.submit().wait().results())

        assert len(results) == 1
        result = results[0]
        # Should use the default values
        assert result.output.name == "default_name"
        assert result.output.score == 0.5
        assert result.output.enabled is False

    def test_bytes_file_upload(self, mock_gcp_clients):
        """Test file upload with bytes input."""
        from pydantic import BaseModel

        class BytesInput(BaseModel):
            data: bytes

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process {{ data }}",
        )

        test_bytes = b"test content"
        bytes_input = BytesInput(data=test_bytes)
        job.add_request("bytes_key", bytes_input)

        payload = job._create_jsonl_payload()

        lines = payload.split("\n")
        assert len(lines) == 1

        data = json.loads(lines[0])
        parts = data["request"]["contents"][0]["parts"]

        # Should have file_data part and text part
        assert len(parts) == 2
        assert "file_data" in parts[0]
        assert "file_uri" in parts[0]["file_data"]
        assert parts[0]["file_data"]["file_uri"].startswith("gs://")
        assert "text" in parts[1]

    def test_path_object_file_upload(self, mock_gcp_clients, tmp_path):
        """Test file upload with Path object input."""
        from pathlib import Path

        from pydantic import BaseModel

        class PathInput(BaseModel):
            file_path: Path

        # Create a temporary file
        temp_file = tmp_path / "test.txt"
        temp_file.write_text("test content")

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process {{ file_path }}",
        )

        path_input = PathInput(file_path=temp_file)
        job.add_request("path_key", path_input)

        payload = job._create_jsonl_payload()

        lines = payload.split("\n")
        assert len(lines) == 1

        data = json.loads(lines[0])
        parts = data["request"]["contents"][0]["parts"]

        # Should have file_data part and text part
        assert len(parts) == 2
        assert "file_data" in parts[0]
        assert "file_uri" in parts[0]["file_data"]
        assert parts[0]["file_data"]["file_uri"].startswith("gs://")
        assert "text" in parts[1]

    def test_dummy_output_with_unknown_complex_type(self, mock_gcp_clients):
        """Test dummy output creation with unknown complex types."""
        from pathlib import Path

        from pydantic import BaseModel

        class OutputWithUnknownType(BaseModel):
            path_field: Path
            custom_field: str

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=OutputWithUnknownType,
            prompt_template="Test",
            simulation_mode=True,
        )

        job.add_request("test", SimpleInput(word="hello"))
        results = list(job.submit().wait().results())

        assert len(results) == 1
        result = results[0]
        # Path types are handled by pydantic, so they get converted properly
        assert isinstance(result.output.path_field, Path)
        assert str(result.output.path_field) == "dummy_path_field"
        assert isinstance(result.output.custom_field, str)
        assert result.output.custom_field == "dummy_custom_field"

    def test_results_job_not_succeeded(self, mock_gcp_clients):
        """Test that results() fails when job is not in succeeded state."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        job.add_request("test1", SimpleInput(word="hello"))
        job.submit()

        # Mock job state as failed
        job._batch_job.state = "JOB_STATE_FAILED"

        with pytest.raises(
            RuntimeError,
            match="Cannot get results for a job that has not completed successfully",
        ):
            list(job.results())

    def test_results_bigquery_parsing_success(self, mock_gcp_clients):
        """Test successful BigQuery result parsing."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        job.add_request("test1", SimpleInput(word="hello"))
        job.submit()

        # Set up instance map
        job._instance_map = {"req_00001_12345678": "test1"}

        # Mock BigQuery results
        mock_row = Mock()
        mock_row.id = "req_00001_12345678"
        mock_row.status = None  # No error status
        mock_row.response = json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "extract_info",
                                        "args": {"result": "test_output"},
                                    }
                                }
                            ]
                        }
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                },
            }
        )

        mock_gcp_clients["bigquery"].query.return_value.result.return_value = [mock_row]

        results = list(job.results())

        assert len(results) == 1
        assert results[0].request_key == "test1"
        assert results[0].output.result == "test_output"
        assert results[0].usage_metadata["totalTokenCount"] == 15

    def test_results_bigquery_parsing_no_function_call(self, mock_gcp_clients):
        """Test BigQuery result parsing when model doesn't return function call."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        job.add_request("test1", SimpleInput(word="hello"))
        job.submit()

        # Set up instance map
        job._instance_map = {"req_00001_12345678": "test1"}

        # Mock BigQuery results without function call
        mock_row = Mock()
        mock_row.id = "req_00001_12345678"
        mock_row.status = None  # No error status
        mock_row.response = json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "I cannot follow the instructions"}]
                        }
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                },
            }
        )

        mock_gcp_clients["bigquery"].query.return_value.result.return_value = [mock_row]

        results = list(job.results())

        assert len(results) == 1
        assert results[0].request_key == "test1"
        assert results[0].output is None
        assert "Failed to parse model output" in results[0].error

    def test_results_bigquery_parsing_validation_error(self, mock_gcp_clients):
        """Test BigQuery result parsing when validation fails."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        job.add_request("test1", SimpleInput(word="hello"))
        job.submit()

        # Set up instance map
        job._instance_map = {"req_00001_12345678": "test1"}

        # Mock BigQuery results with invalid data for schema
        mock_row = Mock()
        mock_row.id = "req_00001_12345678"
        mock_row.status = None  # No error status
        mock_row.response = json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "extract_info",
                                        "args": {
                                            "invalid_field": "test_output"
                                        },  # wrong field name
                                    }
                                }
                            ]
                        }
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                },
            }
        )

        mock_gcp_clients["bigquery"].query.return_value.result.return_value = [mock_row]

        results = list(job.results())

        assert len(results) == 1
        assert results[0].request_key == "test1"
        assert results[0].output is None
        assert "Validation error" in results[0].error

    def test_results_bigquery_query_error(self, mock_gcp_clients):
        """Test BigQuery result parsing when query fails."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        job.add_request("test1", SimpleInput(word="hello"))
        job.submit()

        # Mock BigQuery query failure
        mock_gcp_clients["bigquery"].query.side_effect = Exception("BigQuery error")

        with pytest.raises(
            RuntimeError, match="Error querying or parsing BigQuery results"
        ):
            list(job.results())


class TestSchemaFlattening:
    """Test schema flattening functionality for BigQuery compatibility."""

    def test_get_flattened_schema_no_refs(self, mock_gcp_clients):
        """Test schema flattening when there are no $ref references."""
        from pydantic import BaseModel, Field

        class SimpleSchema(BaseModel):
            name: str = Field(description="A simple string field")
            count: int = Field(description="A simple integer field")

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleSchema,
            prompt_template="Test",
        )

        flattened = job._get_flattened_schema()

        # Should return the schema as-is since there are no $defs
        assert "$defs" not in flattened
        assert "properties" in flattened
        assert "name" in flattened["properties"]
        assert "count" in flattened["properties"]

    def test_get_flattened_schema_with_refs(self, mock_gcp_clients):
        """Test schema flattening when there are $ref references."""
        from enum import Enum

        from pydantic import BaseModel, Field

        class TestEnum(str, Enum):
            OPTION1 = "option1"
            OPTION2 = "option2"

        class SchemaWithRefs(BaseModel):
            choice: TestEnum = Field(description="An enum field")
            name: str = Field(description="A string field")

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SchemaWithRefs,
            prompt_template="Test",
        )

        flattened = job._get_flattened_schema()

        # Should have inlined the enum definition
        assert "$defs" not in flattened
        assert "properties" in flattened
        assert "choice" in flattened["properties"]

        # The choice field should have the enum values directly
        choice_field = flattened["properties"]["choice"]
        assert "enum" in choice_field
        assert choice_field["enum"] == ["option1", "option2"]
        assert choice_field["type"] == "string"

    def test_get_flattened_schema_preserves_descriptions(self, mock_gcp_clients):
        """Test that schema flattening preserves field descriptions."""
        from enum import Enum

        from pydantic import BaseModel, Field

        class TestEnum(str, Enum):
            VALUE1 = "value1"
            VALUE2 = "value2"

        class SchemaWithDescriptions(BaseModel):
            enum_field: TestEnum = Field(description="Custom enum description")

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SchemaWithDescriptions,
            prompt_template="Test",
        )

        flattened = job._get_flattened_schema()

        # Should preserve the custom description
        enum_field = flattened["properties"]["enum_field"]
        assert enum_field["description"] == "Custom enum description"
        assert "enum" in enum_field

    def test_get_flattened_schema_with_lists(self, mock_gcp_clients):
        """Test schema flattening with list types containing refs."""
        from enum import Enum
        from typing import List

        from pydantic import BaseModel, Field

        class TestEnum(str, Enum):
            ITEM1 = "item1"
            ITEM2 = "item2"

        class SchemaWithLists(BaseModel):
            enum_list: List[TestEnum] = Field(description="List of enums")

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SchemaWithLists,
            prompt_template="Test",
        )

        flattened = job._get_flattened_schema()

        # Should have flattened the list items
        assert "$defs" not in flattened
        enum_list_field = flattened["properties"]["enum_list"]
        assert enum_list_field["type"] == "array"
        assert "items" in enum_list_field

        # The items should have the enum values inlined
        items = enum_list_field["items"]
        assert "enum" in items
        assert items["enum"] == ["item1", "item2"]

    def test_get_flattened_schema_ref_not_found(self, mock_gcp_clients):
        """Test schema flattening handles missing ref definitions gracefully."""
        # This test simulates a malformed schema with a $ref that doesn't exist in $defs
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test",
        )

        # Manually create a schema with a broken $ref for testing
        broken_schema = {
            "properties": {"broken_field": {"$ref": "#/$defs/NonExistentType"}},
            "$defs": {},
        }

        # Monkey patch the schema generation to return our broken schema
        original_method = job.output_schema.model_json_schema
        job.output_schema.model_json_schema = lambda: broken_schema

        try:
            flattened = job._get_flattened_schema()

            # Should return the broken $ref as-is since it can't be resolved
            assert (
                flattened["properties"]["broken_field"]["$ref"]
                == "#/$defs/NonExistentType"
            )
        finally:
            # Restore the original method
            job.output_schema.model_json_schema = original_method

    def test_get_flattened_schema_external_ref(self, mock_gcp_clients):
        """Test schema flattening ignores external (non-$defs) refs."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test",
        )

        # Create a schema with an external $ref
        external_ref_schema = {
            "properties": {
                "external_field": {"$ref": "http://example.com/schema#/SomeType"}
            }
        }

        # Monkey patch the schema generation
        original_method = job.output_schema.model_json_schema
        job.output_schema.model_json_schema = lambda: external_ref_schema

        try:
            flattened = job._get_flattened_schema()

            # Should leave external refs unchanged
            assert (
                flattened["properties"]["external_field"]["$ref"]
                == "http://example.com/schema#/SomeType"
            )
        finally:
            # Restore the original method
            job.output_schema.model_json_schema = original_method

    def test_get_flattened_schema_non_defs_ref(self, mock_gcp_clients):
        """Test schema flattening handles refs that don't start with #/$defs/."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test",
        )

        # Create a schema with both $defs and a non-$defs $ref to hit line 277
        non_defs_ref_schema = {
            "properties": {
                "anchor_field": {"$ref": "#/definitions/SomeType"},
                "normal_field": {"type": "string"},
            },
            "$defs": {
                "SomeDefType": {
                    "type": "object",
                    "properties": {"test": {"type": "string"}},
                }
            },
        }

        # Monkey patch the schema generation
        original_method = job.output_schema.model_json_schema
        job.output_schema.model_json_schema = lambda: non_defs_ref_schema

        try:
            flattened = job._get_flattened_schema()

            # Should leave non-$defs refs unchanged (this exercises line 277:
            # return obj)
            assert (
                flattened["properties"]["anchor_field"]["$ref"]
                == "#/definitions/SomeType"
            )
            # Should not have $defs in the flattened schema
            assert "$defs" not in flattened
        finally:
            # Restore the original method
            job.output_schema.model_json_schema = original_method

    def test_get_flattened_schema_deeply_nested_refs(self, mock_gcp_clients):
        """
        Test schema flattening with deeply nested structures to ensure
        all branches are covered.
        """
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test",
        )

        # Create a complex nested schema with various ref patterns to hit all code paths
        complex_schema = {
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "array",
                            "items": {
                                # External ref - should hit line 277
                                "$ref": "http://external.com/schema#/Type"
                            },
                        }
                    },
                }
            }
        }

        # Monkey patch the schema generation
        original_method = job.output_schema.model_json_schema
        job.output_schema.model_json_schema = lambda: complex_schema

        try:
            flattened = job._get_flattened_schema()

            # Should preserve external refs in nested structures
            items_ref = flattened["properties"]["level1"]["properties"]["level2"][
                "items"
            ]["$ref"]
            assert items_ref == "http://external.com/schema#/Type"
        finally:
            # Restore the original method
            job.output_schema.model_json_schema = original_method
