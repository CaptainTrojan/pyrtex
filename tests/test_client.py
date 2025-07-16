# tests/unit/test_client.py

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import uuid

from pydantic import BaseModel
from google.api_core.exceptions import NotFound

from pyrtex.client import Job
from pyrtex.config import InfrastructureConfig, GenerationConfig
from pyrtex.exceptions import ConfigurationError, JobFailedError
from pyrtex.models import BatchResult

from tests.conftest import SimpleInput, SimpleOutput, FileInput, ComplexOutput


class TestJobInitialization:
    """Tests for Job initialization and configuration."""
    
    def test_job_initialization_with_defaults(self, mock_gcp_clients):
        """Test that Job initializes correctly with default config."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        assert job.model == "gemini-2.0-flash-lite-001"
        assert job.output_schema == SimpleOutput
        assert job.prompt_template == "Test: {{ word }}"
        assert isinstance(job.generation_config, GenerationConfig)
        assert isinstance(job.config, InfrastructureConfig)
        assert job.simulation_mode is False
        assert len(job._session_id) == 10
        assert job._requests == []
        assert job._instance_map == {}
        assert job._batch_job is None
        assert job._results_cache is None
    
    def test_job_initialization_with_simulation_mode(self, mock_gcp_clients):
        """Test Job initialization with simulation mode enabled."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True
        )
        
        assert job.simulation_mode is True
    
    def test_job_initialization_with_custom_config(self, mock_gcp_clients):
        """Test Job initialization with custom configuration."""
        config = InfrastructureConfig(
            project_id="custom-project",
            location="us-west1",
            gcs_bucket_name="custom-bucket",
            bq_dataset_id="custom_dataset"
        )
        gen_config = GenerationConfig(temperature=0.5, max_output_tokens=1000)
        
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=ComplexOutput,
            prompt_template="Complex: {{ text }}",
            generation_config=gen_config,
            config=config
        )
        
        assert job.config.project_id == "custom-project"
        assert job.config.location == "us-west1"
        assert job.generation_config.temperature == 0.5
        assert job.generation_config.max_output_tokens == 1000
    
    def test_job_initialization_gcp_failure(self, mocker):
        """Test that Job raises ConfigurationError when GCP initialization fails."""
        # Mock GCP clients to raise an exception
        mocker.patch('google.cloud.storage.Client', side_effect=Exception("Auth failed"))
        
        with pytest.raises(ConfigurationError) as exc_info:
            Job(
                model="gemini-2.0-flash-lite-001",
                output_schema=SimpleOutput,
                prompt_template="Test: {{ word }}"
            )
        
        assert "Failed to initialize GCP clients" in str(exc_info.value)
        assert "gcloud auth application-default login" in str(exc_info.value)
    
    def test_resolve_infra_config_auto_discovery(self, mock_gcp_clients):
        """Test that infrastructure config is auto-resolved from GCP clients."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        # Should use defaults based on discovered project
        assert job.config.project_id == "test-project"
        assert job.config.gcs_bucket_name == "pyrtex-assets-test-project"
        assert job.config.bq_dataset_id == "pyrtex_results"
    
    def test_resolve_infra_config_no_project_id(self, mocker):
        """Test error when project_id cannot be discovered."""
        mock_storage = Mock()
        mock_storage.project = None
        mocker.patch('google.cloud.storage.Client', return_value=mock_storage)
        mocker.patch('google.cloud.bigquery.Client')
        mocker.patch('google.cloud.aiplatform.init')
        
        with pytest.raises(ConfigurationError) as exc_info:
            Job(
                model="gemini-2.0-flash-lite-001",
                output_schema=SimpleOutput,
                prompt_template="Test: {{ word }}"
            )
        
        assert "Could not automatically discover GCP Project ID" in str(exc_info.value)


class TestJobRequests:
    """Tests for adding requests to jobs."""
    
    def test_add_request_success(self, mock_gcp_clients):
        """Test successfully adding a request to a job."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        input_data = SimpleInput(word="hello")
        result = job.add_request("test_key", input_data)
        
        assert result is job  # Should return self for chaining
        assert len(job._requests) == 1
        assert job._requests[0] == ("test_key", input_data)
    
    def test_add_request_after_submission(self, mock_gcp_clients):
        """Test that adding requests after submission raises error."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        job._batch_job = Mock()  # Simulate submitted job
        
        with pytest.raises(RuntimeError) as exc_info:
            job.add_request("test_key", SimpleInput(word="hello"))
        
        assert "Cannot add requests after job has been submitted" in str(exc_info.value)
    
    def test_add_multiple_requests(self, mock_gcp_clients):
        """Test adding multiple requests."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        job.add_request("key1", SimpleInput(word="hello"))
        job.add_request("key2", SimpleInput(word="world"))
        
        assert len(job._requests) == 2
        assert job._requests[0][0] == "key1"
        assert job._requests[1][0] == "key2"


class TestJobSubmission:
    """Tests for job submission logic."""
    
    def test_submit_no_requests(self, mock_gcp_clients):
        """Test that submitting job with no requests raises error."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        with pytest.raises(RuntimeError) as exc_info:
            job.submit()
        
        assert "Cannot submit a job with no requests" in str(exc_info.value)
    
    def test_submit_simulation_mode(self, mock_gcp_clients):
        """Test job submission in simulation mode."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True
        )
        
        job.add_request("test_key", SimpleInput(word="hello"))
        result = job.submit()
        
        assert result is job
        assert job._batch_job == "dummy_job"
    
    def test_submit_dry_run(self, mock_gcp_clients, capsys):
        """Test job submission in dry run mode."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        job.add_request("test_key", SimpleInput(word="hello"))
        
        with patch.object(job, '_create_jsonl_payload', return_value='{"test": "payload"}'):
            result = job.submit(dry_run=True)
        
        assert result is job
        assert job._batch_job is None
        
        captured = capsys.readouterr()
        assert "--- DRY RUN OUTPUT ---" in captured.out
        assert "Dry run enabled. Job was not submitted" in captured.out
    
    def test_submit_success(self, mock_gcp_clients):
        """Test successful job submission."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        job.add_request("test_key", SimpleInput(word="hello"))
        
        with patch.object(job, '_create_jsonl_payload', return_value='{"test": "payload"}'):
            result = job.submit()
        
        assert result is job
        assert job._batch_job is not None
        assert job._batch_job.resource_name == "projects/test-project/locations/us-central1/batchPredictionJobs/test-job"


class TestJobWait:
    """Tests for job wait logic."""
    
    def test_wait_no_job(self, mock_gcp_clients, capsys):
        """Test waiting when no job was submitted."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        result = job.wait()
        
        assert result is job
        captured = capsys.readouterr()
        assert "No job submitted" in captured.out
    
    def test_wait_simulation_mode(self, mock_gcp_clients):
        """Test waiting in simulation mode."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True
        )
        
        job.add_request("test_key", SimpleInput(word="hello"))
        result = job.submit().wait()
        
        assert result is job
    
    def test_wait_placeholder_logic(self, mock_gcp_clients, capsys):
        """Test the placeholder wait logic."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        job._batch_job = Mock()  # Simulate submitted job
        result = job.wait()
        
        assert result is job
        captured = capsys.readouterr()
        assert "WAIT LOGIC NOT YET IMPLEMENTED" in captured.out


class TestJobResults:
    """Tests for job result retrieval."""
    
    def test_results_no_job(self, mock_gcp_clients):
        """Test getting results when no job was submitted."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        with pytest.raises(RuntimeError) as exc_info:
            list(job.results())
        
        assert "Cannot get results for a job that has not been submitted" in str(exc_info.value)
    
    def test_results_simulation_mode(self, mock_gcp_clients):
        """Test getting results in simulation mode."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True
        )
        
        job.add_request("test_key", SimpleInput(word="hello"))
        job.submit()
        
        results = list(job.results())
        
        assert len(results) == 1
        result = results[0]
        assert result.request_key == "test_key"
        assert result.was_successful
        assert isinstance(result.output, SimpleOutput)
        assert "dummy response" in result.raw_response["note"]
    
    def test_results_cached(self, mock_gcp_clients):
        """Test that results are cached after first retrieval."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        job._batch_job = Mock()
        job._results_cache = [BatchResult(request_key="cached", output=SimpleOutput(result="cached"))]
        
        results = list(job.results())
        
        assert len(results) == 1
        assert results[0].request_key == "cached"
    
    def test_results_placeholder_logic(self, mock_gcp_clients, capsys):
        """Test the placeholder results logic."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        job._batch_job = Mock()  # Simulate submitted job
        results = list(job.results())
        
        assert len(results) == 0
        captured = capsys.readouterr()
        assert "RESULTS LOGIC NOT YET IMPLEMENTED" in captured.out


class TestGenerateDummyResults:
    """Tests for dummy result generation in simulation mode."""
    
    def test_generate_dummy_results_simple(self, mock_gcp_clients):
        """Test generating dummy results for simple schema."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True
        )
        
        job.add_request("key1", SimpleInput(word="hello"))
        job.add_request("key2", SimpleInput(word="world"))
        
        results = list(job._generate_dummy_results())
        
        assert len(results) == 2
        assert results[0].request_key == "key1"
        assert results[1].request_key == "key2"
        assert isinstance(results[0].output, SimpleOutput)
        assert isinstance(results[1].output, SimpleOutput)
    
    def test_generate_dummy_results_with_required_fields(self, mock_gcp_clients):
        """Test generating dummy results for schema with required fields."""
        class RequiredFieldOutput(BaseModel):
            required_field: str
            optional_field: str = "default"
        
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=RequiredFieldOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True
        )
        
        job.add_request("key1", SimpleInput(word="hello"))
        
        results = list(job._generate_dummy_results())
        
        assert len(results) == 1
        assert results[0].request_key == "key1"
        assert isinstance(results[0].output, RequiredFieldOutput)
        assert results[0].usage_metadata["promptTokenCount"] == 0
        assert "dummy response" in results[0].raw_response["note"]


class TestCloudResourceSetup:
    """Tests for cloud resource setup."""
    
    def test_setup_cloud_resources_bucket_exists(self, mock_gcp_clients):
        """Test setup when bucket already exists."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        # Should not raise any exceptions
        job._setup_cloud_resources()
        
        # Verify bucket operations were called
        mock_gcp_clients['storage'].get_bucket.assert_called_once()
    
    def test_setup_cloud_resources_bucket_not_found(self, mock_gcp_clients):
        """Test setup when bucket needs to be created."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        # Mock bucket not found, then creation
        mock_gcp_clients['storage'].get_bucket.side_effect = NotFound("Bucket not found")
        
        job._setup_cloud_resources()
        
        # Verify bucket creation was called
        mock_gcp_clients['storage'].create_bucket.assert_called_once()
    
    def test_setup_cloud_resources_dataset_not_found(self, mock_gcp_clients):
        """Test setup when dataset needs to be created."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        # Mock dataset not found, then creation
        mock_gcp_clients['bigquery'].get_dataset.side_effect = NotFound("Dataset not found")
        
        job._setup_cloud_resources()
        
        # Verify dataset creation was called
        mock_gcp_clients['bigquery'].create_dataset.assert_called_once()