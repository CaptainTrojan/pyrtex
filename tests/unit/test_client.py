# tests/unit/test_client.py

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import uuid

from pyrtex.client import Job
from pyrtex.config import InfrastructureConfig, GenerationConfig
from pyrtex.exceptions import ConfigurationError, JobFailedError
from tests.conftest import SimpleInput, SimpleOutput, FileInput


class TestJobInitialization:
    """Test Job class initialization and configuration."""
    
    def test_job_initialization_with_defaults(self, mock_gcp_clients):
        """Test that Job initializes correctly with default config."""
        job = Job(
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        assert job.model == "gemini-1.5-flash"
        assert job.output_schema == SimpleOutput
        assert job.prompt_template == "Test: {{ word }}"
        assert job.simulation_mode is False
        assert isinstance(job.generation_config, GenerationConfig)
        assert isinstance(job.config, InfrastructureConfig)
    
    def test_job_initialization_with_simulation_mode(self, mock_gcp_clients):
        """Test that Job initializes correctly with simulation mode."""
        job = Job(
            model="gemini-1.5-pro",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True
        )
        
        assert job.simulation_mode is True
    
    def test_job_initialization_configuration_error(self, mocker):
        """Test that Job raises ConfigurationError when GCP clients fail."""
        mocker.patch('google.cloud.storage.Client', side_effect=Exception("Auth failed"))
        
        with pytest.raises(ConfigurationError) as exc_info:
            Job(
                model="gemini-1.5-flash",
                output_schema=SimpleOutput,
                prompt_template="Test: {{ word }}"
            )
        
        assert "Failed to initialize GCP clients" in str(exc_info.value)
        assert "gcloud auth application-default login" in str(exc_info.value)
    
    def test_project_id_discovery_failure(self, mocker):
        """Test ConfigurationError when project ID cannot be discovered."""
        mock_storage_client = Mock()
        mock_storage_client.project = None
        
        mocker.patch('google.cloud.storage.Client', return_value=mock_storage_client)
        mocker.patch('google.cloud.bigquery.Client')
        mocker.patch('google.cloud.aiplatform.init')
        
        with pytest.raises(ConfigurationError) as exc_info:
            Job(
                model="gemini-1.5-flash",
                output_schema=SimpleOutput,
                prompt_template="Test: {{ word }}"
            )
        
        assert "Could not automatically discover GCP Project ID" in str(exc_info.value)


class TestJobRequestManagement:
    """Test adding requests to jobs."""
    
    def test_add_request_success(self, mock_gcp_clients):
        """Test successfully adding a request to a job."""
        job = Job(
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
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
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
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
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        with pytest.raises(RuntimeError) as exc_info:
            job.submit()
        
        assert "Cannot submit a job with no requests" in str(exc_info.value)
    
    def test_submit_simulation_mode(self, mock_gcp_clients):
        """Test that simulation mode skips actual submission."""
        job = Job(
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True
        )
        
        job.add_request("key1", SimpleInput(word="hello"))
        result = job.submit()
        
        assert result is job
        assert job._batch_job == "dummy_job"
    
    def test_submit_dry_run(self, mock_gcp_clients, capsys):
        """Test that dry run shows payload without submitting."""
        job = Job(
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
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
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        job.add_request("key1", SimpleInput(word="hello"))
        result = job.submit()
        
        assert result is job
        assert job._batch_job is not None
        mock_gcp_clients['storage'].bucket.assert_called()


class TestJobWaiting:
    """Test job waiting logic."""
    
    def test_wait_simulation_mode(self, mock_gcp_clients):
        """Test that wait() skips in simulation mode."""
        job = Job(
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True
        )
        
        result = job.wait()
        assert result is job
    
    def test_wait_without_submission(self, mock_gcp_clients, capsys):
        """Test wait() behavior when no job was submitted."""
        job = Job(
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        result = job.wait()
        captured = capsys.readouterr()
        
        assert "No job submitted, nothing to wait for" in captured.out
        assert result is job


class TestJobResults:
    """Test job results retrieval."""
    
    def test_results_simulation_mode(self, mock_gcp_clients):
        """Test that results() returns dummy data in simulation mode."""
        job = Job(
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True
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
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        with pytest.raises(RuntimeError) as exc_info:
            list(job.results())
        
        assert "Cannot get results for a job that has not been submitted" in str(exc_info.value)
    
    def test_results_uses_cache(self, mock_gcp_clients):
        """Test that results() uses cache when available."""
        job = Job(
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        # Set up mock cache
        from pyrtex.models import BatchResult
        cached_result = BatchResult(
            request_key="cached_key",
            output=SimpleOutput(result="cached"),
            raw_response={"cached": True}
        )
        job._results_cache = [cached_result]
        job._batch_job = Mock()  # Simulate submitted job
        
        results = list(job.results())
        
        assert len(results) == 1
        assert results[0].request_key == "cached_key"
        assert results[0].output.result == "cached"


class TestDummyResultsGeneration:
    """Test the _generate_dummy_results method."""
    
    def test_generate_dummy_results_simple_schema(self, mock_gcp_clients):
        """Test dummy results generation for simple schema."""
        job = Job(
            model="gemini-1.5-flash",
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
            model="gemini-1.5-flash",
            output_schema=RequiredFieldsOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True
        )
        
        job.add_request("key1", SimpleInput(word="hello"))
        
        results = list(job._generate_dummy_results())
        
        assert len(results) == 1
        assert results[0].was_successful
        assert isinstance(results[0].output, RequiredFieldsOutput)
