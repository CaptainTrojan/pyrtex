# tests/conftest.py

from pathlib import Path
from typing import Union
from unittest.mock import MagicMock, Mock

import google.cloud.aiplatform as aiplatform
import google.cloud.bigquery as bigquery
import google.cloud.storage as storage
import pytest
from google.cloud.aiplatform_v1.types import JobState
from pydantic import BaseModel


# Test schemas for reuse across tests
class SimpleInput(BaseModel):
    word: str


class SimpleOutput(BaseModel):
    result: str


class FileInput(BaseModel):
    text: str = "default text"  # Make text optional with default
    file_path: Union[str, Path, bytes, None] = None
    file_content: Union[bytes, None] = None
    image: Union[str, Path, bytes, None] = None  # Support legacy tests


class ComplexOutput(BaseModel):
    summary: str
    confidence: float


@pytest.fixture
def mock_storage_client():
    """Mock Google Cloud Storage client."""
    mock_client = Mock(spec=storage.Client)
    mock_client.project = "test-project"

    # Mock bucket operations
    mock_bucket = Mock()
    mock_blob = Mock()
    mock_bucket.blob.return_value = mock_blob
    mock_client.bucket.return_value = mock_bucket
    mock_client.get_bucket.return_value = mock_bucket
    mock_client.create_bucket.return_value = mock_bucket

    return mock_client


@pytest.fixture
def mock_bigquery_client():
    """Mock Google Cloud BigQuery client."""
    mock_client = Mock(spec=bigquery.Client)
    mock_client.project = "test-project"

    # Mock dataset operations
    mock_dataset = Mock()
    mock_dataset.default_table_expiration_ms = 24 * 60 * 60 * 1000
    mock_client.get_dataset.return_value = mock_dataset
    mock_client.create_dataset.return_value = mock_dataset
    mock_client.update_dataset.return_value = None

    # Mock query operations
    mock_query_job = Mock()
    mock_query_job.result.return_value = []  # Empty result by default
    mock_client.query.return_value = mock_query_job

    return mock_client


@pytest.fixture
def mock_batch_job():
    """Mock Vertex AI BatchPredictionJob with refresh method for status property."""
    mock_job = Mock(spec=aiplatform.BatchPredictionJob)
    mock_job.resource_name = (
        "projects/test-project/locations/us-central1/batchPredictionJobs/test-job"
    )
    mock_job.name = "test-job"
    mock_job.state = JobState.JOB_STATE_SUCCEEDED
    # Add refresh method explicitly (may not exist in spec in some lib versions)
    mock_job.refresh = Mock()

    # Mock output_info for results method
    mock_output_info = Mock()
    mock_output_info.bigquery_output_table = "bq://test-project.test_dataset.test_table"
    mock_job.output_info = mock_output_info

    return mock_job


@pytest.fixture
def mock_gcp_auth(mocker):
    """Mock GCP authentication to avoid slow real auth calls during unit tests."""
    mock_credentials = Mock()
    mock_credentials.project_id = "test-project"

    # Mock the main authentication entry point
    mocker.patch("google.auth.default", return_value=(mock_credentials, "test-project"))

    # Also mock service account methods for comprehensive coverage
    mocker.patch(
        "google.oauth2.service_account.Credentials.from_service_account_info",
        return_value=mock_credentials,
    )
    mocker.patch(
        "google.oauth2.service_account.Credentials.from_service_account_file",
        return_value=mock_credentials,
    )

    return mock_credentials


@pytest.fixture
def mock_gcp_clients(
    mocker, mock_gcp_auth, mock_storage_client, mock_bigquery_client, mock_batch_job
):
    """Mock all GCP clients and services including fast authentication."""
    mocker.patch("google.cloud.storage.Client", return_value=mock_storage_client)
    mocker.patch("google.cloud.bigquery.Client", return_value=mock_bigquery_client)
    mocker.patch("google.cloud.aiplatform.init")
    mocker.patch(
        "google.cloud.aiplatform.BatchPredictionJob.submit", return_value=mock_batch_job
    )

    return {
        "storage": mock_storage_client,
        "bigquery": mock_bigquery_client,
        "batch_job": mock_batch_job,
        "credentials": mock_gcp_auth,
    }


@pytest.fixture
def mock_gcp_clients_no_auth(
    mocker, mock_storage_client, mock_bigquery_client, mock_batch_job
):
    """Mock GCP clients without authentication mocking for auth-specific tests."""
    mocker.patch("google.cloud.storage.Client", return_value=mock_storage_client)
    mocker.patch("google.cloud.bigquery.Client", return_value=mock_bigquery_client)
    mocker.patch("google.cloud.aiplatform.init")
    mocker.patch(
        "google.cloud.aiplatform.BatchPredictionJob.submit", return_value=mock_batch_job
    )

    return {
        "storage": mock_storage_client,
        "bigquery": mock_bigquery_client,
        "batch_job": mock_batch_job,
    }


@pytest.fixture
def sample_bigquery_results():
    """Sample BigQuery results for testing result parsing."""
    return [
        {
            "id": "req_00001_abcd1234",
            "response": {
                "candidates": [
                    {"content": {"parts": [{"text": '{"result": "test_output"}'}]}}
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                },
            },
        },
        {
            "id": "req_00002_efgh5678",
            "response": {
                "candidates": [
                    {"content": {"parts": [{"text": "Error: Invalid response format"}]}}
                ],
                "usageMetadata": {
                    "promptTokenCount": 8,
                    "candidatesTokenCount": 3,
                    "totalTokenCount": 11,
                },
            },
        },
    ]
