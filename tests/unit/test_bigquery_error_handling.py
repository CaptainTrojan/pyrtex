# tests/unit/test_bigquery_error_handling.py

import json
from unittest.mock import Mock

import pytest
from google.cloud.aiplatform_v1.types import JobState

from pyrtex.client import Job
from tests.conftest import SimpleInput, SimpleOutput


class TestBigQueryErrorHandling:
    """Test BigQuery error handling scenarios for coverage."""

    def test_status_column_with_error_dict(self, mock_gcp_clients):
        """Test status column with error dictionary."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        job._instance_map = {"req_00000_12345678": "test_key_1"}

        # Create mock row with status containing error dict
        mock_rows = [
            Mock(
                id="req_00000_12345678",
                status=json.dumps({
                    "error": {
                        "code": 400,
                        "message": "Invalid request format"
                    }
                }),
                response=None,
            )
        ]

        # Mock the BigQuery client and job
        mock_batch_job = Mock()
        mock_batch_job.state = JobState.JOB_STATE_SUCCEEDED
        mock_batch_job.output_info.bigquery_output_table = "bq://project.dataset.table"
        job._batch_job = mock_batch_job

        mock_bigquery_client = Mock()
        mock_query_job = Mock()
        mock_query_job.result.return_value = mock_rows
        mock_bigquery_client.query.return_value = mock_query_job
        job._bigquery_client = mock_bigquery_client

        results = list(job.results())

        assert len(results) == 1
        result = results[0]
        assert result.request_key == "test_key_1"
        assert result.output is None
        assert "API Error 400: Invalid request format" in result.error

    def test_status_column_with_error_string(self, mock_gcp_clients):
        """Test status column with error string."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        job._instance_map = {"req_00000_12345678": "test_key_1"}

        # Create mock row with status containing error string
        mock_rows = [
            Mock(
                id="req_00000_12345678",
                status=json.dumps({
                    "error": "Something went wrong"
                }),
                response=None,
            )
        ]

        # Mock the BigQuery client and job
        mock_batch_job = Mock()
        mock_batch_job.state = JobState.JOB_STATE_SUCCEEDED
        mock_batch_job.output_info.bigquery_output_table = "bq://project.dataset.table"
        job._batch_job = mock_batch_job

        mock_bigquery_client = Mock()
        mock_query_job = Mock()
        mock_query_job.result.return_value = mock_rows
        mock_bigquery_client.query.return_value = mock_query_job
        job._bigquery_client = mock_bigquery_client

        results = list(job.results())

        assert len(results) == 1
        result = results[0]
        assert result.request_key == "test_key_1"
        assert result.output is None
        assert "API Error: Something went wrong" in result.error

    def test_status_column_without_error_key(self, mock_gcp_clients):
        """Test status column without error key."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        job._instance_map = {"req_00000_12345678": "test_key_1"}

        # Create mock row with status without error key
        mock_rows = [
            Mock(
                id="req_00000_12345678",
                status=json.dumps({
                    "some_other_field": "some_value"
                }),
                response=None,
            )
        ]

        # Mock the BigQuery client and job
        mock_batch_job = Mock()
        mock_batch_job.state = JobState.JOB_STATE_SUCCEEDED
        mock_batch_job.output_info.bigquery_output_table = "bq://project.dataset.table"
        job._batch_job = mock_batch_job

        mock_bigquery_client = Mock()
        mock_query_job = Mock()
        mock_query_job.result.return_value = mock_rows
        mock_bigquery_client.query.return_value = mock_query_job
        job._bigquery_client = mock_bigquery_client

        results = list(job.results())

        assert len(results) == 1
        result = results[0]
        assert result.request_key == "test_key_1"
        assert result.output is None
        assert "Request failed with status:" in result.error

    def test_status_column_json_decode_error(self, mock_gcp_clients):
        """Test status column with invalid JSON."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        job._instance_map = {"req_00000_12345678": "test_key_1"}

        # Create mock row with invalid JSON status
        mock_rows = [
            Mock(
                id="req_00000_12345678",
                status="invalid json {{{",
                response=None,
            )
        ]

        # Mock the BigQuery client and job
        mock_batch_job = Mock()
        mock_batch_job.state = JobState.JOB_STATE_SUCCEEDED
        mock_batch_job.output_info.bigquery_output_table = "bq://project.dataset.table"
        job._batch_job = mock_batch_job

        mock_bigquery_client = Mock()
        mock_query_job = Mock()
        mock_query_job.result.return_value = mock_rows
        mock_bigquery_client.query.return_value = mock_query_job
        job._bigquery_client = mock_bigquery_client

        results = list(job.results())

        assert len(results) == 1
        result = results[0]
        assert result.request_key == "test_key_1"
        assert result.output is None
        assert "Request failed with status:" in result.error

    def test_empty_response(self, mock_gcp_clients):
        """Test empty response handling."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        job._instance_map = {"req_00000_12345678": "test_key_1"}

        # Create mock row with empty response
        mock_rows = [
            Mock(
                id="req_00000_12345678",
                status=None,
                response="",
            )
        ]

        # Mock the BigQuery client and job
        mock_batch_job = Mock()
        mock_batch_job.state = JobState.JOB_STATE_SUCCEEDED
        mock_batch_job.output_info.bigquery_output_table = "bq://project.dataset.table"
        job._batch_job = mock_batch_job

        mock_bigquery_client = Mock()
        mock_query_job = Mock()
        mock_query_job.result.return_value = mock_rows
        mock_bigquery_client.query.return_value = mock_query_job
        job._bigquery_client = mock_bigquery_client

        results = list(job.results())

        assert len(results) == 1
        result = results[0]
        assert result.request_key == "test_key_1"
        assert result.output is None
        assert "Empty response from API" in result.error

    def test_response_json_decode_error(self, mock_gcp_clients):
        """Test response JSON decode error."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        job._instance_map = {"req_00000_12345678": "test_key_1"}

        # Create mock row with invalid JSON response
        mock_rows = [
            Mock(
                id="req_00000_12345678",
                status=None,
                response="invalid json {{{",
            )
        ]

        # Mock the BigQuery client and job
        mock_batch_job = Mock()
        mock_batch_job.state = JobState.JOB_STATE_SUCCEEDED
        mock_batch_job.output_info.bigquery_output_table = "bq://project.dataset.table"
        job._batch_job = mock_batch_job

        mock_bigquery_client = Mock()
        mock_query_job = Mock()
        mock_query_job.result.return_value = mock_rows
        mock_bigquery_client.query.return_value = mock_query_job
        job._bigquery_client = mock_bigquery_client

        results = list(job.results())

        assert len(results) == 1
        result = results[0]
        assert result.request_key == "test_key_1"
        assert result.output is None
        assert "Failed to parse response JSON:" in result.error

    def test_response_error_dict(self, mock_gcp_clients):
        """Test response with error dictionary."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        job._instance_map = {"req_00000_12345678": "test_key_1"}

        # Create mock row with response containing error dict
        mock_rows = [
            Mock(
                id="req_00000_12345678",
                status=None,
                response=json.dumps({
                    "error": {
                        "code": 500,
                        "message": "Internal server error"
                    },
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 0,
                        "totalTokenCount": 10,
                    },
                }),
            )
        ]

        # Mock the BigQuery client and job
        mock_batch_job = Mock()
        mock_batch_job.state = JobState.JOB_STATE_SUCCEEDED
        mock_batch_job.output_info.bigquery_output_table = "bq://project.dataset.table"
        job._batch_job = mock_batch_job

        mock_bigquery_client = Mock()
        mock_query_job = Mock()
        mock_query_job.result.return_value = mock_rows
        mock_bigquery_client.query.return_value = mock_query_job
        job._bigquery_client = mock_bigquery_client

        results = list(job.results())

        assert len(results) == 1
        result = results[0]
        assert result.request_key == "test_key_1"
        assert result.output is None
        assert "Response Error 500: Internal server error" in result.error

    def test_response_error_string(self, mock_gcp_clients):
        """Test response with error string."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        job._instance_map = {"req_00000_12345678": "test_key_1"}

        # Create mock row with response containing error string
        mock_rows = [
            Mock(
                id="req_00000_12345678",
                status=None,
                response=json.dumps({
                    "error": "Request failed",
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 0,
                        "totalTokenCount": 10,
                    },
                }),
            )
        ]

        # Mock the BigQuery client and job
        mock_batch_job = Mock()
        mock_batch_job.state = JobState.JOB_STATE_SUCCEEDED
        mock_batch_job.output_info.bigquery_output_table = "bq://project.dataset.table"
        job._batch_job = mock_batch_job

        mock_bigquery_client = Mock()
        mock_query_job = Mock()
        mock_query_job.result.return_value = mock_rows
        mock_bigquery_client.query.return_value = mock_query_job
        job._bigquery_client = mock_bigquery_client

        results = list(job.results())

        assert len(results) == 1
        result = results[0]
        assert result.request_key == "test_key_1"
        assert result.output is None
        assert "Response Error: Request failed" in result.error

    def test_status_column_dict_object(self, mock_gcp_clients):
        """Test status column as dict object (not JSON string)."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        job._instance_map = {"req_00000_12345678": "test_key_1"}

        # Create mock row with status as dict (not JSON string)
        mock_rows = [
            Mock(
                id="req_00000_12345678",
                status={
                    "error": {
                        "code": 403,
                        "message": "Access denied"
                    }
                },
                response=None,
            )
        ]

        # Mock the BigQuery client and job
        mock_batch_job = Mock()
        mock_batch_job.state = JobState.JOB_STATE_SUCCEEDED
        mock_batch_job.output_info.bigquery_output_table = "bq://project.dataset.table"
        job._batch_job = mock_batch_job

        mock_bigquery_client = Mock()
        mock_query_job = Mock()
        mock_query_job.result.return_value = mock_rows
        mock_bigquery_client.query.return_value = mock_query_job
        job._bigquery_client = mock_bigquery_client

        results = list(job.results())

        assert len(results) == 1
        result = results[0]
        assert result.request_key == "test_key_1"
        assert result.output is None
        assert "API Error 403: Access denied" in result.error
