# tests/unit/test_usage_metadata.py

from unittest.mock import Mock

import pytest

from pyrtex.client import Job
from tests.conftest import SimpleInput, SimpleOutput


class TestUsageMetadataProcessing:
    """Test the usage metadata processing functionality."""

    def test_process_usage_metadata_none_input(self, mock_gcp_clients):
        """Test _process_usage_metadata with None input."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        # Test with None
        result = job._process_usage_metadata(None)
        assert result is None

        # Test with empty dict
        result = job._process_usage_metadata({})
        assert result == {}

    def test_process_usage_metadata_empty_candidates_list(self, mock_gcp_clients):
        """Test _process_usage_metadata with empty candidatesTokensDetails list."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        # Test with empty candidatesTokensDetails list
        usage_metadata = {
            "candidatesTokensDetails": [],
            "promptTokenCount": 10,
            "totalTokenCount": 15,
        }

        result = job._process_usage_metadata(usage_metadata)

        # Should not modify the empty list
        assert result["candidatesTokensDetails"] == []
        assert result["promptTokenCount"] == 10
        assert result["totalTokenCount"] == 15

    def test_process_usage_metadata_missing_token_count(self, mock_gcp_clients):
        """Test _process_usage_metadata with missing tokenCount in details."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        # Test with candidatesTokensDetails missing tokenCount
        usage_metadata = {
            "candidatesTokensDetails": [{"modality": "TEXT"}],  # missing tokenCount
            "promptTokensDetails": [{"modality": "TEXT"}],  # missing tokenCount
            "totalTokenCount": 15,
        }

        result = job._process_usage_metadata(usage_metadata)

        # Should not modify the structure when tokenCount is missing
        assert result["candidatesTokensDetails"] == [{"modality": "TEXT"}]
        assert result["promptTokensDetails"] == [{"modality": "TEXT"}]
        assert result["totalTokenCount"] == 15

    def test_process_usage_metadata_empty_prompt_list(self, mock_gcp_clients):
        """Test _process_usage_metadata with empty promptTokensDetails list."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        # Test with empty promptTokensDetails list
        usage_metadata = {
            "candidatesTokensDetails": [{"tokenCount": 5}],
            "promptTokensDetails": [],  # empty list
            "totalTokenCount": 15,
        }

        result = job._process_usage_metadata(usage_metadata)

        # Should process candidatesTokensDetails but not modify empty
        # promptTokensDetails
        assert result["candidatesTokensDetails"] == 5
        assert result["promptTokensDetails"] == []
        assert result["totalTokenCount"] == 15

    def test_process_usage_metadata_successful_extraction(self, mock_gcp_clients):
        """Test _process_usage_metadata with successful token extraction."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        # Test with valid token details
        usage_metadata = {
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 8}],
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 63}],
            "totalTokenCount": 71,
            "trafficType": "ON_DEMAND",
        }

        result = job._process_usage_metadata(usage_metadata)

        # Should extract token counts correctly
        assert result["candidatesTokensDetails"] == 8
        assert result["promptTokensDetails"] == 63
        assert result["totalTokenCount"] == 71
        assert result["trafficType"] == "ON_DEMAND"


class TestBigQueryResultParsing:
    """Test BigQuery result parsing edge cases."""

    def test_results_with_dict_response(self, mock_gcp_clients):
        """Test results parsing when BigQuery returns dict instead of JSON string."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        # Set up the job as submitted and succeeded
        job._instance_map = {"req_00000_12345678": "test_key"}

        # Create mock row where response is already a dict (not JSON string)
        mock_row = Mock()
        mock_row.id = "req_00000_12345678"
        mock_row.status = None  # No error status
        mock_row.response = {  # This is a dict, not a JSON string
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

        # Mock BigQuery client and job
        from google.cloud.aiplatform_v1.types import JobState

        mock_batch_job = Mock()
        mock_batch_job.state = JobState.JOB_STATE_SUCCEEDED
        mock_batch_job.output_info.bigquery_output_table = "bq://project.dataset.table"
        job._batch_job = mock_batch_job

        mock_bigquery_client = Mock()
        mock_query_job = Mock()
        mock_query_job.result.return_value = [mock_row]
        mock_bigquery_client.query.return_value = mock_query_job
        job._bigquery_client = mock_bigquery_client

        # Test the result parsing - this should cover line 306
        results = list(job.results())

        assert len(results) == 1
        result = results[0]
        assert result.request_key == "test_key"
        assert result.output.result == "test_output"
        assert result.was_successful
