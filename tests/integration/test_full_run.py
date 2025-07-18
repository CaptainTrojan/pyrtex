# tests/integration/test_full_run.py

import os
import tempfile
from pathlib import Path
from typing import Union

import pytest
from google.cloud.aiplatform_v1.types import JobState
from pydantic import BaseModel

# Import from the actual package structure
from pyrtex.client import Job
from pyrtex.models import BatchResult

# Skip decorator for tests that require project ID
requires_project_id = pytest.mark.skipif(
    not os.getenv("GOOGLE_PROJECT_ID"),
    reason="Requires GOOGLE_PROJECT_ID environment variable to be set",
)


# Test schemas
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
    keywords: list[str]


class TestDryRun:
    """Test dry run functionality."""

    def test_dry_run_output_text_only(self, mock_gcp_clients, capsys):
        """Verify that dry_run produces plausible JSONL output and doesn't submit."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process this word: {{ word }}",
        )
        job.add_request(request_key="test1", data=SimpleInput(word="hello"))

        job.submit(dry_run=True)

        captured = capsys.readouterr()
        assert "--- DRY RUN OUTPUT ---" in captured.out
        assert "Generated JSONL Payload" in captured.out
        assert '"text": "Process this word: hello"' in captured.out
        assert "extract_info" in captured.out
        assert "Dry run enabled. Job was not submitted." in captured.err

    def test_dry_run_output_with_files(self, mock_gcp_clients, capsys):
        """Verify dry run works with file inputs."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process {{ text }} from the uploaded file",
        )

        test_content = b"This is test file content"
        job.add_request(
            request_key="file_test",
            data=FileInput(text="analyze this", file_content=test_content),
        )

        job.submit(dry_run=True)

        captured = capsys.readouterr()
        assert "--- DRY RUN OUTPUT ---" in captured.out
        assert "file_data" in captured.out
        assert "mime_type" in captured.out
        assert "Process analyze this from the uploaded file" in captured.out

    def test_dry_run_multiple_requests(self, mock_gcp_clients, capsys):
        """Test dry run with multiple requests."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Word: {{ word }}",
        )

        job.add_request("req1", SimpleInput(word="hello"))
        job.add_request("req2", SimpleInput(word="world"))
        job.add_request("req3", SimpleInput(word="test"))

        job.submit(dry_run=True)

        captured = capsys.readouterr()
        assert "--- DRY RUN OUTPUT ---" in captured.out
        assert "Generated JSONL Payload (first 3 lines):" in captured.out
        # Should show all 3 lines since we only have 3 requests
        assert captured.out.count('"id": "req_') == 3


class TestSimulationMode:
    """Test simulation mode functionality."""

    def test_simulation_mode_basic(self, mock_gcp_clients):
        """Verify that simulation_mode returns dummy data without hitting GCP."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process: {{ word }}",
            simulation_mode=True,
        )
        job.add_request(request_key="sim1", data=SimpleInput(word="world"))

        # .submit() and .wait() should be no-ops
        results = list(job.submit().wait().results())

        assert len(results) == 1
        result = results[0]
        assert result.request_key == "sim1"
        assert result.was_successful
        assert isinstance(result.output, SimpleOutput)
        assert result.usage_metadata["promptTokenCount"] == 0
        assert result.usage_metadata["candidatesTokenCount"] == 0
        assert "dummy response" in result.raw_response["note"]

    def test_simulation_mode_multiple_requests(self, mock_gcp_clients):
        """Test simulation mode with multiple requests."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=ComplexOutput,
            prompt_template="Analyze: {{ text }}",
            simulation_mode=True,
        )

        inputs = [
            SimpleInput(word="first"),
            SimpleInput(word="second"),
            SimpleInput(word="third"),
        ]

        for i, input_data in enumerate(inputs):
            job.add_request(f"req_{i}", input_data)

        results = list(job.submit().wait().results())

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.request_key == f"req_{i}"
            assert result.was_successful
            assert isinstance(result.output, ComplexOutput)
            assert result.usage_metadata["totalTokenCount"] == 0

    def test_simulation_mode_with_complex_schema(self, mock_gcp_clients):
        """Test simulation mode with complex output schema."""

        class DetailedOutput(BaseModel):
            title: str
            description: str
            tags: list[str]
            score: float
            metadata: dict[str, str]

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=DetailedOutput,
            prompt_template="Analyze: {{ word }}",
            simulation_mode=True,
        )

        job.add_request("complex_test", SimpleInput(word="analyze"))

        results = list(job.submit().wait().results())

        assert len(results) == 1
        result = results[0]
        assert result.was_successful
        assert isinstance(result.output, DetailedOutput)
        # The dummy data should be schema-compliant
        assert hasattr(result.output, "title")
        assert hasattr(result.output, "description")
        assert hasattr(result.output, "tags")
        assert hasattr(result.output, "score")
        assert hasattr(result.output, "metadata")

    def test_simulation_mode_chaining(self, mock_gcp_clients):
        """Test that simulation mode supports method chaining."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process: {{ word }}",
            simulation_mode=True,
        )

        # Test chaining: add_request -> submit -> wait -> results
        results = list(
            job.add_request("chain_test", SimpleInput(word="chain"))
            .submit()
            .wait()
            .results()
        )

        assert len(results) == 1
        assert results[0].request_key == "chain_test"
        assert results[0].was_successful


class TestRealWorldScenarios:
    """Integration tests that use real GCP services and incur costs."""

    @pytest.mark.incurs_costs
    @requires_project_id
    def test_full_run_simple_text(self):
        """
        Full end-to-end test that submits a real job to Vertex AI.
        Requires GCP authentication and will incur small costs.
        """
        # Simple prompt engineering: Tell the model exactly what to do
        prompt = """The user provided a word: '{{ word }}'.
        Your task is to call the extract_info function with this exact word
        in the 'result' field.
        Make sure to use the function calling capability."""

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template=prompt,
        )
        job.add_request(
            request_key="e2e_test_key", data=SimpleInput(word="pyrtex_works")
        )

        # The magic one-liner
        results = list(job.submit().wait().results())

        assert len(results) == 1
        result = results[0]

        assert result.was_successful
        assert result.request_key == "e2e_test_key"
        assert result.output.result == "pyrtex_works"
        assert result.error is None
        assert result.usage_metadata["totalTokenCount"] > 0

    @pytest.mark.incurs_costs
    @requires_project_id
    def test_full_run_with_file(self):
        """Test full run with file input - uses real GCP services."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test document for analysis.")
            temp_file_path = f.name

        try:
            prompt = """Analyze the uploaded file and the text "{{ text }}".
            Call the extract_info function with a summary in the 'result' field."""

            job = Job(
                model="gemini-2.0-flash-lite-001",
                output_schema=SimpleOutput,
                prompt_template=prompt,
            )

            job.add_request(
                request_key="file_test",
                data=FileInput(text="additional context", file_path=temp_file_path),
            )

            results = list(job.submit().wait().results())

            assert len(results) == 1
            result = results[0]

            assert result.was_successful
            assert result.request_key == "file_test"
            assert result.output.result is not None
            assert len(result.output.result) > 0
            assert result.usage_metadata["totalTokenCount"] > 0

        finally:
            # Clean up
            os.unlink(temp_file_path)

    @pytest.mark.incurs_costs
    @requires_project_id
    def test_full_run_batch_processing(self):
        """Test batch processing with multiple requests - uses real GCP services."""
        prompt = """Process the word "{{ word }}" and return it in the result field."""

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template=prompt,
        )

        test_words = ["alpha", "beta", "gamma", "delta", "epsilon"]

        for i, word in enumerate(test_words):
            job.add_request(f"batch_{i}", SimpleInput(word=word))

        results = list(job.submit().wait().results())

        assert len(results) == len(test_words)

        # Check that all results are successful
        successful_results = [r for r in results if r.was_successful]
        assert len(successful_results) == len(test_words)

        # CRITICAL: Verify that the mapping between request keys and results is correct
        # Results from BigQuery may not come back in the same order as submitted
        result_by_key = {r.request_key: r for r in results}

        for i, word in enumerate(test_words):
            request_key = f"batch_{i}"
            assert (
                request_key in result_by_key
            ), f"Missing result for request key: {request_key}"

            result = result_by_key[request_key]
            assert (
                result.was_successful
            ), f"Request {request_key} failed: {result.error}"
            assert (
                result.output.result == word
            ), (
                f"Expected '{word}' but got '{result.output.result}' "
                f"for key {request_key}"
            )

    @pytest.mark.incurs_costs
    @requires_project_id
    def test_model_robustness_with_simple_prompt(self):
        """
        Test that the model can handle simple prompts and still use
        function calling.
        """
        # Even though this prompt doesn't explicitly instruct function calling,
        # the model should be smart enough to use it because tools are available
        prompt = """Just respond with plain text: {{ word }}"""

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template=prompt,
        )

        job.add_request("robustness_test", SimpleInput(word="test"))

        results = list(job.submit().wait().results())

        assert len(results) == 1
        result = results[0]

        # The model should succeed even with a simple prompt because it's smart
        # enough to use the available function calling tools
        assert result.was_successful
        assert result.output is not None
        assert result.output.result == "test"
        assert result.error is None


class TestErrorScenarios:
    """Test error scenarios that don't require real GCP."""

    def test_submit_without_requests(self, mock_gcp_clients):
        """Test error when submitting without requests."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        with pytest.raises(RuntimeError, match="Cannot submit a job with no requests"):
            job.submit()

    def test_results_without_submission(self, mock_gcp_clients):
        """Test error when getting results without submission."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        with pytest.raises(
            RuntimeError,
            match="Cannot get results for a job that has not been submitted",
        ):
            list(job.results())

    def test_add_request_after_submission(self, mock_gcp_clients):
        """Test error when adding requests after submission."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True,
        )

        job.add_request("test1", SimpleInput(word="hello"))
        job.submit()  # This sets _batch_job in simulation mode

        with pytest.raises(
            RuntimeError, match="Cannot add requests after job has been submitted"
        ):
            job.add_request("test2", SimpleInput(word="world"))


class TestRealBigQueryResultParsing:
    """Test the real BigQuery result parsing logic without mocking."""

    def test_bigquery_result_parsing_with_mock_data(self, mock_gcp_clients):
        """Test the BigQuery result parsing logic with mock row data."""
        import json
        from unittest.mock import Mock

        from pyrtex.client import Job
        from pyrtex.models import BatchResult

        # Create a job with real configuration but mock the BigQuery client
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        # Set up the job as if it was submitted
        job._instance_map = {
            "req_00000_12345678": "test_key_1",
            "req_00001_87654321": "test_key_2",
        }

        # Create mock BigQuery rows that simulate real response data
        mock_rows = [
            Mock(
                id="req_00000_12345678",
                response=json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "functionCall": {
                                                "name": "extract_info",
                                                "args": {"result": "test_output_1"},
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
                ),
            ),
            Mock(
                id="req_00001_87654321",
                response=json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "functionCall": {
                                                "name": "extract_info",
                                                "args": {"result": "test_output_2"},
                                            }
                                        }
                                    ]
                                }
                            }
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 12,
                            "candidatesTokenCount": 8,
                            "totalTokenCount": 20,
                        },
                    }
                ),
            ),
        ]

        # Mock the BigQuery client and job
        mock_batch_job = Mock()
        mock_batch_job.state = JobState.JOB_STATE_SUCCEEDED
        mock_batch_job.output_info.bigquery_output_table = "bq://project.dataset.table"
        job._batch_job = mock_batch_job

        # Mock the BigQuery client directly
        mock_bigquery_client = Mock()
        mock_query_job = Mock()
        mock_query_job.result.return_value = mock_rows
        mock_bigquery_client.query.return_value = mock_query_job
        job._bigquery_client = mock_bigquery_client

        # Test the actual result parsing logic
        results = list(job.results())

        # Verify results
        assert len(results) == 2

        result1 = results[0]
        assert result1.request_key == "test_key_1"
        assert result1.output.result == "test_output_1"
        assert result1.usage_metadata["totalTokenCount"] == 15
        assert result1.error is None

        result2 = results[1]
        assert result2.request_key == "test_key_2"
        assert result2.output.result == "test_output_2"
        assert result2.usage_metadata["totalTokenCount"] == 20
        assert result2.error is None

        # Verify the BigQuery query was called correctly
        expected_query = "SELECT id, response FROM `project.dataset.table`"
        job._bigquery_client.query.assert_called_once_with(expected_query)

    def test_bigquery_result_parsing_with_model_errors(self, mock_gcp_clients):
        """Test BigQuery result parsing when model returns errors."""
        import json
        from unittest.mock import Mock

        from pyrtex.client import Job

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        job._instance_map = {"req_00000_12345678": "test_key_1"}

        # Create mock row with invalid response (no function call)
        mock_rows = [
            Mock(
                id="req_00000_12345678",
                response=json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {"text": "I cannot follow the instructions"}
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
                ),
            )
        ]

        # Mock the BigQuery client and job
        mock_batch_job = Mock()
        mock_batch_job.state = JobState.JOB_STATE_SUCCEEDED
        mock_batch_job.output_info.bigquery_output_table = "bq://project.dataset.table"
        job._batch_job = mock_batch_job

        # Mock the BigQuery client directly
        mock_bigquery_client = Mock()
        mock_query_job = Mock()
        mock_query_job.result.return_value = mock_rows
        mock_bigquery_client.query.return_value = mock_query_job
        job._bigquery_client = mock_bigquery_client

        # Test the result parsing with errors
        results = list(job.results())

        # Verify error handling
        assert len(results) == 1
        result = results[0]
        assert result.request_key == "test_key_1"
        assert result.output is None
        assert result.error is not None
        assert "Failed to parse model output" in result.error
        assert result.usage_metadata["totalTokenCount"] == 15

    def test_batch_result_order_independence(self, mock_gcp_clients):
        """Test that result ordering is independent of BigQuery result order."""
        import json
        from unittest.mock import Mock

        from pyrtex.client import Job

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
        )

        # Set up instance map with multiple requests
        job._instance_map = {
            "req_00000_batch_0": "batch_0",
            "req_00001_batch_1": "batch_1",
            "req_00002_batch_2": "batch_2",
        }

        # Create mock rows that come back in DIFFERENT order than submitted
        # This simulates BigQuery returning results in arbitrary order
        mock_rows = [
            # Second request returned first
            Mock(
                id="req_00001_batch_1",
                response=json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "functionCall": {
                                                "name": "extract_info",
                                                "args": {"result": "beta"},
                                            }
                                        }
                                    ]
                                }
                            }
                        ],
                        "usageMetadata": {"totalTokenCount": 10},
                    }
                ),
            ),
            # Third request returned second
            Mock(
                id="req_00002_batch_2",
                response=json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "functionCall": {
                                                "name": "extract_info",
                                                "args": {"result": "gamma"},
                                            }
                                        }
                                    ]
                                }
                            }
                        ],
                        "usageMetadata": {"totalTokenCount": 12},
                    }
                ),
            ),
            # First request returned last
            Mock(
                id="req_00000_batch_0",
                response=json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "functionCall": {
                                                "name": "extract_info",
                                                "args": {"result": "alpha"},
                                            }
                                        }
                                    ]
                                }
                            }
                        ],
                        "usageMetadata": {"totalTokenCount": 8},
                    }
                ),
            ),
        ]

        # Mock the BigQuery client and job
        mock_batch_job = Mock()
        mock_batch_job.state = JobState.JOB_STATE_SUCCEEDED
        mock_batch_job.output_info.bigquery_output_table = "bq://project.dataset.table"
        job._batch_job = mock_batch_job

        # Mock the BigQuery client
        mock_bigquery_client = Mock()
        mock_query_job = Mock()
        mock_query_job.result.return_value = mock_rows
        mock_bigquery_client.query.return_value = mock_query_job
        job._bigquery_client = mock_bigquery_client

        # Get results and verify correct mapping regardless of order
        results = list(job.results())

        assert len(results) == 3

        # Build lookup by request key
        result_by_key = {r.request_key: r for r in results}

        # Verify each request maps to correct result
        assert "batch_0" in result_by_key
        assert result_by_key["batch_0"].output.result == "alpha"

        assert "batch_1" in result_by_key
        assert result_by_key["batch_1"].output.result == "beta"

        assert "batch_2" in result_by_key
        assert result_by_key["batch_2"].output.result == "gamma"
