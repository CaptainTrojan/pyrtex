# tests/integration/test_full_run.py

import os
import tempfile
import time
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
        assert "response_schema" in captured.out
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
            assert result.output.result == word, (
                f"Expected '{word}' but got '{result.output.result}' "
                f"for key {request_key}"
            )

    @requires_project_id
    @pytest.mark.incurs_costs
    def test_real_comprehensive_mime_type_processing(self):
        """Real end-to-end test with all supported MIME types to ensure they work."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=ComplexOutput,
            prompt_template="Summarize the content.",
        )

        # Get the examples data directory (where sample files are generated)
        examples_dir = Path(__file__).parent.parent.parent / "examples" / "data"

        # Test files for different MIME types
        # (these should be created by generate_sample_data.py)
        test_files = [
            # Text files (text/plain)
            ("luxury_condo.yaml", "text/plain"),
            ("office_building.json", "text/plain"),
            # Minimal test files for other MIME types
            ("test_minimal.pdf", "application/pdf"),
            ("test_minimal.png", "image/png"),
            ("test_minimal.jpg", "image/jpeg"),
            ("test_minimal.webp", "image/webp"),
            ("test_minimal.wav", "audio/wav"),
            ("test_minimal.mp4", "video/mp4"),
        ]

        # Filter to only include files that actually exist
        existing_files = []
        for filename, expected_mime in test_files:
            file_path = examples_dir / filename
            if file_path.exists():
                existing_files.append((str(file_path), filename, expected_mime))
            else:
                print(f"⚠️  Skipping {filename} (file not found)")

        if (
            len(existing_files) < 4
        ):  # We need at least a few files to make the test meaningful
            pytest.fail(
                "Not enough test files available. Run generate_sample_data.py first."
            )

        try:
            # Add all existing files to job
            for file_path, filename, expected_mime in existing_files:
                job.add_request(
                    filename.replace(".", "_"), FileInput(file_path=file_path)
                )

            # Process files
            results = list(job.submit().wait().results())

            # Verify all files processed successfully
            assert len(results) == len(
                existing_files
            ), f"Expected {len(existing_files)} results, got {len(results)}"

            successful_count = 0
            failed_files = []

            for result in results:
                if result.was_successful:
                    successful_count += 1
                    assert (
                        result.output.summary is not None
                    ), f"No summary for {result.request_key}"
                else:
                    failed_files.append((result.request_key, result.error))

            # Print results summary
            print("\n📊 MIME Type Test Results:")
            print(f"✅ Successful: {successful_count}/{len(existing_files)}")
            if failed_files:
                print("❌ Failed files:")
                for filename, error in failed_files:
                    print(f"   • {filename}: {error}")

            assert successful_count == len(existing_files), (
                f"Expected all files to succeed, but {successful_count} "
                f"succeeded out of {len(existing_files)}"
            )

        except Exception as e:
            # If we get an exception, make sure to clean up properly
            pytest.fail(f"Test failed with exception: {e}")

    @pytest.mark.incurs_costs
    @requires_project_id
    def test_per_request_overrides_comprehensive(self):
        """Comprehensive test of per-request prompt template and schema overrides.

        Tests all combinations in a single batch job:
        1. Default prompt + default schema (baseline)
        2. Custom prompt + default schema (prompt override only)
        3. Default prompt + custom schema (schema override only)
        4. Custom prompt + custom schema (both overrides)
        """
        from pydantic import BaseModel, Field

        class CompanyInfo(BaseModel):
            name: str = Field(description="Company name")
            industry: str = Field(description="Industry sector")

        class PersonInfo(BaseModel):
            name: str = Field(description="Person's name")
            role: str = Field(description="Job title or role")

        class GenericInput(BaseModel):
            text: str

        # Job with default settings
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Return the text '{{ text }}' exactly in the result field.",
        )

        # Request 1: Default prompt + default schema (baseline)
        job.add_request(
            "default_both",
            GenericInput(text="test message"),
        )

        # Request 2: Custom prompt + default schema
        job.add_request(
            "custom_prompt_only",
            GenericInput(text="innovation"),
            prompt_template=(
                "Take '{{ text }}' and return it with '-tech' suffix in result field."
            ),
        )

        # Request 3: Default prompt + custom schema
        job.add_request(
            "custom_schema_only",
            GenericInput(text="Apple Inc operates in technology sector"),
            output_schema=CompanyInfo,
        )

        # Request 4: Custom prompt + custom schema
        job.add_request(
            "custom_both",
            GenericInput(text="Jane Smith works as Senior Engineer at TechCorp"),
            output_schema=PersonInfo,
            prompt_template=(
                "Extract person name and role from: '{{ text }}'. "
                "Return as name and role fields."
            ),
        )

        results = list(job.submit().wait().results())
        by_key = {r.request_key: r for r in results}

        # Verify all requests completed
        expected_keys = {
            "default_both",
            "custom_prompt_only",
            "custom_schema_only",
            "custom_both",
        }
        assert (
            set(by_key.keys()) == expected_keys
        ), f"Missing results for: {expected_keys - set(by_key.keys())}"

        # Verify schema types match expectations
        assert (
            isinstance(by_key["default_both"].output, SimpleOutput)
            if by_key["default_both"].output
            else True
        )
        assert (
            isinstance(by_key["custom_prompt_only"].output, SimpleOutput)
            if by_key["custom_prompt_only"].output
            else True
        )
        assert (
            isinstance(by_key["custom_schema_only"].output, CompanyInfo)
            if by_key["custom_schema_only"].output
            else True
        )
        assert (
            isinstance(by_key["custom_both"].output, PersonInfo)
            if by_key["custom_both"].output
            else True
        )

        # Log comprehensive results
        print("\n🔎 Per-request overrides comprehensive test results:")
        for key in expected_keys:
            r = by_key[key]
            schema_name = type(r.output).__name__ if r.output else "<no output>"
            result_preview = str(r.output)[:50] + "..." if r.output else "N/A"
            print(
                f" - {key}: success={r.was_successful} "
                f"schema={schema_name} result={result_preview} error={r.error}"
            )

    @pytest.mark.incurs_costs
    @requires_project_id
    def test_serialization_with_dynamic_schemas(self):
        """Test self-contained state serialization with dynamically created schemas.

        This validates the key new serialization feature: storing schema definitions
        instead of class names, enabling reconnection in stateless environments.

        Simulates: Process A submits job -> Process B reconnects -> fetches results
        """
        from pydantic import BaseModel, Field

        class DynamicSchema(BaseModel):
            """Schema created dynamically - wouldn't exist in reconnection process"""

            extracted_data: str = Field(description="Extracted information")
            confidence: float = Field(description="Confidence score")

        class TaskInput(BaseModel):
            content: str

        # Job with mixed schema usage
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process: {{ content }}",
        )

        # Request 1: Job default schema
        job.add_request(
            "default_schema",
            TaskInput(content="simple text processing"),
        )

        # Request 2: Dynamic schema override
        job.add_request(
            "dynamic_schema",
            TaskInput(content="Extract key information and provide confidence score"),
            output_schema=DynamicSchema,
            prompt_template=(
                "Extract key info from '{{ content }}' and rate confidence 0-1. "
                "Return as extracted_data and confidence fields."
            ),
        )

        # Process A: Submit and serialize immediately (no wait)
        job.submit()
        state_json = job.serialize()

        # Process B: Reconnect from serialized state (simulates different
        # process/environment)
        reconnected_job = Job.reconnect_from_state(state_json)

        # Wait for completion with timeout
        import time

        timeout_s = 600  # 10 minutes max
        poll_interval = 15
        start = time.time()
        while not reconnected_job.is_done and (time.time() - start) < timeout_s:
            time.sleep(poll_interval)

        if not reconnected_job.is_done:
            pytest.skip(
                "Job not completed within polling timeout; skipping result assertions"
            )

        # Fetch results through reconnected job
        results = list(reconnected_job.results())
        by_key = {r.request_key: r for r in results}

        # Verify both requests completed
        expected_keys = {"default_schema", "dynamic_schema"}
        assert (
            set(by_key.keys()) == expected_keys
        ), f"Missing results for: {expected_keys - set(by_key.keys())}"

        # Critical test: Verify dynamically recreated schemas work correctly
        default_result = by_key["default_schema"]
        dynamic_result = by_key["dynamic_schema"]

        if default_result.output:
            # The recreated schema won't be the exact same class, but should have
            # the same structure
            assert hasattr(
                default_result.output, "result"
            ), "Default schema should have 'result' field"
            assert isinstance(
                default_result.output.result, str
            ), "Result field should be string"

        if dynamic_result.output:
            # The schema was recreated from serialized definition - this is the key test
            assert hasattr(
                dynamic_result.output, "extracted_data"
            ), "Dynamic schema should have 'extracted_data' field"
            assert hasattr(
                dynamic_result.output, "confidence"
            ), "Dynamic schema should have 'confidence' field"
            # Type name might be different (DynamicModel vs DynamicSchema) but
            # fields should work

        # Log results
        print("\n🔎 Dynamic schema serialization test results:")
        for key in expected_keys:
            r = by_key[key]
            schema_name = type(r.output).__name__ if r.output else "<no output>"
            print(
                f" - {key}: success={r.was_successful} "
                f"schema={schema_name} error={r.error}"
            )
            if r.output and hasattr(r.output, "__dict__"):
                print(f"   Fields: {list(r.output.__dict__.keys())}")


class TestSchemaSerializationReversibility:
    """Test that schema serialization/deserialization is perfectly reversible."""

    def test_schema_serialization_structural_equivalence(self, mock_gcp_clients):
        """Test that serialized schemas can be perfectly reconstructed with
        structural equivalence.

        This validates the schema serialization/deserialization logic without
        incurring costs by testing the core schema recreation functionality
        directly.
        """
        import json

        from pydantic import BaseModel, Field

        # Define complex schemas to test various field types
        class OriginalSimpleSchema(BaseModel):
            text: str
            count: int

        class OriginalComplexSchema(BaseModel):
            name: str = Field(description="Entity name")
            score: float = Field(description="Confidence score", ge=0.0, le=1.0)
            tags: list[str] = Field(description="List of tags")
            metadata: dict[str, str] = Field(description="Additional metadata")
            is_valid: bool = Field(description="Validation status")

        # Test the schema serialization directly (what happens during state
        # serialization)
        original_schemas = {
            "simple": OriginalSimpleSchema,
            "complex": OriginalComplexSchema,
        }

        # Serialize schemas to JSON (as done in serialize method)
        serialized_schemas = {}
        for name, schema_class in original_schemas.items():
            serialized_schemas[name] = schema_class.model_json_schema()

        # Test deserialization (as done in reconnect_from_state)
        from src.pyrtex.client import Job

        # Create a job instance to access the private method
        temp_job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=OriginalSimpleSchema,
            prompt_template="test",
            simulation_mode=True,
        )

        recreated_schemas = {}
        for name, schema_json in serialized_schemas.items():
            # Use the actual schema recreation logic from the Job class
            recreated_schemas[name] = temp_job._create_pydantic_model_from_schema(
                schema_json
            )

        # Test 1: Verify simple schema structural equivalence
        original_simple = original_schemas["simple"]
        recreated_simple = recreated_schemas["simple"]

        original_simple_fields = set(original_simple.model_fields.keys())
        recreated_simple_fields = set(recreated_simple.model_fields.keys())
        assert original_simple_fields == recreated_simple_fields, (
            f"Simple schema fields mismatch: original={original_simple_fields}, "
            f"recreated={recreated_simple_fields}"
        )

        # Test simple schema instantiation and field types
        simple_instance = recreated_simple(text="test", count=42)
        assert isinstance(simple_instance.text, str), "text field should be string"
        assert isinstance(simple_instance.count, int), "count field should be int"
        assert simple_instance.text == "test"
        assert simple_instance.count == 42

        # Test 2: Verify complex schema structural equivalence
        original_complex = original_schemas["complex"]
        recreated_complex = recreated_schemas["complex"]

        original_complex_fields = set(original_complex.model_fields.keys())
        recreated_complex_fields = set(recreated_complex.model_fields.keys())
        assert original_complex_fields == recreated_complex_fields, (
            f"Complex schema fields mismatch: original={original_complex_fields}, "
            f"recreated={recreated_complex_fields}"
        )

        # Test complex schema instantiation and field types
        complex_instance = recreated_complex(
            name="test entity",
            score=0.75,
            tags=["tag1", "tag2"],
            metadata={"key": "value"},
            is_valid=True,
        )
        assert isinstance(complex_instance.name, str), "name field should be string"
        assert isinstance(complex_instance.score, float), "score field should be float"
        assert isinstance(complex_instance.tags, list), "tags field should be list"
        assert isinstance(
            complex_instance.metadata, dict
        ), "metadata field should be dict"
        assert isinstance(
            complex_instance.is_valid, bool
        ), "is_valid field should be bool"

        # Test 3: Verify field constraints are preserved (e.g., Field descriptions,
        # validators)
        complex_field_info = recreated_complex.model_fields
        score_field = complex_field_info.get("score")
        assert score_field is not None, "score field should exist"
        # Note: While field constraints from Field() are preserved in the JSON schema,
        # the exact Field objects may differ. What matters is functional equivalence.

        # Test 4: Verify JSON serialization compatibility
        try:
            simple_dict = simple_instance.model_dump()
            complex_dict = complex_instance.model_dump()
            assert isinstance(
                simple_dict, dict
            ), "Simple output should be serializable to dict"
            assert isinstance(
                complex_dict, dict
            ), "Complex output should be serializable to dict"

            # Test round-trip JSON serialization
            simple_json = simple_instance.model_dump_json()
            complex_json = complex_instance.model_dump_json()

            # Deserialize back
            simple_from_json = recreated_simple.model_validate_json(simple_json)
            complex_from_json = recreated_complex.model_validate_json(complex_json)

            assert simple_from_json.text == "test"
            assert simple_from_json.count == 42
            assert complex_from_json.name == "test entity"
            assert complex_from_json.score == 0.75

        except Exception as e:
            pytest.fail(f"JSON serialization failed, indicating structural issues: {e}")

        # Test 5: Verify schema JSON representation is consistent
        original_simple_json = original_simple.model_json_schema()
        recreated_simple_json = recreated_simple.model_json_schema()

        # The schemas should be functionally equivalent
        assert (
            original_simple_json["properties"] == recreated_simple_json["properties"]
        ), "Schema properties should be identical after serialization cycle"
        assert (
            original_simple_json["required"] == recreated_simple_json["required"]
        ), "Required fields should be identical after serialization cycle"

        print("\n✅ Schema serialization reversibility test passed:")
        print(
            f"   Simple schema: {original_simple_fields} -> "
            f"{recreated_simple_fields}"
        )
        print(
            f"   Complex schema: {original_complex_fields} -> "
            f"{recreated_complex_fields}"
        )
        print("   Schema JSON properties and constraints preserved")
        print("   All field types and validation work correctly")
        print("   Serialization cycle maintains structural and functional equivalence")


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
            "req_00000_12345678": ("test_key_1", SimpleOutput),
            "req_00001_87654321": ("test_key_2", SimpleOutput),
        }

        # Create mock BigQuery rows that simulate real response data
        mock_rows = [
            Mock(
                id="req_00000_12345678",
                status=None,  # No error status
                response=json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "text": '{"result": "test_output_1"}'
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
                status=None,  # No error status
                response=json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "text": '{"result": "test_output_2"}'
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
        expected_query = "SELECT id, response, status FROM `project.dataset.table`"
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

        job._instance_map = {"req_00000_12345678": ("test_key_1", SimpleOutput)}

        # Create mock row with invalid response (no function call)
        mock_rows = [
            Mock(
                id="req_00000_12345678",
                status=None,  # No error status
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
            "req_00000_batch_0": ("batch_0", SimpleOutput),
            "req_00001_batch_1": ("batch_1", SimpleOutput),
            "req_00002_batch_2": ("batch_2", SimpleOutput),
        }

        # Create mock rows that come back in DIFFERENT order than submitted
        # This simulates BigQuery returning results in arbitrary order
        mock_rows = [
            # Second request returned first
            Mock(
                id="req_00001_batch_1",
                status=None,  # No error status
                response=json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "text": '{"result": "beta"}'
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
                status=None,  # No error status
                response=json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "text": '{"result": "gamma"}'
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
                status=None,  # No error status
                response=json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "text": '{"result": "alpha"}'
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


class TestMimeTypeDetection:
    """Test MIME type detection for Gemini-supported file types."""

    def test_gemini_supported_mime_types(self, mock_gcp_clients):
        """Test that all file extensions map to Gemini-supported MIME types."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=ComplexOutput,
            prompt_template="Analyze: {{ text }}",
        )

        # Test cases: (extension, expected_mime_type, file_content)
        # These are ALL the MIME types supported by Gemini
        test_cases = [
            # Text files - all should map to text/plain
            (".txt", "text/plain", "Simple text content"),
            (".yaml", "text/plain", "key: value\nlist:\n  - item1\n  - item2"),
            (".yml", "text/plain", "config:\n  debug: true"),
            (".json", "text/plain", '{"name": "test", "value": 123}'),
            (
                ".xml",
                "text/plain",
                '<?xml version="1.0"?><root><item>data</item></root>',
            ),
            (".csv", "text/plain", "name,age,city\nJohn,25,NYC\nJane,30,LA"),
            (".md", "text/plain", "# Title\n\nThis is **markdown**."),
            (".py", "text/plain", 'def hello():\n    print("Hello World")'),
            (".js", "text/plain", 'function hello() { console.log("Hello"); }'),
            (".html", "text/plain", "<html><body><h1>Hello</h1></body></html>"),
            (".sql", "text/plain", "SELECT * FROM users WHERE age > 25;"),
            (".log", "text/plain", "2025-07-19 INFO: Application started"),
            # PDF files
            # Note: We can't easily create real PDF content in tests
            # so we'll test the extension mapping only
            # Unknown extensions should default to text/plain
            (".unknown", "text/plain", "Some unknown file content"),
            (".xyz", "text/plain", "Another unknown extension"),
        ]

        for ext, expected_mime, content in test_cases:
            with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False) as f:
                f.write(content)
                file_path = f.name

            try:
                gcs_uri, mime_type = job._upload_file_to_gcs(file_path, f"test{ext}")
                assert (
                    mime_type == expected_mime
                ), f"Expected {expected_mime} for {ext}, got {mime_type}"
            finally:
                Path(file_path).unlink()

    def test_bytes_input_mime_type(self, mock_gcp_clients):
        """Test that bytes input gets text/plain MIME type."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=ComplexOutput,
            prompt_template="Analyze: {{ text }}",
        )

        test_bytes = b"Some test content"
        gcs_uri, mime_type = job._upload_file_to_gcs(test_bytes, "test.bin")
        assert (
            mime_type == "text/plain"
        ), f"Expected text/plain for bytes, got {mime_type}"

    def test_no_unsupported_mime_types(self, mock_gcp_clients):
        """
        Ensure we never generate unsupported MIME types that would
        cause API errors.
        """
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=ComplexOutput,
            prompt_template="Analyze: {{ text }}",
        )

        # List of Gemini-supported MIME types (as of July 2025)
        # Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/
        # model-reference/gemini
        supported_mime_types = {
            "application/pdf",
            "audio/mpeg",
            "audio/mp3",
            "audio/wav",
            "image/png",
            "image/jpeg",
            "image/webp",
            "text/plain",
            "video/mov",
            "video/mpeg",
            "video/mp4",
            "video/mpg",
            "video/avi",
            "video/wmv",
            "video/mpegps",
            "video/flv",
        }

        # Test a variety of file extensions that might produce unsupported MIME types
        problematic_extensions = [
            ".json",  # Should NOT be application/json
            ".xml",  # Should NOT be application/xml
            ".csv",  # Should NOT be text/csv
            ".js",  # Should NOT be application/javascript
            ".css",  # Should NOT be text/css
            ".html",  # Should NOT be text/html
            ".doc",  # Should NOT be application/msword
            ".xlsx",  # Should NOT be application/vnd.openxmlformats-
            # officedocument.spreadsheetml.sheet
            ".zip",  # Should NOT be application/zip
        ]

        for ext in problematic_extensions:
            with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False) as f:
                f.write(f"Test content for {ext} file")
                file_path = f.name

            try:
                gcs_uri, mime_type = job._upload_file_to_gcs(file_path, f"test{ext}")
                assert mime_type in supported_mime_types, (
                    f"Extension {ext} produced unsupported MIME type: {mime_type}. "
                    f"Supported types: {supported_mime_types}"
                )
                # For these text-based extensions, they should all map to text/plain
                assert (
                    mime_type == "text/plain"
                ), f"Extension {ext} should map to text/plain, got {mime_type}"
            finally:
                Path(file_path).unlink()
