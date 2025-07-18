# tests/unit/test_payload.py

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from pyrtex.client import Job
from pyrtex.config import GenerationConfig
from tests.conftest import FileInput, SimpleInput, SimpleOutput


class TestPayloadGeneration:
    """Test JSONL payload generation."""

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
            prompt_template="Process: {{ word }}",
            generation_config=custom_config,
        )

        job.add_request("key1", SimpleInput(word="test"))
        payload = job._create_jsonl_payload()

        line = json.loads(payload.split("\n")[0])
        gen_config = line["request"]["generation_config"]

        assert gen_config["temperature"] == 0.7
        assert gen_config["max_output_tokens"] == 1024
        assert gen_config["top_p"] == 0.9

    def test_create_jsonl_payload_with_file_path(self, mock_gcp_clients):
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

            file_input = FileInput(text="some text", file_path=temp_file_path)
            job.add_request("key1", file_input)

            # Mock the file upload
            job._upload_file_to_gcs = Mock(
                return_value=("gs://test-bucket/test-file.txt", "text/plain")
            )

            payload = job._create_jsonl_payload()

            line = json.loads(payload.split("\n")[0])
            parts = line["request"]["contents"][0]["parts"]

            # Should have both file_data and text parts
            assert len(parts) == 2

            # Find the file_data part
            file_part = next(p for p in parts if "file_data" in p)
            assert (
                file_part["file_data"]["file_uri"] == "gs://test-bucket/test-file.txt"
            )
            assert file_part["file_data"]["mime_type"] == "text/plain"

            # Find the text part
            text_part = next(p for p in parts if "text" in p)
            assert text_part["text"] == "Process some text from file"

            # Verify upload was called
            job._upload_file_to_gcs.assert_called_once()

        finally:
            # Clean up
            os.unlink(temp_file_path)

    def test_create_jsonl_payload_with_bytes_input(self, mock_gcp_clients):
        """Test payload generation with bytes input."""
        from pydantic import BaseModel

        class BytesInput(BaseModel):
            text: str
            file_bytes: bytes

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process {{ text }}",
        )

        bytes_input = BytesInput(text="process this", file_bytes=b"binary content")
        job.add_request("key1", bytes_input)

        # Mock the file upload
        job._upload_file_to_gcs = Mock(
            return_value=("gs://test-bucket/file_bytes", "application/octet-stream")
        )

        payload = job._create_jsonl_payload()

        line = json.loads(payload.split("\n")[0])
        parts = line["request"]["contents"][0]["parts"]

        # Should have both file_data and text parts
        assert len(parts) == 2

        # Find the file_data part
        file_part = next(p for p in parts if "file_data" in p)
        assert file_part["file_data"]["file_uri"] == "gs://test-bucket/file_bytes"
        assert file_part["file_data"]["mime_type"] == "application/octet-stream"

        # Verify upload was called with bytes
        job._upload_file_to_gcs.assert_called_once()
        call_args = job._upload_file_to_gcs.call_args[0]
        assert call_args[0] == b"binary content"

    def test_create_jsonl_payload_path_object(self, mock_gcp_clients):
        """Test payload generation with Path object input."""
        from pydantic import BaseModel

        class PathInput(BaseModel):
            text: str
            file_path: Path

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            temp_file_path = Path(f.name)

        try:
            job = Job(
                model="gemini-2.0-flash-lite-001",
                output_schema=SimpleOutput,
                prompt_template="Process {{ text }}",
            )

            path_input = PathInput(text="process this", file_path=temp_file_path)
            job.add_request("key1", path_input)

            # Mock the file upload
            job._upload_file_to_gcs = Mock(
                return_value=("gs://test-bucket/test-file.txt", "text/plain")
            )

            payload = job._create_jsonl_payload()

            line = json.loads(payload.split("\n")[0])
            parts = line["request"]["contents"][0]["parts"]

            # Should have both file_data and text parts
            assert len(parts) == 2

            # Find the file_data part
            file_part = next(p for p in parts if "file_data" in p)
            assert (
                file_part["file_data"]["file_uri"] == "gs://test-bucket/test-file.txt"
            )

            # Verify upload was called
            job._upload_file_to_gcs.assert_called_once()

        finally:
            # Clean up
            os.unlink(temp_file_path)

    def test_create_jsonl_payload_instance_mapping(self, mock_gcp_clients):
        """Test that instance IDs are properly mapped to request keys."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process: {{ word }}",
        )

        job.add_request("first_key", SimpleInput(word="hello"))
        job.add_request("second_key", SimpleInput(word="world"))

        payload = job._create_jsonl_payload()

        lines = payload.split("\n")
        assert len(lines) == 2

        # Parse both lines to get instance IDs
        first_line = json.loads(lines[0])
        second_line = json.loads(lines[1])

        first_id = first_line["id"]
        second_id = second_line["id"]

        # Check that instance mapping is correct
        assert job._instance_map[first_id] == "first_key"
        assert job._instance_map[second_id] == "second_key"

        # IDs should be unique
        assert first_id != second_id
        assert first_id.startswith("req_00000_")
        assert second_id.startswith("req_00001_")

    def test_create_jsonl_payload_template_rendering(self, mock_gcp_clients):
        """Test that Jinja2 template rendering works correctly."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Hello {{ word }}! Your task is to process '{{ word }}' and return it.",
        )

        job.add_request("key1", SimpleInput(word="world"))

        payload = job._create_jsonl_payload()

        line = json.loads(payload.split("\n")[0])
        text_part = line["request"]["contents"][0]["parts"][0]

        expected_text = "Hello world! Your task is to process 'world' and return it."
        assert text_part["text"] == expected_text

    def test_create_jsonl_payload_schema_serialization(self, mock_gcp_clients):
        """Test that output schema is properly serialized in the payload."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process: {{ word }}",
        )

        job.add_request("key1", SimpleInput(word="test"))

        payload = job._create_jsonl_payload()

        line = json.loads(payload.split("\n")[0])
        function_decl = line["request"]["tools"][0]["function_declarations"][0]

        assert function_decl["name"] == "extract_info"
        assert (
            function_decl["description"]
            == "Extracts structured information based on the schema."
        )

        # Check that schema is properly included
        parameters = function_decl["parameters"]
        assert "properties" in parameters
        assert "result" in parameters["properties"]
        assert parameters["properties"]["result"]["type"] == "string"


class TestFileUpload:
    """Test file upload functionality."""

    def test_upload_file_to_gcs_with_string_path(self, mock_gcp_clients):
        """Test file upload with string path."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process: {{ word }}",
        )

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            temp_file_path = f.name

        try:
            gcs_uri, mime_type = job._upload_file_to_gcs(
                temp_file_path, "test/path.txt"
            )

            assert gcs_uri == "gs://pyrtex-assets-test-project/test/path.txt"
            assert mime_type == "text/plain"

            # Verify mock calls
            mock_gcp_clients["storage"].bucket.assert_called_with(
                "pyrtex-assets-test-project"
            )

        finally:
            # Clean up
            os.unlink(temp_file_path)

    def test_upload_file_to_gcs_with_bytes(self, mock_gcp_clients):
        """Test file upload with bytes data."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process: {{ word }}",
        )

        test_bytes = b"binary test content"
        gcs_uri, mime_type = job._upload_file_to_gcs(test_bytes, "test/binary.bin")

        assert gcs_uri == "gs://pyrtex-assets-test-project/test/binary.bin"
        assert mime_type == "application/octet-stream"

        # Verify mock calls
        mock_gcp_clients["storage"].bucket.assert_called_with(
            "pyrtex-assets-test-project"
        )

    def test_upload_file_to_gcs_with_path_object(self, mock_gcp_clients):
        """Test file upload with Path object."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process: {{ word }}",
        )

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"test": "data"}')
            temp_file_path = Path(f.name)

        try:
            gcs_uri, mime_type = job._upload_file_to_gcs(
                temp_file_path, "test/data.json"
            )

            assert gcs_uri == "gs://pyrtex-assets-test-project/test/data.json"
            assert mime_type == "application/json"

            # Verify mock calls
            mock_gcp_clients["storage"].bucket.assert_called_with(
                "pyrtex-assets-test-project"
            )

        finally:
            # Clean up
            os.unlink(temp_file_path)
