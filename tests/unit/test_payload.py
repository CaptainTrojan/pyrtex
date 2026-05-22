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

        # Check response schema configuration (new JSON mode)
        assert gen_config["response_mime_type"] == "application/json"
        assert "response_schema" in gen_config
        response_schema = gen_config["response_schema"]
        assert response_schema["type"] == "object"
        assert "properties" in response_schema

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

    def test_create_jsonl_payload_with_attachment(self, mock_gcp_clients):
        """Attachments passed via attachments= become file_data parts."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            temp_file_path = Path(f.name)

        try:
            job = Job(
                model="gemini-2.0-flash-lite-001",
                output_schema=SimpleOutput,
                prompt_template="Process {{ word }}",
            )

            job._stage_attachment = Mock(
                return_value=("gs://test-bucket/test-file.txt", "text/plain")
            )

            job.add_request(
                "key1", SimpleInput(word="alpha"), attachments=[temp_file_path]
            )
            payload = job._create_jsonl_payload()

            line = json.loads(payload.split("\n")[0])
            parts = line["request"]["contents"][0]["parts"]

            assert len(parts) == 2
            # Files first, prompt last — the order Gemini expects.
            assert parts[0]["file_data"]["file_uri"] == "gs://test-bucket/test-file.txt"
            assert parts[0]["file_data"]["mime_type"] == "text/plain"
            assert parts[1]["text"] == "Process alpha"

            job._stage_attachment.assert_called_once()

        finally:
            os.unlink(temp_file_path)

    def test_add_request_rejects_unsupported_extension(self, mock_gcp_clients):
        """Unsupported file extensions are rejected at add_request time."""
        from pyrtex.exceptions import ValidationError

        with tempfile.NamedTemporaryFile(mode="w", suffix=".docx", delete=False) as f:
            f.write("ignored")
            temp = Path(f.name)

        try:
            job = Job(
                model="gemini-2.0-flash-lite-001",
                output_schema=SimpleOutput,
                prompt_template="Process the file.",
            )
            with pytest.raises(ValidationError, match="unsupported extension"):
                job.add_request("k", attachments=[temp])
        finally:
            os.unlink(temp)

    def test_multiple_attachments_become_ordered_file_parts(
        self, mock_gcp_clients, tmp_path
    ):
        """A single request can carry many attachments; order is preserved
        and the prompt text always comes last."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"pdf")
        png = tmp_path / "shot.png"
        png.write_bytes(b"png")
        yaml = tmp_path / "data.yaml"
        yaml.write_text("k: v")

        staged_uris = iter(
            [
                ("gs://staged/0_doc.pdf", "application/pdf"),
                ("gs://staged/1_shot.png", "image/png"),
                ("gs://staged/2_data.yaml", "text/plain"),
            ]
        )

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Compare the attached files.",
        )
        gcs_paths_seen = []

        def fake_stage(source, gcs_path):
            gcs_paths_seen.append(gcs_path)
            return next(staged_uris)

        job._stage_attachment = Mock(side_effect=fake_stage)

        job.add_request("multi", attachments=[pdf, png, yaml])
        payload = job._create_jsonl_payload()
        parts = json.loads(payload.split("\n")[0])["request"]["contents"][0]["parts"]

        # Three files then one text part, in the order they were passed.
        assert len(parts) == 4
        assert parts[0]["file_data"]["file_uri"] == "gs://staged/0_doc.pdf"
        assert parts[1]["file_data"]["file_uri"] == "gs://staged/1_shot.png"
        assert parts[2]["file_data"]["file_uri"] == "gs://staged/2_data.yaml"
        assert parts[3]["text"] == "Compare the attached files."

        # Each attachment got a unique, index-prefixed GCS path under the
        # request's session folder.
        assert len(set(gcs_paths_seen)) == 3
        assert gcs_paths_seen[0].endswith("/000_doc.pdf")
        assert gcs_paths_seen[1].endswith("/001_shot.png")
        assert gcs_paths_seen[2].endswith("/002_data.yaml")

    def test_add_request_accepts_attachments_only(self, mock_gcp_clients):
        """data is optional; pure-attachment requests are first-class."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
            f.write("x")
            temp = Path(f.name)
        try:
            job = Job(
                model="gemini-2.0-flash-lite-001",
                output_schema=SimpleOutput,
                prompt_template="Summarize the attached PDF.",
            )
            job._stage_attachment = Mock(
                return_value=("gs://test-bucket/x.pdf", "application/pdf")
            )
            job.add_request("only-files", attachments=[temp])
            payload = job._create_jsonl_payload()
            line = json.loads(payload.split("\n")[0])
            parts = line["request"]["contents"][0]["parts"]
            assert len(parts) == 2
            assert "file_data" in parts[0]
            assert parts[1]["text"] == "Summarize the attached PDF."
        finally:
            os.unlink(temp)

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
        assert job._instance_map[first_id][0] == "first_key"
        assert job._instance_map[second_id][0] == "second_key"

        # IDs should be unique
        assert first_id != second_id
        assert first_id.startswith("req_00000_")
        assert second_id.startswith("req_00001_")

    def test_create_jsonl_payload_template_rendering(self, mock_gcp_clients):
        """Test that Jinja2 template rendering works correctly."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template=(
                "Hello {{ word }}! Your task is to process '{{ word }}' "
                "and return it."
            ),
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
        gen_config = line["request"]["generation_config"]

        # Check response schema configuration (new JSON mode)
        assert "response_schema" in gen_config
        assert gen_config["response_mime_type"] == "application/json"

        response_schema = gen_config["response_schema"]
        assert "properties" in response_schema
        assert "result" in response_schema["properties"]
        assert response_schema["properties"]["result"]["type"] == "string"


class TestStageAttachment:
    """Test attachment staging dispatch (local Path, gs://, s3://)."""

    def test_stage_attachment_with_string_path(self, mock_gcp_clients):
        """Local string paths are uploaded via upload_from_filename."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process: {{ word }}",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            temp_file_path = f.name

        try:
            gcs_uri, mime_type = job._stage_attachment(temp_file_path, "test/path.txt")
            assert gcs_uri == "gs://pyrtex-assets-test-project/test/path.txt"
            assert mime_type == "text/plain"
            mock_gcp_clients["storage"].bucket.assert_called_with(
                "pyrtex-assets-test-project"
            )
        finally:
            os.unlink(temp_file_path)

    def test_stage_attachment_with_path_object(self, mock_gcp_clients):
        """Path objects are uploaded with mime sniffed from extension."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process: {{ word }}",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"test": "data"}')
            temp_file_path = Path(f.name)

        try:
            gcs_uri, mime_type = job._stage_attachment(temp_file_path, "test/data.json")
            assert gcs_uri == "gs://pyrtex-assets-test-project/test/data.json"
            # Vertex batch only accepts text/plain for textual formats.
            assert mime_type == "text/plain"
        finally:
            os.unlink(temp_file_path)

    def test_stage_attachment_gs_uri_passthrough(self, mock_gcp_clients):
        """gs:// URIs are returned as-is; no re-upload."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process: {{ word }}",
        )
        gcs_uri, mime_type = job._stage_attachment(
            "gs://some-other-bucket/folder/doc.pdf", "ignored.pdf"
        )
        assert gcs_uri == "gs://some-other-bucket/folder/doc.pdf"
        assert mime_type == "application/pdf"
        # No upload calls made for pass-through.
        mock_gcp_clients["storage"].bucket.return_value.blob.assert_not_called()

    def test_stage_attachment_s3_without_boto3_raises(self, mock_gcp_clients, mocker):
        """When boto3 isn't available, s3 sources raise a clear error."""
        import sys

        from pyrtex.exceptions import ConfigurationError

        # Force boto3 import to fail.
        mocker.patch.dict(sys.modules, {"boto3": None})

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process: {{ word }}",
        )
        with pytest.raises(ConfigurationError, match="pyrtex\\[s3\\]"):
            job._stage_attachment("s3://bucket/key.pdf", "test/k.pdf")

    def test_stage_attachment_s3_get_object_failure_wraps_error(
        self, mock_gcp_clients, mocker
    ):
        """A failure on S3 get_object surfaces as a ValidationError."""
        import sys
        import types

        from pyrtex.exceptions import ValidationError

        fake_boto3 = types.ModuleType("boto3")
        fake_client = Mock()
        fake_client.get_object.side_effect = RuntimeError("access denied")
        fake_boto3.client = Mock(return_value=fake_client)
        mocker.patch.dict(sys.modules, {"boto3": fake_boto3})

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process: {{ word }}",
        )
        with pytest.raises(ValidationError, match="Failed to fetch S3 object"):
            job._stage_attachment("s3://my-bucket/key.pdf", "staged/k.pdf")

    def test_stage_attachment_s3_streams_to_gcs(self, mock_gcp_clients, mocker):
        """S3 source: fetch from S3 and stream the body into the GCS blob."""
        import sys
        import types

        # Build a fake boto3 module exposing a client() returning a mock.
        fake_boto3 = types.ModuleType("boto3")
        fake_client = Mock()
        fake_body = Mock(name="StreamingBody")
        fake_client.get_object.return_value = {"Body": fake_body, "ContentLength": 42}
        fake_boto3.client = Mock(return_value=fake_client)
        mocker.patch.dict(sys.modules, {"boto3": fake_boto3})

        mock_blob = Mock()
        mock_gcp_clients["storage"].bucket.return_value.blob.return_value = mock_blob

        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process: {{ word }}",
        )
        gcs_uri, mime_type = job._stage_attachment(
            "s3://my-bucket/path/to/doc.pdf", "staged/doc.pdf"
        )

        assert gcs_uri == "gs://pyrtex-assets-test-project/staged/doc.pdf"
        assert mime_type == "application/pdf"

        fake_boto3.client.assert_called_once_with("s3")
        fake_client.get_object.assert_called_once_with(
            Bucket="my-bucket", Key="path/to/doc.pdf"
        )
        mock_blob.upload_from_file.assert_called_once_with(
            fake_body, content_type="application/pdf"
        )
