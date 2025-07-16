# tests/unit/test_payload.py

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import tempfile
import os

from pydantic import BaseModel

from pyrtex.client import Job
from pyrtex.config import GenerationConfig
from tests.conftest import SimpleInput, SimpleOutput, FileInput


class TestCreateJsonlPayload:
    """Tests for _create_jsonl_payload method."""
    
    def test_create_jsonl_payload_text_only(self, mock_gcp_clients):
        """Test JSONL payload creation with text-only fields."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process this word: {{ word }}"
        )
        
        job.add_request("key1", SimpleInput(word="hello"))
        job.add_request("key2", SimpleInput(word="world"))
        
        payload = job._create_jsonl_payload()
        
        lines = payload.split('\n')
        assert len(lines) == 2
        
        # Parse first line
        first_request = json.loads(lines[0])
        assert first_request["id"].startswith("req_00000_")
        assert len(first_request["request"]["contents"]) == 1
        assert first_request["request"]["contents"][0]["role"] == "user"
        
        # Should have one text part
        parts = first_request["request"]["contents"][0]["parts"]
        assert len(parts) == 1
        assert parts[0]["text"] == "Process this word: hello"
        
        # Should have generation config
        assert "generation_config" in first_request["request"]
        assert first_request["request"]["generation_config"]["temperature"] == 0.0
        
        # Should have tools for function calling
        assert "tools" in first_request["request"]
        assert len(first_request["request"]["tools"]) == 1
        assert first_request["request"]["tools"][0]["function_declarations"][0]["name"] == "extract_info"
        
        # Parse second line
        second_request = json.loads(lines[1])
        assert second_request["id"].startswith("req_00001_")
        parts = second_request["request"]["contents"][0]["parts"]
        assert parts[0]["text"] == "Process this word: world"
    
    def test_create_jsonl_payload_with_file_path(self, mock_gcp_clients):
        """Test JSONL payload creation with file path."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process this text: {{ text }}"
        )
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test content")
            temp_file_path = f.name
        
        try:
            job.add_request("key1", FileInput(text="analyze", file_path=temp_file_path))
            
            # Mock the file upload
            with patch.object(job, '_upload_file_to_gcs', return_value=("gs://bucket/file.txt", "text/plain")):
                payload = job._create_jsonl_payload()
            
            lines = payload.split('\n')
            assert len(lines) == 1
            
            request = json.loads(lines[0])
            parts = request["request"]["contents"][0]["parts"]
            
            # Should have file_data part and text part
            assert len(parts) == 2
            
            # Check file_data part
            file_part = next((p for p in parts if "file_data" in p), None)
            assert file_part is not None
            assert file_part["file_data"]["file_uri"] == "gs://bucket/file.txt"
            assert file_part["file_data"]["mime_type"] == "text/plain"
            
            # Check text part
            text_part = next((p for p in parts if "text" in p), None)
            assert text_part is not None
            assert text_part["text"] == "Process this text: analyze"
            
        finally:
            # Clean up
            os.unlink(temp_file_path)
    
    def test_create_jsonl_payload_with_bytes(self, mock_gcp_clients):
        """Test JSONL payload creation with bytes data."""
        class BytesInput(BaseModel):
            description: str
            image_data: bytes
        
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Analyze this image: {{ description }}"
        )
        
        image_bytes = b"fake image data"
        job.add_request("key1", BytesInput(description="test image", image_data=image_bytes))
        
        # Mock the file upload
        with patch.object(job, '_upload_file_to_gcs', return_value=("gs://bucket/image_data", "application/octet-stream")):
            payload = job._create_jsonl_payload()
        
        lines = payload.split('\n')
        assert len(lines) == 1
        
        request = json.loads(lines[0])
        parts = request["request"]["contents"][0]["parts"]
        
        # Should have file_data part and text part
        assert len(parts) == 2
        
        # Check file_data part
        file_part = next((p for p in parts if "file_data" in p), None)
        assert file_part is not None
        assert file_part["file_data"]["file_uri"] == "gs://bucket/image_data"
        assert file_part["file_data"]["mime_type"] == "application/octet-stream"
        
        # Check text part
        text_part = next((p for p in parts if "text" in p), None)
        assert text_part is not None
        assert text_part["text"] == "Analyze this image: test image"
    
    def test_create_jsonl_payload_with_path_object(self, mock_gcp_clients):
        """Test JSONL payload creation with Path object."""
        class PathInput(BaseModel):
            text: str
            file_path: Path
        
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process: {{ text }}"
        )
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test content")
            temp_file_path = Path(f.name)
        
        try:
            job.add_request("key1", PathInput(text="analyze", file_path=temp_file_path))
            
            # Mock the file upload
            with patch.object(job, '_upload_file_to_gcs', return_value=("gs://bucket/file.txt", "text/plain")):
                payload = job._create_jsonl_payload()
            
            lines = payload.split('\n')
            assert len(lines) == 1
            
            request = json.loads(lines[0])
            parts = request["request"]["contents"][0]["parts"]
            
            # Should have file_data part and text part
            assert len(parts) == 2
            
            # Check file_data part exists
            file_part = next((p for p in parts if "file_data" in p), None)
            assert file_part is not None
            
        finally:
            # Clean up
            temp_file_path.unlink()
    
    def test_create_jsonl_payload_instance_mapping(self, mock_gcp_clients):
        """Test that instance mapping is correctly maintained."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process: {{ word }}"
        )
        
        job.add_request("user_key_1", SimpleInput(word="hello"))
        job.add_request("user_key_2", SimpleInput(word="world"))
        
        payload = job._create_jsonl_payload()
        
        lines = payload.split('\n')
        assert len(lines) == 2
        
        first_request = json.loads(lines[0])
        second_request = json.loads(lines[1])
        
        # Check that instance mapping is correct
        first_id = first_request["id"]
        second_id = second_request["id"]
        
        assert job._instance_map[first_id] == "user_key_1"
        assert job._instance_map[second_id] == "user_key_2"
        
        assert len(job._instance_map) == 2
    
    def test_create_jsonl_payload_custom_generation_config(self, mock_gcp_clients):
        """Test JSONL payload with custom generation configuration."""
        custom_config = GenerationConfig(
            temperature=0.7,
            max_output_tokens=1000,
            top_p=0.8,
            top_k=40
        )
        
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Process: {{ word }}",
            generation_config=custom_config
        )
        
        job.add_request("key1", SimpleInput(word="hello"))
        
        payload = job._create_jsonl_payload()
        
        lines = payload.split('\n')
        request = json.loads(lines[0])
        
        gen_config = request["request"]["generation_config"]
        assert gen_config["temperature"] == 0.7
        assert gen_config["max_output_tokens"] == 1000
        assert gen_config["top_p"] == 0.8
        assert gen_config["top_k"] == 40
    
    def test_create_jsonl_payload_schema_integration(self, mock_gcp_clients):
        """Test that output schema is correctly integrated into tools."""
        class CustomOutput(BaseModel):
            name: str
            age: int
            active: bool
        
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=CustomOutput,
            prompt_template="Extract: {{ text }}"
        )
        
        job.add_request("key1", SimpleInput(word="test"))
        
        payload = job._create_jsonl_payload()
        
        lines = payload.split('\n')
        request = json.loads(lines[0])
        
        # Check function declaration
        func_decl = request["request"]["tools"][0]["function_declarations"][0]
        assert func_decl["name"] == "extract_info"
        assert func_decl["description"] == "Extracts structured information based on the schema."
        
        # Check that schema is present
        parameters = func_decl["parameters"]
        assert "properties" in parameters
        assert "name" in parameters["properties"]
        assert "age" in parameters["properties"]
        assert "active" in parameters["properties"]
        
        # Check tool config
        tool_config = request["request"]["tool_config"]
        assert tool_config["function_calling_config"]["mode"] == "any"


class TestUploadFileToGcs:
    """Tests for _upload_file_to_gcs method."""
    
    def test_upload_string_file(self, mock_gcp_clients):
        """Test uploading a string file path."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test content")
            temp_file_path = f.name
        
        try:
            mock_blob = Mock()
            mock_gcp_clients['storage'].bucket.return_value.blob.return_value = mock_blob
            
            gcs_uri, mime_type = job._upload_file_to_gcs(temp_file_path, "test/file.txt")
            
            assert gcs_uri == "gs://pyrtex-assets-test-project/test/file.txt"
            assert mime_type == "text/plain"
            
            # Verify blob operations
            mock_blob.upload_from_filename.assert_called_once_with(temp_file_path, content_type="text/plain")
            
        finally:
            os.unlink(temp_file_path)
    
    def test_upload_bytes_data(self, mock_gcp_clients):
        """Test uploading bytes data."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        mock_blob = Mock()
        mock_gcp_clients['storage'].bucket.return_value.blob.return_value = mock_blob
        
        test_bytes = b"test binary data"
        gcs_uri, mime_type = job._upload_file_to_gcs(test_bytes, "test/data.bin")
        
        assert gcs_uri == "gs://pyrtex-assets-test-project/test/data.bin"
        assert mime_type == "application/octet-stream"
        
        # Verify blob operations
        mock_blob.upload_from_string.assert_called_once_with(test_bytes, content_type="application/octet-stream")
    
    def test_upload_path_object(self, mock_gcp_clients):
        """Test uploading a Path object."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write('{"test": "data"}')
            temp_file_path = Path(f.name)
        
        try:
            mock_blob = Mock()
            mock_gcp_clients['storage'].bucket.return_value.blob.return_value = mock_blob
            
            gcs_uri, mime_type = job._upload_file_to_gcs(temp_file_path, "test/file.json")
            
            assert gcs_uri == "gs://pyrtex-assets-test-project/test/file.json"
            assert mime_type == "application/json"
            
            # Verify blob operations
            mock_blob.upload_from_filename.assert_called_once_with(str(temp_file_path), content_type="application/json")
            
        finally:
            temp_file_path.unlink()
    
    def test_upload_unknown_mimetype(self, mock_gcp_clients):
        """Test uploading file with unknown mime type."""
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        # Create a temporary file with unknown extension
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.unknown') as f:
            f.write("test content")
            temp_file_path = f.name
        
        try:
            mock_blob = Mock()
            mock_gcp_clients['storage'].bucket.return_value.blob.return_value = mock_blob
            
            gcs_uri, mime_type = job._upload_file_to_gcs(temp_file_path, "test/file.unknown")
            
            assert gcs_uri == "gs://pyrtex-assets-test-project/test/file.unknown"
            assert mime_type == "application/octet-stream"  # Default fallback
            
        finally:
            os.unlink(temp_file_path)