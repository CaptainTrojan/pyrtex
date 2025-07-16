# tests/integration/test_full_run.py

import pytest
from pydantic import BaseModel
from pyrtex.client import Job
import tempfile
import os

# Import from the actual package structure
from pyrtex.client import Job
from pyrtex.models import BatchResult


# Test schemas
class SimpleInput(BaseModel):
    word: str


class SimpleOutput(BaseModel):
    result: str


class FileInput(BaseModel):
    text: str
    file_content: bytes


class ComplexOutput(BaseModel):
    summary: str
    confidence: float
    keywords: list[str]


class TestDryRun:
    """Test dry run functionality."""
    
    @pytest.mark.e2e
    def test_dry_run_output_text_only(self, capsys):
        """Verify that dry_run produces plausible JSONL output and doesn't submit."""
        job = Job(
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template="Process this word: {{ word }}"
        )
        job.add_request(request_key="test1", data=SimpleInput(word="hello"))
        
        job.submit(dry_run=True)
        
        captured = capsys.readouterr()
        assert "--- DRY RUN OUTPUT ---" in captured.out
        assert "Generated JSONL Payload" in captured.out
        assert '"text": "Process this word: hello"' in captured.out
        assert "extract_info" in captured.out
        assert "Dry run enabled. Job was not submitted." in captured.err
    
    @pytest.mark.e2e
    def test_dry_run_output_with_files(self, capsys):
        """Verify dry run works with file inputs."""
        job = Job(
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template="Process {{ text }} from the uploaded file"
        )
        
        test_content = b"This is test file content"
        job.add_request(
            request_key="file_test",
            data=FileInput(text="analyze this", file_content=test_content)
        )
        
        job.submit(dry_run=True)
        
        captured = capsys.readouterr()
        assert "--- DRY RUN OUTPUT ---" in captured.out
        assert "file_data" in captured.out
        assert "mime_type" in captured.out
        assert "Process analyze this from the uploaded file" in captured.out
    
    @pytest.mark.e2e
    def test_dry_run_multiple_requests(self, capsys):
        """Test dry run with multiple requests."""
        job = Job(
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template="Word: {{ word }}"
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
    
    @pytest.mark.e2e
    def test_simulation_mode_basic(self):
        """Verify that simulation_mode returns dummy data without hitting GCP."""
        job = Job(
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template="Process: {{ word }}",
            simulation_mode=True
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
    
    @pytest.mark.e2e
    def test_simulation_mode_multiple_requests(self):
        """Test simulation mode with multiple requests."""
        job = Job(
            model="gemini-1.5-pro",
            output_schema=ComplexOutput,
            prompt_template="Analyze: {{ text }}",
            simulation_mode=True
        )
        
        inputs = [
            SimpleInput(word="first"),
            SimpleInput(word="second"),
            SimpleInput(word="third")
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
    
    @pytest.mark.e2e
    def test_simulation_mode_with_complex_schema(self):
        """Test simulation mode with complex output schema."""
        class DetailedOutput(BaseModel):
            title: str
            description: str
            tags: list[str]
            score: float
            metadata: dict[str, str]
        
        job = Job(
            model="gemini-1.5-flash",
            output_schema=DetailedOutput,
            prompt_template="Analyze: {{ word }}",
            simulation_mode=True
        )
        
        job.add_request("complex_test", SimpleInput(word="analyze"))
        
        results = list(job.submit().wait().results())
        
        assert len(results) == 1
        result = results[0]
        assert result.was_successful
        assert isinstance(result.output, DetailedOutput)
        # The dummy data should be schema-compliant
        assert hasattr(result.output, 'title')
        assert hasattr(result.output, 'description')
        assert hasattr(result.output, 'tags')
        assert hasattr(result.output, 'score')
        assert hasattr(result.output, 'metadata')
    
    @pytest.mark.e2e
    def test_simulation_mode_chaining(self):
        """Test that simulation mode supports method chaining."""
        job = Job(
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template="Process: {{ word }}",
            simulation_mode=True
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
    """Integration tests that would run against real GCP (marked to skip by default)."""
    
    @pytest.mark.e2e
    @pytest.mark.incurs_costs
    @pytest.mark.skip(reason="Requires real GCP setup and incurs costs")
    def test_full_run_simple_text(self):
        """
        The full end-to-end test. This will submit a real job to Vertex AI.
        It requires the user to be authenticated and to have set up GCP resources.
        """
        # Simple prompt engineering: Tell the model exactly what to do
        prompt = '''The user provided a word: '{{ word }}'. 
        Your task is to call the extract_info function with this exact word in the 'result' field.
        Make sure to use the function calling capability.'''
        
        job = Job(
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template=prompt
        )
        job.add_request(request_key="e2e_test_key", data=SimpleInput(word="pyrtex_works"))
        
        # The magic one-liner
        results = list(job.submit().wait().results())
        
        assert len(results) == 1
        result = results[0]
        
        assert result.was_successful
        assert result.request_key == "e2e_test_key"
        assert result.output.result == "pyrtex_works"
        assert result.error is None
        assert result.usage_metadata["totalTokenCount"] > 0
    
    @pytest.mark.e2e
    @pytest.mark.incurs_costs
    @pytest.mark.skip(reason="Requires real GCP setup and incurs costs")
    def test_full_run_with_file(self):
        """Test full run with file input."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document for analysis.")
            temp_file_path = f.name
        
        try:
            prompt = '''Analyze the uploaded file and the text "{{ text }}".
            Call the extract_info function with a summary in the 'result' field.'''
            
            job = Job(
                model="gemini-1.5-flash",
                output_schema=SimpleOutput,
                prompt_template=prompt
            )
            
            job.add_request(
                request_key="file_test",
                data=FileInput(text="additional context", file_path=temp_file_path)
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
    
    @pytest.mark.e2e
    @pytest.mark.incurs_costs
    @pytest.mark.skip(reason="Requires real GCP setup and incurs costs")
    def test_full_run_batch_processing(self):
        """Test batch processing with multiple requests."""
        prompt = '''Process the word "{{ word }}" and return it in the result field.'''
        
        job = Job(
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template=prompt
        )
        
        test_words = ["alpha", "beta", "gamma", "delta", "epsilon"]
        
        for i, word in enumerate(test_words):
            job.add_request(f"batch_{i}", SimpleInput(word=word))
        
        results = list(job.submit().wait().results())
        
        assert len(results) == len(test_words)
        
        # Check that all results are successful
        successful_results = [r for r in results if r.was_successful]
        assert len(successful_results) == len(test_words)
        
        # Check that we got the expected words back
        result_words = {r.output.result for r in successful_results}
        assert result_words == set(test_words)
    
    @pytest.mark.e2e
    @pytest.mark.incurs_costs
    @pytest.mark.skip(reason="Requires real GCP setup and incurs costs")
    def test_error_handling_invalid_prompt(self):
        """Test error handling with invalid prompts."""
        # This prompt doesn't instruct the model to use function calling
        prompt = '''Just respond with plain text: {{ word }}'''
        
        job = Job(
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template=prompt
        )
        
        job.add_request("error_test", SimpleInput(word="test"))
        
        results = list(job.submit().wait().results())
        
        assert len(results) == 1
        result = results[0]
        
        # This should fail because the model won't use function calling
        assert not result.was_successful
        assert result.error is not None
        assert result.output is None


class TestErrorScenarios:
    """Test error scenarios that don't require real GCP."""
    
    @pytest.mark.e2e
    def test_submit_without_requests(self):
        """Test error when submitting without requests."""
        job = Job(
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        with pytest.raises(RuntimeError, match="Cannot submit a job with no requests"):
            job.submit()
    
    @pytest.mark.e2e
    def test_results_without_submission(self):
        """Test error when getting results without submission."""
        job = Job(
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}"
        )
        
        with pytest.raises(RuntimeError, match="Cannot get results for a job that has not been submitted"):
            list(job.results())
    
    @pytest.mark.e2e
    def test_add_request_after_submission(self):
        """Test error when adding requests after submission."""
        job = Job(
            model="gemini-1.5-flash",
            output_schema=SimpleOutput,
            prompt_template="Test: {{ word }}",
            simulation_mode=True
        )
        
        job.add_request("test1", SimpleInput(word="hello"))
        job.submit()  # This sets _batch_job in simulation mode
        
        with pytest.raises(RuntimeError, match="Cannot add requests after job has been submitted"):
            job.add_request("test2", SimpleInput(word="world"))
