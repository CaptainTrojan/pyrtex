# tests/unit/test_job_serialization.py

import json
from unittest.mock import Mock, patch

import pytest
from google.cloud.aiplatform_v1.types import JobState
from pydantic import BaseModel, Field

from pyrtex.client import Job
from pyrtex.config import InfrastructureConfig
from tests.conftest import SimpleInput, SimpleOutput


class AltOutput(BaseModel):
    value: str = Field(description="Alt value")


def test_serialize_before_submission_raises(mock_gcp_clients):
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=SimpleOutput,
        prompt_template="Test: {{ word }}",
    )
    with pytest.raises(
        RuntimeError, match="Cannot serialize a job that has not been submitted"
    ):
        job.serialize()


def test_serialize_happy_path(mock_gcp_clients):
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=SimpleOutput,
        prompt_template="Test: {{ word }}",
    )
    job.add_request("r1", SimpleInput(word="alpha"))
    job.add_request("r2", SimpleInput(word="beta"), output_schema=AltOutput)
    job.submit()

    state_json = job.serialize()
    state = json.loads(state_json)

    assert (
        state["batch_job_resource_name"] == mock_gcp_clients["batch_job"].resource_name
    )
    assert state["session_id"] == job._session_id
    assert "infrastructure_config" in state
    assert isinstance(state["instance_map"], dict)
    # Validate each mapping is a two-element sequence [request_key, fq_schema]
    for item in state["instance_map"].values():
        assert isinstance(item, (list, tuple)) and len(item) == 2
        assert (
            item[1].endswith((".AltOutput", ".SimpleOutput"))
            or item[1].endswith("AltOutput")
            or item[1].endswith("SimpleOutput")
        )
    # Ensure both schemas present
    schema_names = {v[1].split(".")[-1] for v in state["instance_map"].values()}
    assert {"SimpleOutput", "AltOutput"}.issubset(schema_names)


def test_reconnect_from_state_restores_job(mock_gcp_clients):
    # Create original job with custom infra config values to ensure round-trip
    config = InfrastructureConfig(
        project_id="test-project",
        location="us-central1",
        gcs_bucket_name="custom-bucket",
        bq_dataset_id="custom_dataset",
    )
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=SimpleOutput,
        prompt_template="Test: {{ word }}",
        config=config,
    )
    job.add_request("r1", SimpleInput(word="hello"))
    job.submit()
    state_json = job.serialize()

    # Patch BatchPredictionJob so reconnect uses a predictable mock
    with patch("google.cloud.aiplatform.BatchPredictionJob") as mock_bp:
        mock_batch = Mock()
        mock_batch.state = JobState.JOB_STATE_SUCCEEDED
        mock_batch.resource_name = job._batch_job.resource_name
        mock_bp.return_value = mock_batch

        # Also patch initialization to avoid re-running auth (already covered elsewhere)
        with patch.object(Job, "_initialize_gcp"):
            re_job = Job.reconnect_from_state(state_json)

    # Verify core properties
    assert re_job._session_id == job._session_id
    assert re_job.config.project_id == job.config.project_id
    assert re_job.config.gcs_bucket_name == job.config.gcs_bucket_name
    assert re_job.config.bq_dataset_id == job.config.bq_dataset_id
    assert isinstance(re_job._instance_map, dict)
    assert len(re_job._instance_map) == len(job._instance_map)
    # Cannot add new requests after reconnection
    with pytest.raises(
        RuntimeError, match="Cannot add requests after job has been submitted"
    ):
        re_job.add_request("new", SimpleInput(word="x"))


def test_reconnect_from_state_import_error(mocker):
    # Build minimal fake state referencing a non-existent schema
    fake_state = {
        "batch_job_resource_name": "projects/x/locations/y/batchPredictionJobs/123",
        "session_id": "abc123",
        "infrastructure_config": InfrastructureConfig(
            project_id="p", location="us-central1"
        ).model_dump(mode="json"),
        "instance_map": {"req_00000_deadbeef": ("key", "nonexistent.module.Schema")},
    }
    state_json = json.dumps(fake_state)

    # Patch init & BatchPredictionJob to avoid real calls
    mocker.patch.object(Job, "_initialize_gcp")
    mocker.patch("google.cloud.aiplatform.BatchPredictionJob", return_value=Mock())

    with pytest.raises(
        RuntimeError, match="Failed to import schema"
    ):  # import error propagated
        Job.reconnect_from_state(state_json)


def test_status_property(mock_gcp_clients):
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=SimpleOutput,
        prompt_template="Test: {{ word }}",
    )
    # No batch job yet
    assert job.status is None

    # After submission should refresh and return state
    job.add_request("r", SimpleInput(word="hi"))
    job.submit()
    s = job.status
    assert s == mock_gcp_clients["batch_job"].state


def test_is_done_states(mock_gcp_clients):
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=SimpleOutput,
        prompt_template="Test: {{ word }}",
    )
    # Not submitted
    assert job.is_done is False

    job.add_request("r", SimpleInput(word="x"))
    job.submit()

    # Simulate various terminal and non-terminal states
    terminal = [
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_FAILED,
        JobState.JOB_STATE_CANCELLED,
        JobState.JOB_STATE_EXPIRED,
    ]
    non_terminal = [JobState.JOB_STATE_PENDING, JobState.JOB_STATE_RUNNING]

    for st in terminal:
        mock_gcp_clients["batch_job"].state = st
        assert job.is_done is True

    for st in non_terminal:
        mock_gcp_clients["batch_job"].state = st
        assert job.is_done is False
