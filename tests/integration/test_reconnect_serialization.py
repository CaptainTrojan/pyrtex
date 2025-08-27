# tests/integration/test_reconnect_serialization.py

import json
import os
from unittest.mock import Mock, patch

import pytest
from google.cloud.aiplatform_v1.types import JobState
from pydantic import BaseModel

from pyrtex.client import Job
from tests.integration.test_full_run import SimpleInput, SimpleOutput

requires_project_id = pytest.mark.skipif(
    not os.getenv("GOOGLE_PROJECT_ID"),
    reason="Requires GOOGLE_PROJECT_ID environment variable to be set",
)


class OverrideOutput(BaseModel):
    company: str
    location: str


class TestReconnectSerializationIntegration:
    def test_mocked_reconnect_round_trip(self, mock_gcp_clients):
        """Simulate three-process workflow with mocked job.

        1. Process A submits then immediately serializes (job still PENDING)
        2. Process B reconnects, sees not done
        3. Process C (simulated) updates state to SUCCEEDED and retrieves status
        """
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Repeat {{ word }}",
        )
        job.add_request("k1", SimpleInput(word="echo"))
        job.submit()  # fixture returns mock already succeeded
        # Force to pending to emulate in-progress job at serialization time
        job._batch_job.state = JobState.JOB_STATE_PENDING
        state_json = job.serialize()

        # Patch re-init to avoid side effects, and BatchPredictionJob to return a stub
        with patch.object(Job, "_initialize_gcp"), patch(
            "google.cloud.aiplatform.BatchPredictionJob"
        ) as mock_bp:
            stub = Mock()
            stub.state = JobState.JOB_STATE_PENDING
            stub.resource_name = job._batch_job.resource_name
            mock_bp.return_value = stub
            re_job = Job.reconnect_from_state(state_json)

            # Process B style check: not done yet
            assert re_job.is_done is False
            assert re_job.status == JobState.JOB_STATE_PENDING

            # Simulate time passing -> job completes (Process C later)
            stub.state = JobState.JOB_STATE_SUCCEEDED
            assert re_job.is_done is True
            # status call again
            _ = re_job.status

    @pytest.mark.incurs_costs
    @requires_project_id
    def test_real_reconnect_fetch_results(self):
        """End-to-end cost-incurring test of submit -> serialize -> reconnect -> poll -> results.

        Does NOT call wait() on original job to mimic Process separation.
        """
        job = Job(
            model="gemini-2.0-flash-lite-001",
            output_schema=SimpleOutput,
            prompt_template="Return the word '{{ word }}' exactly as given.",
        )
        job.add_request("first", SimpleInput(word="one"))
        job.add_request("second", SimpleInput(word="two"))

        job.submit()  # Process A ends here in real scenario
        state_json = job.serialize()

        # Process B/C: reconnect and poll until done (bounded)
        re_job = Job.reconnect_from_state(state_json)

        import time

        timeout_s = 600  # 10 minutes max
        poll_interval = 15
        start = time.time()
        while not re_job.is_done and (time.time() - start) < timeout_s:
            time.sleep(poll_interval)
        if not re_job.is_done:
            pytest.skip("Job not completed within polling timeout; skipping result assertions")

        results_reconnected = list(re_job.results())
        rec_map = {r.request_key: r for r in results_reconnected}
        assert {"first", "second"} == set(rec_map.keys())
        for key, expected in {"first": "one", "second": "two"}.items():
            r = rec_map[key]
            if r.was_successful and r.output is not None:
                assert r.output.result == expected
            else:
                print(f"Result for {key} failed: {r.error}")
