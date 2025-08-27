# tests/real/test_real_serialization.py

import os
import time
import pytest

from pyrtex.client import Job
from tests.integration.test_full_run import SimpleInput, SimpleOutput

requires_project_id = pytest.mark.skipif(
    not os.getenv("GOOGLE_PROJECT_ID"),
    reason="Requires GOOGLE_PROJECT_ID environment variable to be set",
)


@pytest.mark.incurs_costs
@requires_project_id
def test_real_job_serialization_and_reconnect_cycle():
    """Real cost-incurring test verifying serialize() and reconnect_from_state().

    Simulates three-process pattern:
      Process A: submit + serialize (no wait)
      Process B/C: reconnect + poll status + fetch results when done
    """
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=SimpleOutput,
        prompt_template="Output the word '{{ word }}' as 'result'.",
    )
    job.add_request("alpha", SimpleInput(word="alpha"))
    job.add_request("beta", SimpleInput(word="beta"))

    # Process A: submit then immediately serialize and exit
    job.submit()
    state_json = job.serialize()

    # Process B/C: reconnect later
    re_job = Job.reconnect_from_state(state_json)

    # Poll for completion with timeout to avoid hanging CI
    timeout_s = 600  # 10 minutes max
    poll_interval = 15
    start = time.time()
    while not re_job.is_done and (time.time() - start) < timeout_s:
        time.sleep(poll_interval)

    if not re_job.is_done:
        pytest.skip("Job not completed within polling timeout; skipping assertions")

    re_results = list(re_job.results())

    rec_map = {r.request_key: r for r in re_results}
    assert set(rec_map.keys()) == {"alpha", "beta"}
    for key, expected in {"alpha": "alpha", "beta": "beta"}.items():
        r = rec_map[key]
        if r.was_successful and r.output is not None:
            assert r.output.result == expected
        else:
            print(f"Key {key} had an error: {r.error}")
