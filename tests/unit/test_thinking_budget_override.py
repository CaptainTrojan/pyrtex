from pydantic import BaseModel

from pyrtex.client import Job


class InputModel(BaseModel):
    content: str


class OutputModel(BaseModel):
    result: str


def test_thinking_budget_override_for_gemini_25_pro():
    """Ensure thinking_budget=0 is raised to 128 for gemini-2.5-pro model.

    This covers the branch in Job._create_jsonl_payload that enforces a minimum
    thinking budget when the model is 'gemini-2.5-pro' and the user attempts to
    disable thinking (budget=0).
    """
    # Default GenerationConfig has thinking_budget=0
    job = Job(
        model="gemini-2.5-pro",
        output_schema=OutputModel,
        prompt_template="Echo: {{ content }}",
        simulation_mode=True,  # Avoid any GCP initialization side effects
    )
    job.add_request("req1", InputModel(content="hello"))

    # Call the internal payload builder directly (safe: only text input, no GCS needed)
    payload = job._create_jsonl_payload()
    first_line = payload.splitlines()[0]
    data = __import__("json").loads(first_line)

    thinking_budget = data["request"]["generation_config"]["thinking_config"][
        "thinking_budget"
    ]

    assert (
        thinking_budget == 128
    ), "thinking_budget should be coerced to 128 for gemini-2.5-pro when set to 0"
