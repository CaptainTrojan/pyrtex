"""Example 08: Per-request output schema overrides.

This demonstrates how a single Job can return heterogeneous structured
outputs by supplying an alternate output_schema in add_request().

Run (simulation mode, no GCP access needed):
    python examples/08_schema_overrides.py
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from pyrtex.client import Job


# ----------------------------- Input Schema ----------------------------- #
class ExtractInput(BaseModel):
    text: str = Field(description="Raw text to analyze")


# ---------------------------- Output Schemas ---------------------------- #
class PersonInfo(BaseModel):
    name: str
    email: str


class CompanyInfo(BaseModel):
    company: str
    domain: str


# ------------------------------ Main Logic ------------------------------ #

def main():
    # Create a job with a GLOBAL default output schema (PersonInfo)
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=PersonInfo,  # default schema
        prompt_template=(
            "Extract structured data. Return a function call that matches the schema.\n"
            "If describing a person, include name & email. For a company, include name & domain.\n"
            "Content: {{ text }}"
        ),
    )

    # Request 1 uses the default schema (PersonInfo)
    job.add_request(
        request_key="person_1",
        data=ExtractInput(text="Alice Anderson can be reached at alice@example.com."),
    )

    # Request 2 overrides the output schema with CompanyInfo
    job.add_request(
        request_key="company_1",
        data=ExtractInput(text="Acme Corp operates online at acme.com."),
        output_schema=CompanyInfo,  # per-request override
    )

    # Request 3 again uses the default PersonInfo
    job.add_request(
        request_key="person_2",
        data=ExtractInput(text="Contact Bob via bob@sample.org for details."),
    )

    # Submit (simulation) & iterate mixed results
    for result in job.submit().wait().results():
        print(f"Request Key: {result.request_key}")
        if result.was_successful:
            print("  Parsed Output Type:", type(result.output).__name__)
            print("  Parsed Output:", result.output.model_dump())
        else:
            print("  Error:", result.error)
        print("  Usage Metadata:", result.usage_metadata)
        print("---")


if __name__ == "__main__":  # pragma: no cover
    main()
