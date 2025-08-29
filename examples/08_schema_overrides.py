"""Example 08: Per-request schema and prompt template overrides.

This demonstrates how a single Job can return heterogeneous structured
outputs by supplying alternate output_schema and prompt_template in add_request().

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


class ProductInfo(BaseModel):
    product_name: str
    category: str
    price_range: str


# ------------------------------ Main Logic ------------------------------ #


def main():
    # Create a job with a GLOBAL default output schema (PersonInfo) and prompt template
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=PersonInfo,  # default schema
        prompt_template=(
            "Extract structured data. Return a function call that matches the schema.\n"
            "If describing a person, include name & email. For a company, include name"
            " & domain.\n"
            "Content: {{ text }}"
        ),
    )

    # Request 1 uses the default schema (PersonInfo) and default prompt
    job.add_request(
        request_key="person_1",
        data=ExtractInput(text="Alice Anderson can be reached at alice@example.com."),
    )

    # Request 2 overrides BOTH the output schema and prompt template
    job.add_request(
        request_key="company_1",
        data=ExtractInput(text="Acme Corp operates online at acme.com."),
        output_schema=CompanyInfo,  # per-request schema override
        prompt_template=(  # per-request prompt override
            "You are a business analyst. Extract company information from the text.\n"
            "Focus on the company name and their primary web domain.\n"
            "Text to analyze: {{ text }}"
        ),
    )

    # Request 3 again uses the default PersonInfo schema but with a custom prompt
    job.add_request(
        request_key="person_2",
        data=ExtractInput(text="Contact Bob via bob@sample.org for details."),
        prompt_template=(  # prompt override only, schema stays default
            "Acting as a contact manager, parse this text for personal details.\n"
            "Extract the person's name and email address carefully.\n"
            "Input: {{ text }}"
        ),
    )

    # Request 4 demonstrates overriding schema only (prompt stays default)
    job.add_request(
        request_key="product_1",
        data=ExtractInput(
            text="The iPhone 15 Pro is a premium smartphone priced around $999-1199."
        ),
        output_schema=ProductInfo,  # schema override only
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
