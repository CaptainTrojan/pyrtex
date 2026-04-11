"""Example 11: Dynamic Schemas.

This demonstrates how to properly create dynamic output schemas for Pyrtex.
You might need this if your application reads schema definitions from a database
or an external API (like a JSON Schema) rather than hardcoded Python classes.

Limitations for Vertex AI (which Pyrtex enforces during validation):
1. 'anyOf' / Union types generally fail or are poorly supported. Stick to Optional.
2. Enums with values that can be interpreted as booleans ("yes", "no", "true", "false")
   are rejected. Use "approved", "rejected", etc.

Run (simulation mode, no GCP access needed):
    python examples/11_dynamic_schema.py
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from pyrtex.client import Job, create_model_from_schema


# ----------------------------- Input Schema ----------------------------- #
class DocumentInput(BaseModel):
    content: str = Field(description="Text to analyze")


def main():
    # This approach is best when your schema definitions come as JSON/Dictionaries.
    # We parse the dictionary into a Pydantic model recursively, which then passes
    # Pyrtex's _validate_schema() checks.
    json_schema_def = {
        "title": "CompanyExtract",
        "type": "object",
        "properties": {
            "company_name": {"type": "string"},
            "is_public": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "status": {"type": "string", "enum": ["active", "acquired", "bankrupt"]},
        },
        "required": ["company_name"],
    }

    DynamicCompanyModel = create_model_from_schema(json_schema_def)

    job = Job(
        model="gemini-2.5-flash-lite",
        output_schema=DynamicCompanyModel,
        prompt_template="Extract company info from: {{ content }}",
        simulation_mode=False,
    )

    job.add_request(
        "req2",
        DocumentInput(
            content="Globex is a public company. Status is active. Tags: tech, energy"
        ),
    )

    for result in job.submit().wait().results():
        print(f"Req2 Output Schema: {result.output.__class__.__name__}")
        if result.was_successful:
            print("Output data dict:", result.output.model_dump())
        else:
            print(f"Failed: {result.error}")


if __name__ == "__main__":
    main()
