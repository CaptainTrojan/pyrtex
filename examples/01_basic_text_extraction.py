#!/usr/bin/env python3
"""
Example 1: Basic Text Extraction

Demonstrates the fundamental PyRTex workflow: extract structured data from text input.
"""

from pydantic import BaseModel

from pyrtex import Job


class PersonInfo(BaseModel):
    """Simple output schema for person information."""

    name: str
    age: int
    occupation: str


class TextInput(BaseModel):
    """Input schema for text processing."""

    text: str


def main():
    # Create a job with a simple prompt template
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=PersonInfo,
        prompt_template="Extract person information from: {{ text }}",
    )

    # Add a request
    job.add_request(
        "person1",
        TextInput(
            text="John Smith is a 30-year-old software engineer working at Tech Corp."
        ),
    )

    # Submit, wait, and get results
    for result in job.submit().wait().results():
        if result.was_successful:
            person = result.output
            print(f"Name: {person.name}")
            print(f"Age: {person.age}")
            print(f"Occupation: {person.occupation}")
        else:
            print(f"Error: {result.error}")


if __name__ == "__main__":
    main()
