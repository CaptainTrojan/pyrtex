#!/usr/bin/env python3
"""
Example 2: Enum Constraints

Demonstrates using enums to constrain model outputs to specific valid values.
"""

from enum import Enum

from pydantic import BaseModel, Field

from pyrtex import Job


class Sentiment(str, Enum):
    """Sentiment classification options."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Priority(str, Enum):
    """Priority levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EmailAnalysis(BaseModel):
    """Email analysis with enum constraints."""

    sentiment: Sentiment = Field(description="Overall sentiment of the email")
    priority: Priority = Field(description="Urgency level")
    summary: str = Field(description="Brief summary")


class EmailInput(BaseModel):
    """Input schema for email text."""

    email_text: str


def main():
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=EmailAnalysis,
        prompt_template="Analyze this email: {{ email_text }}",
    )

    # Add sample email
    job.add_request(
        "email1",
        EmailInput(
            email_text=(
                "URGENT: The server is down and customers are complaining! "
                "Please fix ASAP!"
            )
        ),
    )

    for result in job.submit().wait().results():
        if result.was_successful:
            analysis = result.output
            print(f"Sentiment: {analysis.sentiment}")
            print(f"Priority: {analysis.priority}")
            print(f"Summary: {analysis.summary}")


if __name__ == "__main__":
    main()
