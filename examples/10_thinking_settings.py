#!/usr/bin/env python3
"""
Example 10: Thinking Settings

Demonstrates using ThinkingConfig to control the model's reasoning process.
The thinking_budget parameter controls how much computational budget the model
allocates to internal reasoning before generating the final output.
"""

from pydantic import BaseModel, Field

from pyrtex import Job
from pyrtex.config import GenerationConfig, ThinkingConfig


class ComplexAnalysis(BaseModel):
    """Complex analysis output requiring careful reasoning."""

    reasoning_steps: list[str] = Field(description="Key reasoning steps taken")
    confidence_level: float = Field(description="Confidence in analysis (0.0-1.0)")
    main_conclusion: str = Field(description="Primary conclusion")
    alternative_interpretations: list[str] = Field(
        description="Other possible interpretations"
    )
    recommendations: list[str] = Field(
        description="Specific actionable recommendations"
    )


class AnalysisInput(BaseModel):
    """Input schema for complex analysis."""

    scenario: str


def main():
    print("=== Example: Comparing Thinking Budgets ===")

    scenarios = [
        ("quick", 0, "Quick decision needed"),
        ("balanced", 512, "Balanced analysis"),
        ("thorough", 2048, "Thorough evaluation"),
    ]

    for name, budget, description in scenarios:
        print(f"\n--- {description} (budget: {budget}) ---")

        job = Job(
            model="gemini-2.5-flash-lite",
            output_schema=ComplexAnalysis,
            prompt_template="Analyze the investment opportunity: {{ scenario }}",
            generation_config=GenerationConfig(
                temperature=0.2, thinking_config=ThinkingConfig(thinking_budget=budget)
            ),
        )

        job.add_request(
            f"investment_{name}",
            AnalysisInput(
                scenario=(
                    "A renewable energy startup is seeking Series A funding. They have "
                    "promising technology but no revenue yet. Market potential is huge "
                    "but highly competitive. The team has strong technical backgrounds "
                    "but limited business experience."
                )
            ),
        )

        for result in job.submit().wait().results():
            if result.was_successful:
                analysis = result.output
                print(f"  Steps: {len(analysis.reasoning_steps)}")
                print(f"  Confidence: {analysis.confidence_level:.2f}")
                print(f"  Recommendations: {len(analysis.recommendations)}")
            else:
                print(f"  Error: {result.error}")


if __name__ == "__main__":
    main()
