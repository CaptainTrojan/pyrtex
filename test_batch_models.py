#!/usr/bin/env python3
"""
Batch Model Support Test

This script tests which Gemini models actually support batch predictions in Vertex AI.
For each model, it attempts to submit a minimal batch prediction job via pyrtex.

Usage:
    python test_batch_models.py
"""

import logging
import sys
from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel

from src.pyrtex import Job

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single model test."""

    model: str
    success: bool = False
    error: Optional[str] = None
    notes: str = ""

    def to_dict(self):
        return {
            "Model": self.model,
            "Status": "✓" if self.success else "✗",
            "Error": self.error or "—",
            "Notes": self.notes,
        }


class SimpleOutput(BaseModel):
    """Minimal output schema for batch testing."""

    response: str


class TestInput(BaseModel):
    """Minimal input schema for batch testing."""

    prompt: str


def submit_model_job(model: str):
    """
    Submit a batch job for a model. Returns (model_name, job, error).
    If submission fails, job will be None and error will be the exception.
    """
    try:
        logger.info(f"Submitting {model}...")

        # Create a job using pyrtex
        job = Job(
            model=model,
            output_schema=SimpleOutput,
            prompt_template="{{ prompt }}",
        )

        # Add a single minimal request
        job.add_request(
            "test_req",
            TestInput(prompt="Say HELLO"),
        )

        # Attempt submission
        job.submit()
        logger.info(f"✓ {model} submitted")
        return model, job, None

    except Exception as e:
        logger.error(f"✗ {model} submission failed: {type(e).__name__}")
        return model, None, e

def main():
    """Main test runner with submit-then-wait flow."""
    print("\n" + "="*80)
    print("GEMINI BATCH PREDICTION MODEL SUPPORT TEST (via Pyrtex)")
    print("="*80 + "\n")

    # Models to test - organized by family
    models_to_test = [
        # Gemini 3.1 & 3.0 (Newest)
        ("gemini-3.1-pro", "Gemini 3.1 Family (Preview)"),
        ("gemini-3.1-flash", None),
        ("gemini-3.1-flash-lite", None),
        ("gemini-3.1-flash-image", None),
        ("gemini-3-flash", None),
        ("gemini-3-pro-image", None),
        # Gemini 2.5 (Generally Available)
        ("gemini-2.5-pro", "Gemini 2.5 Family (GA)"),
        ("gemini-2.5-flash", None),
        ("gemini-2.5-flash-lite", None),
        ("gemini-2.5-flash-image", None),
        # Gemini 2.0
        ("gemini-2.0-flash-001", "Gemini 2.0 Family"),
        ("gemini-2.0-flash-lite-001", None),
        # Gemini 1.5
        ("gemini-1.5-pro-001", "Gemini 1.5 Family"),
        ("gemini-1.5-pro-002", None),
        ("gemini-1.5-flash-001", None),
        ("gemini-1.5-flash-002", None),
        ("gemini-1.5-flash-8b", None),
    ]

    # Submit all tests first so jobs can run in parallel on Vertex AI.
    current_family = None
    submission_results = {}

    for model, family_header in models_to_test:
        if family_header:
            if current_family:
                print()  # Spacing between families
            current_family = family_header
            print(f"\n{family_header}")
            print("-" * len(family_header))

        model_name, job, error = submit_model_job(model)
        submission_results[model_name] = (job, error)

    # Then wait for all submitted jobs and collect final status.
    results: List[TestResult] = []
    for model, (job, submission_error) in submission_results.items():
        result = TestResult(model=model)

        if submission_error:
            result.error = str(submission_error)
            error_str = result.error.lower()
            if "unsupported" in error_str or "not found" in error_str:
                result.notes = "Model not supported for batch"
            elif "invalid" in error_str or "does not support" in error_str:
                result.notes = "Invalid model or no batch support"
            elif "permission" in error_str:
                result.notes = "Permission denied"
            else:
                result.notes = "Submission failed"
            results.append(result)
            continue

        try:
            logger.info(f"Waiting for {model} completion...")
            job.wait()
            result.success = True
            result.notes = "Completed successfully"
        except Exception as e:
            result.error = str(e)
            result.notes = "Job failed while waiting"

        results.append(result)
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80 + "\n")

    # Create formatted table
    headers = ["Model", "Status", "Error", "Notes"]

    # Calculate column widths
    rows = [r.to_dict() for r in results]
    col_widths = {}
    for header in headers:
        col_widths[header] = max(
            len(header), max((len(str(row.get(header, "—"))) for row in rows), default=0)
        )

    # Print header
    header_row = " | ".join(
        f"{header:<{col_widths[header]}}" for header in headers
    )
    print(header_row)
    print("-" * len(header_row))

    # Print rows
    for row in rows:
        row_str = " | ".join(
            f"{str(row.get(header, '—')):<{col_widths[header]}}" for header in headers
        )
        print(row_str)

    # Summary statistics
    print("\n" + "="*80)
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    print(f"Summary: {successful}/{len(results)} models passed submission")
    print(f"         {failed} models failed submission")
    print("="*80 + "\n")

    # Detailed report
    print("DETAILED RESULTS:\n")

    print(f"✓ SUPPORTED MODELS ({successful}):")
    for result in results:
        if result.success:
            print(f"  • {result.model}")
            if result.notes:
                print(f"    Note: {result.notes}")

    if failed > 0:
        print(f"\n✗ UNSUPPORTED/FAILED MODELS ({failed}):")
        for result in results:
            if not result.success:
                print(f"  • {result.model}")
                if result.error:
                    # Truncate long error messages
                    error_msg = result.error
                    if len(error_msg) > 100:
                        error_msg = error_msg[:97] + "..."
                    print(f"    Error: {error_msg}")
                if result.notes:
                    print(f"    Note: {result.notes}")

    print("\n" + "="*80)
    print("Test completed!")
    print("="*80 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
