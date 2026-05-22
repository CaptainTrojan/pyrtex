#!/usr/bin/env python3
"""
Batch Model Test Results Export & Analysis Utility

This script provides utility functions for:
- Exporting test results to CSV
- Comparing results across test runs
- Analyzing model compatibility patterns

Can be used standalone or imported by test_batch_models.py
"""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TestResult:
    """Result of a single model test."""

    model: str
    submission_successful: bool
    submission_error: Optional[str] = None
    job_resource_name: Optional[str] = None
    job_state: Optional[str] = None
    job_error: Optional[str] = None
    notes: str = ""


def export_results_to_csv(results: List[TestResult], output_path: str) -> None:
    """Export test results to a CSV file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "submission_successful",
        "submission_error",
        "job_resource_name",
        "job_state",
        "job_error",
        "notes",
    ]

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    print(f"✓ Results exported to: {output_file}")


def export_results_to_json(results: List[TestResult], output_path: str) -> None:
    """Export test results to a JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "summary": {
            "total_models_tested": len(results),
            "models_supported": sum(1 for r in results if r.submission_successful),
            "models_failed": sum(1 for r in results if not r.submission_successful),
        },
        "results": [asdict(r) for r in results],
    }

    with open(output_file, "w") as jsonfile:
        json.dump(data, jsonfile, indent=2)

    print(f"✓ Results exported to: {output_file}")


def analyze_results(results: List[TestResult]) -> Dict:
    """Analyze test results and return insights."""
    successful_models = [r for r in results if r.submission_successful]
    failed_models = [r for r in results if not r.submission_successful]

    # Group by model family
    families = {}
    for result in results:
        family = extract_model_family(result.model)
        if family not in families:
            families[family] = {"total": 0, "passed": 0}
        families[family]["total"] += 1
        if result.submission_successful:
            families[family]["passed"] += 1

    # Analyze error patterns
    error_patterns = {}
    for result in failed_models:
        if result.submission_error:
            error_msg = result.submission_error.lower()
            # Categorize error
            if "unsupported" in error_msg or "not found" in error_msg:
                category = "Model Not Supported"
            elif "permission" in error_msg:
                category = "Permission Denied"
            elif "invalid" in error_msg:
                category = "Invalid Argument"
            elif "location" in error_msg or "region" in error_msg:
                category = "Region/Location Issue"
            else:
                category = "Other"

            if category not in error_patterns:
                error_patterns[category] = []
            error_patterns[category].append(result.model)

    return {
        "total_tested": len(results),
        "total_passed": len(successful_models),
        "total_failed": len(failed_models),
        "pass_rate": (
            len(successful_models) / len(results) * 100 if results else 0
        ),
        "by_family": families,
        "error_patterns": error_patterns,
        "successful_models": [r.model for r in successful_models],
        "failed_models": [r.model for r in failed_models],
    }


def extract_model_family(model_name: str) -> str:
    """Extract model family from model name."""
    # Handle custom model paths
    if "/" in model_name:
        model_name = model_name.split("/")[-1]

    # Extract family (e.g., "gemini-3.1" from "gemini-3.1-flash")
    parts = model_name.split("-")
    if len(parts) >= 2:
        # Handle version numbers
        if parts[0] == "gemini":
            if len(parts) >= 3 and "." in parts[1]:
                return f"{parts[0]}-{parts[1]}"
            else:
                return parts[0]
    return model_name


def print_analysis(analysis: Dict) -> None:
    """Print formatted analysis results."""
    print("\n" + "="*80)
    print("ANALYSIS REPORT")
    print("="*80)

    print(f"\nOverall Statistics:")
    print(f"  Total models tested: {analysis['total_tested']}")
    print(f"  Passed: {analysis['total_passed']}")
    print(f"  Failed: {analysis['total_failed']}")
    print(f"  Pass rate: {analysis['pass_rate']:.1f}%")

    print(f"\nBy Model Family:")
    for family, stats in sorted(analysis["by_family"].items()):
        pass_rate = (
            stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
        )
        print(f"  {family}: {stats['passed']}/{stats['total']} ({pass_rate:.0f}%)")

    if analysis["error_patterns"]:
        print(f"\nError Patterns:")
        for pattern, models in sorted(analysis["error_patterns"].items()):
            print(f"  {pattern} ({len(models)} models):")
            for model in models[:3]:  # Show first 3
                print(f"    - {model}")
            if len(models) > 3:
                print(f"    ... and {len(models) - 3} more")

    print(f"\nSuccessful Models ({analysis['total_passed']}):")
    for model in sorted(analysis["successful_models"]):
        print(f"  ✓ {model}")

    if analysis["failed_models"]:
        print(f"\nFailed Models ({analysis['total_failed']}):")
        for model in sorted(analysis["failed_models"]):
            print(f"  ✗ {model}")

    print("\n" + "="*80 + "\n")


def compare_results(
    previous_results: List[TestResult], current_results: List[TestResult]
) -> Dict:
    """Compare two test runs and identify changes."""
    prev_successful = {r.model for r in previous_results if r.submission_successful}
    curr_successful = {r.model for r in current_results if r.submission_successful}

    newly_supported = curr_successful - prev_successful
    newly_failed = prev_successful - curr_successful
    still_supported = curr_successful & prev_successful
    still_failed = (
        {r.model for r in previous_results if not r.submission_successful}
        - newly_supported
    )

    return {
        "newly_supported": sorted(newly_supported),
        "newly_failed": sorted(newly_failed),
        "still_supported": sorted(still_supported),
        "still_failed": sorted(still_failed),
    }


def print_comparison(comparison: Dict) -> None:
    """Print formatted comparison results."""
    print("\n" + "="*80)
    print("COMPARISON WITH PREVIOUS RUN")
    print("="*80)

    if comparison["newly_supported"]:
        print(f"\n✓ Newly Supported ({len(comparison['newly_supported'])}):")
        for model in comparison["newly_supported"]:
            print(f"  • {model}")

    if comparison["newly_failed"]:
        print(f"\n✗ Newly Failed ({len(comparison['newly_failed'])}):")
        for model in comparison["newly_failed"]:
            print(f"  • {model}")

    if not comparison["newly_supported"] and not comparison["newly_failed"]:
        print("\nNo changes detected between test runs.")

    print(f"\nStill Supported: {len(comparison['still_supported'])}")
    print(f"Still Failed: {len(comparison['still_failed'])}")
    print("\n" + "="*80 + "\n")


def generate_markdown_report(results: List[TestResult], output_path: str) -> None:
    """Generate a markdown report of test results."""
    analysis = analyze_results(results)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write("# Batch Model Support Test Results\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Total Models Tested:** {analysis['total_tested']}\n")
        f.write(f"- **Passed:** {analysis['total_passed']}\n")
        f.write(f"- **Failed:** {analysis['total_failed']}\n")
        f.write(f"- **Pass Rate:** {analysis['pass_rate']:.1f}%\n\n")

        f.write("## Results by Model Family\n\n")
        for family, stats in sorted(analysis["by_family"].items()):
            pass_rate = (
                stats["passed"] / stats["total"] * 100
                if stats["total"] > 0
                else 0
            )
            f.write(
                f"### {family}\n"
                f"- **Status:** {stats['passed']}/{stats['total']} supported ({pass_rate:.0f}%)\n\n"
            )

        f.write("## Supported Models\n\n")
        for model in sorted(analysis["successful_models"]):
            f.write(f"- ✓ {model}\n")

        if analysis["failed_models"]:
            f.write("\n## Unsupported Models\n\n")
            for model in sorted(analysis["failed_models"]):
                f.write(f"- ✗ {model}\n")

        if analysis["error_patterns"]:
            f.write("\n## Error Categories\n\n")
            for pattern, models in sorted(analysis["error_patterns"].items()):
                f.write(f"### {pattern}\n")
                f.write(f"Affected models: {', '.join(models[:3])}")
                if len(models) > 3:
                    f.write(f", and {len(models) - 3} more")
                f.write("\n\n")

    print(f"✓ Markdown report generated: {output_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Batch Model Test Utilities")
        print("\nUsage:")
        print(
            "  python test_utils.py analyze <results.json>     # Analyze JSON results"
        )
        print(
            "  python test_utils.py compare <old.json> <new.json>  # Compare two runs"
        )
        print("\nExample:")
        print(
            "  python test_utils.py analyze results_2026-05-05.json"
        )
        sys.exit(0)

    command = sys.argv[1]

    if command == "analyze" and len(sys.argv) > 2:
        with open(sys.argv[2]) as f:
            data = json.load(f)
            results = [TestResult(**r) for r in data["results"]]
            analysis = analyze_results(results)
            print_analysis(analysis)
    elif command == "compare" and len(sys.argv) > 3:
        with open(sys.argv[2]) as f:
            prev_data = json.load(f)
            prev_results = [TestResult(**r) for r in prev_data["results"]]

        with open(sys.argv[3]) as f:
            curr_data = json.load(f)
            curr_results = [TestResult(**r) for r in curr_data["results"]]

        comparison = compare_results(prev_results, curr_results)
        print_comparison(comparison)
    else:
        print(f"Unknown command or missing arguments: {command}")
        sys.exit(1)
