#!/usr/bin/env python3
"""
Example 6: Multi-Attachment Requests

Demonstrates the variable-length ``attachments=`` API: each request can carry
any number of files, and attachments may come from local paths, ``s3://``
URIs (with ``pip install 'pyrtex[s3]'``), or ``gs://`` URIs.
"""

from pathlib import Path

from pydantic import BaseModel

from pyrtex import Job


class PropertyAnalysis(BaseModel):
    """Comparison of a property against multiple potential buyers."""

    property_type: str
    estimated_price: float
    best_buyer_name: str
    best_buyer_company: str
    recommendation: str
    reason: str


def main():
    data_dir = Path(__file__).parent / "data"

    # Each request bundles one property file with the business cards of three
    # potential buyers — the model picks the best fit across all attachments.
    # Different requests can carry different numbers of attachments.
    requests = [
        (
            "luxury_condo",
            [
                data_dir / "luxury_condo.yaml",
                data_dir / "business_card_1.png",
                data_dir / "business_card_2.png",
                data_dir / "business_card_3.png",
            ],
        ),
        (
            "suburban_house",
            [
                data_dir / "suburban_house.yaml",
                data_dir / "business_card_1.png",
                data_dir / "business_card_2.png",
            ],
        ),
        (
            "office_building",
            [
                data_dir / "office_building.json",
                data_dir / "business_card_3.png",
            ],
        ),
    ]

    # Skip requests whose fixtures are missing.
    requests = [
        (key, files) for key, files in requests if all(f.exists() for f in files)
    ]
    if not requests:
        print("No valid file combinations found. Run generate_sample_data.py first.")
        return

    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=PropertyAnalysis,
        prompt_template=(
            "You are given a property data file followed by one or more "
            "potential buyers' business cards. Pick the buyer that best fits "
            "the property and explain why."
        ),
    )

    for key, files in requests:
        job.add_request(key, attachments=files)

    # The same API accepts remote URIs — pyrtex fetches them in your process
    # and stages them into the job's GCS bucket. Useful from Lambda contexts
    # where you'd otherwise download to /tmp first:
    #
    #   job.add_request(
    #       "from_s3",
    #       attachments=[
    #           "s3://my-bucket/contracts/2025-01-lease.pdf",
    #           "gs://shared-assets/floorplans/unit-204.png",
    #       ],
    #   )

    print(f"Processing {len(requests)} property analyses...")
    for result in job.submit().wait().results():
        if result.was_successful:
            analysis = result.output
            print(f"\n{result.request_key}:")
            print(f"  Property: {analysis.property_type}")
            print(f"  Estimated price: ${analysis.estimated_price:,.0f}")
            print(
                f"  Best buyer: {analysis.best_buyer_name} "
                f"({analysis.best_buyer_company})"
            )
            print(f"  Recommendation: {analysis.recommendation}")
            print(f"  Reason: {analysis.reason}")


if __name__ == "__main__":
    main()
