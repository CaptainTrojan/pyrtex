#!/usr/bin/env python3
"""
Example 6: Multi-File Input

Demonstrates processing multiple files together in a single request.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from pyrtex import Job


class PropertyAnalysis(BaseModel):
    """Analysis combining property data and business card information."""

    property_type: str
    estimated_price: float
    buyer_name: str
    buyer_company: str
    recommendation: str
    reason: str


class MultiFileInput(BaseModel):
    """Input schema with multiple files in one request."""

    property_file: str = Field(description="Path to property data file")
    business_card: str = Field(description="Path to business card image")


def main():
    data_dir = Path(__file__).parent / "data"

    # Define combinations of property + business card
    combinations = [
        ("luxury_condo.yaml", "business_card_1.png"),
        ("suburban_house.yaml", "business_card_2.png"),
        ("office_building.json", "business_card_3.png"),
    ]

    # Check if files exist
    valid_combinations = []
    for prop_file, card_file in combinations:
        prop_path = data_dir / prop_file
        card_path = data_dir / card_file
        if prop_path.exists() and card_path.exists():
            valid_combinations.append((prop_path, card_path))

    if not valid_combinations:
        print("No valid file combinations found. Run generate_sample_data.py first.")
        return

    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=PropertyAnalysis,
        prompt_template="""
Analyze the property data and business card together.

Property details: {{ property_file }}
Potential buyer: {{ business_card }}

Based on both files, provide a purchase recommendation.
""",
    )

    # Add each combination as a single request with multiple files
    for i, (prop_path, card_path) in enumerate(valid_combinations, 1):
        job.add_request(
            f"combo_{i}",
            MultiFileInput(property_file=str(prop_path), business_card=str(card_path)),
        )

    print(f"Processing {len(valid_combinations)} property-buyer combinations...")

    for result in job.submit().wait().results():
        if result.was_successful:
            analysis = result.output
            print(f"\n{result.request_key}:")
            print(f"  Property: {analysis.property_type}")
            print(f"  Price: ${analysis.estimated_price:,.0f}")
            print(f"  Buyer: {analysis.buyer_name} ({analysis.buyer_company})")
            print(f"  Recommendation: {analysis.recommendation}")
            print(f"  Reason: {analysis.reason}")


if __name__ == "__main__":
    main()
