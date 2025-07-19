#!/usr/bin/env python3
"""
Example 5: Batch Processing

Demonstrates processing multiple inputs in a single batch job.
"""

from pathlib import Path

from pydantic import BaseModel

from pyrtex import Job


class ContactInfo(BaseModel):
    """Contact information extracted from business card."""

    name: str
    company: str
    email: str
    phone: str


class ImageInput(BaseModel):
    """Input schema for image processing."""

    image: Path


def main():
    data_dir = Path(__file__).parent / "data"

    # Find all business card images
    business_cards = [
        "business_card_1.png",
        "business_card_2.png",
        "business_card_3.png",
    ]

    # Check if files exist
    existing_cards = []
    for card in business_cards:
        card_path = data_dir / card
        if card_path.exists():
            existing_cards.append(card_path)

    if not existing_cards:
        print("No business card images found. Run generate_sample_data.py first.")
        return

    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=ContactInfo,
        prompt_template="Extract contact information from this business card: {{ image }}",
    )

    # Add each business card as a separate request
    for i, card_path in enumerate(existing_cards, 1):
        job.add_request(f"card_{i}", ImageInput(image=card_path))

    # Process all cards in a single batch
    results = list(job.submit().wait().results())

    print(f"Processed {len(results)} business cards:")
    for result in results:
        if result.was_successful:
            contact = result.output
            print(f"\n{result.request_key}:")
            print(f"  Name: {contact.name}")
            print(f"  Company: {contact.company}")
            print(f"  Email: {contact.email}")
            print(f"  Phone: {contact.phone}")


if __name__ == "__main__":
    main()
