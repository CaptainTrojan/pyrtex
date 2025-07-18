#!/usr/bin/env python3
"""
Example 1: Batch Business Card Text Extraction

This example demonstrates the power of batch processing by extracting contact
information from multiple business cards with different designs and layouts.
It shows how PyRTex can process various formats in a single batch and extract
consistent structured data regardless of visual design differences.

IMPORTANT: This example also demonstrates proper result handling. Results from
BigQuery may not preserve the original submission order, so we use request keys
to map results back to the correct input files.
"""

from pathlib import Path

from pydantic import BaseModel

from pyrtex import Job


# Define the input schema for the request
class ImageInput(BaseModel):
    image: Path  # Image file path


# Define the output schema for contact information
class ContactInfo(BaseModel):
    name: str
    title: str
    company: str
    email: str
    phone: str
    website: str
    address: str
    services: list[str]


def main():
    # Set up the job with Gemini model for image processing
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=ContactInfo,
        prompt_template="""
        Extract all contact information from this business card image.

        Please extract exactly what you see on the card:
        - Person's name and title
        - Company name
        - Email address
        - Phone number
        - Website
        - Physical address
        - List of services offered

        Be precise and extract exactly what appears on the business card.
        Return empty strings for any fields not visible on the card.
        """,
    )    # Define business cards to process
    data_dir = Path(__file__).parent / "data"
    business_cards = [
        {
            "file": "business_card_1.png",
        },
        {
            "file": "business_card_2.png",
        },
        {
            "file": "business_card_3.png",
        },
    ]

    # Check if all files exist
    missing_files = []
    for card in business_cards:
        card_path = data_dir / card["file"]
        if not card_path.exists():
            missing_files.append(card["file"])

    if missing_files:
        print(f"Missing business card files: {', '.join(missing_files)}")
        print(
            "Please run 'python generate_sample_data.py' first to create sample files."
        )
        return

    # Add all business cards to the batch for processing
    print("Setting up batch processing for multiple business card designs...")
    for i, card in enumerate(business_cards, 1):
        card_path = data_dir / card["file"]
        # Use the filename as the request key for easy mapping later
        request_key = card["file"].replace(".png", "")  # e.g., "business_card_1"
        job.add_request(
            request_key,
            ImageInput(
                image=card_path,
            ),
        )
        print(f"  Added: {card['file']} (key: {request_key})")

    # Process all cards in a single batch
    print(f"\nProcessing {len(business_cards)} business cards in batch...")
    print("This demonstrates PyRTex's ability to handle different formats uniformly.")

    # Submit job, wait for completion, then get results
    results = list(job.submit().wait().results())

    # IMPORTANT: Results may not come back in the same order as submitted!
    # Create a lookup dictionary by request key for proper mapping
    results_by_key = {result.request_key: result for result in results}

    # Display results for each card in the original order
    print(f"\n{'='*80}")
    print("BATCH PROCESSING RESULTS - UNIFIED DATA EXTRACTION")
    print(f"{'='*80}")

    total_tokens = 0
    for i, card_info in enumerate(business_cards, 1):
        # Map back to the correct result using the request key
        request_key = card_info["file"].replace(".png", "")
        result = results_by_key[request_key]

        print(f"\n--- Business Card {i} ---")
        print(f"File: {card_info['file']} (Request Key: {request_key})")

        if result.was_successful:
            contact = result.output
            print("✅ Extraction successful!")
            print(f"Name: {contact.name}")
            print(f"Title: {contact.title}")
            print(f"Company: {contact.company}")
            print(f"Email: {contact.email}")
            print(f"Phone: {contact.phone}")
            print(f"Website: {contact.website}")
            print(f"Address: {contact.address}")
            services_text = (
                ', '.join(contact.services) if contact.services else 'None listed'
            )
            print(f"Services: {services_text}")

            # Track token usage
            if hasattr(result, "usage_metadata") and result.usage_metadata:
                tokens = result.usage_metadata.get("totalTokenCount", 0)
                total_tokens += tokens
                print(f"Tokens used: {tokens}")
        else:
            print(f"❌ Error: {result.error}")

    # Summary statistics
    print(f"\n{'='*80}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total cards processed: {len(business_cards)}")
    print(f"Successful extractions: {sum(1 for r in results if r.was_successful)}")
    print(f"Failed extractions: {sum(1 for r in results if not r.was_successful)}")
    print(f"Total tokens used: {total_tokens}")
    print(f"Average tokens per card: {total_tokens / len(business_cards):.0f}")

    print("\n🎯 Key Insight: Despite different designs, layouts, and color schemes,")
    print("   PyRTex successfully extracted consistent structured data from all cards!")

    print("\n💡 Important Note: This example demonstrates proper result handling.")
    print("   Results from BigQuery may not preserve submission order, so we use")
    print("   request keys to map results back to the correct input files.")


if __name__ == "__main__":
    main()
