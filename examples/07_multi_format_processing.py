#!/usr/bin/env python3
"""
Example 7: Multi-Format Processing

Demonstrates processing different file types (images, PDFs, text, audio, video)
in a single batch.
"""

from pathlib import Path
from typing import Union

from pydantic import BaseModel

from pyrtex import Job


class ExtractedData(BaseModel):
    """Generic extracted data from any file type."""

    file_type: str
    content_summary: str
    key_information: list[str]
    data_points: dict[str, str]


class FileInput(BaseModel):
    """Input schema for any file type."""

    file_path: Union[str, Path]


def main():
    data_dir = Path(__file__).parent / "data"

    # Define different file types to process
    files_to_process = [
        ("sample_invoice.pdf", "PDF document"),
        ("product_catalog.png", "Image file"),
        ("business_card_1.png", "Image file"),
        ("luxury_condo.yaml", "Text file"),
        ("office_building.json", "Text file"),
        ("test_minimal.mp4", "Video file"),
        ("test_minimal.wav", "Audio file"),
    ]

    # Check which files exist
    valid_files = []
    for filename, file_type in files_to_process:
        file_path = data_dir / filename
        if file_path.exists():
            valid_files.append((file_path, file_type))

    if not valid_files:
        print("No sample files found. Run generate_sample_data.py first.")
        return

    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=ExtractedData,
        prompt_template="""
Extract key information from this file: {{ file_path }}

For audio/video files, analyze the content and provide a transcript summary
if applicable. For other files, extract relevant data.

Provide:
- A brief content summary
- List of key information points
- Important data points as key-value pairs
""",
    )

    # Add each file type to the batch
    for file_path, file_type in valid_files:
        filename = file_path.name
        job.add_request(filename, FileInput(file_path=file_path))

    print(f"Processing {len(valid_files)} files of different types:")
    for file_path, file_type in valid_files:
        print(f"  - {file_path.name} ({file_type})")

    print("\nExtracting data from all file types...")

    for result in job.submit().wait().results():
        if result.was_successful:
            data = result.output
            print(f"\n📄 {result.request_key} ({data.file_type}):")
            print(f"   Summary: {data.content_summary}")
            print(f"   Key Info: {', '.join(data.key_information[:3])}...")
            print(f"   Data Points: {len(data.data_points)} extracted")
        else:
            print(f"❌ Error processing {result.request_key}: {result.error}")


if __name__ == "__main__":
    main()
