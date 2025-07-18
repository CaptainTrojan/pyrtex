#!/usr/bin/env python3
"""
Example 1: Business Card Text Extraction

This example demonstrates extracting contact information from a business card image.
It shows how PyRTex can process images and extract structured data using Pydantic models.
"""

from pathlib import Path
from pydantic import BaseModel
from pyrtex import Job

# Define the input schema for the request
class ImageInput(BaseModel):
    image: Path  # Image file path 
    extraction_focus: str  # What to focus on extracting

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
    job = Job[ContactInfo](
        model="gemini-2.0-flash-lite-001",
        output_schema=ContactInfo,
        prompt_template="""
        Extract all contact information from this business card image.
        
        Focus on: {{ extraction_focus }}
                
        Please extract exactly what you see on the card:
        - Person's name and title
        - Company name
        - Email address
        - Phone number
        - Website
        - Physical address
        - List of services offered
        
        Be precise and extract exactly what appears on the business card.
        """
    )
    
    # Read business card image
    data_dir = Path(__file__).parent / "data"
    image_path = data_dir / "business_card.png"
    
    if not image_path.exists():
        print(f"Image file not found: {image_path}")
        print("Please run 'python generate_sample_data.py' first to create sample files.")
        return
    
    # Add the image processing request using Path object directly
    job.add_request("business_card_extraction", ImageInput(
        image=image_path,  # Pass Path object directly
        extraction_focus="contact information and business services"
    ))
    
    # Process and get results
    print("Extracting contact information from business card image...")
    print(f"Processing: {image_path}")
    
    # Submit job, wait for completion, then get results
    results = list(job.submit().wait().results())
    
    # Display results
    for result in results:
        if result.was_successful:
            contact = result.output
            print(f"\n--- Business Card Information ---")
            print(f"Name: {contact.name}")
            print(f"Title: {contact.title}")
            print(f"Company: {contact.company}")
            print(f"Email: {contact.email}")
            print(f"Phone: {contact.phone}")
            print(f"Website: {contact.website}")
            print(f"Address: {contact.address}")
            print(f"Services:")
            for service in contact.services:
                print(f"  • {service}")
        else:
            print(f"Error: {result.error}")
    
    print(f"\n--- Processing Stats ---")
    print(f"Input: Business card image ({image_path.name})")
    print(f"Tokens used: {results[0].usage_metadata.get('totalTokenCount', 'N/A') if results else 'N/A'}")

if __name__ == "__main__":
    main()
