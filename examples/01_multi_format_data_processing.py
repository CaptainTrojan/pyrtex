#!/usr/bin/env python3
"""
Example 1: Multi-Format Data Processing

This example demonstrates PyRTex's ability to handle various data types and formats
in a single batch job. It processes business cards (images) to extract contact
information, showcasing different field types: strings, lists, booleans, and numbers.

Key Features Demonstrated:
- Image processing with Gemini vision models
- Mixed data types in output schema (str, list, bool, int, float)
- Batch processing of multiple files
- Proper result handling with request keys
"""

from pathlib import Path

from pydantic import BaseModel

from pyrtex import Job


class ImageInput(BaseModel):
    """Input schema for image processing."""
    image: Path


class ContactInfo(BaseModel):
    """Contact information with various data types."""
    # String fields
    name: str
    company: str
    email: str
    phone: str
    
    # List field
    services: list[str]
    
    # Boolean field
    has_social_media: bool
    
    # Numeric fields
    years_in_business: int
    estimated_annual_revenue: float  # in millions


def main():
    """Process business card images to extract structured contact information."""
    
    # Get the data directory
    data_dir = Path(__file__).parent / "data"
    
    # List of business card files to process
    business_cards = [
        "business_card_1.png",
        "business_card_2.png", 
        "business_card_3.png"
    ]
    
    # Check if files exist
    missing_files = []
    for filename in business_cards:
        file_path = data_dir / filename
        if not file_path.exists():
            missing_files.append(filename)
    
    if missing_files:
        print("❌ Missing business card files:")
        for filename in missing_files:
            print(f"   • {filename}")
        print("\n💡 Run generate_sample_data.py first to create the sample files.")
        return
    
    print("📇 Processing business card images...")
    print("🖼️  Files to analyze:")
    for card in business_cards:
        print(f"   • {card}")
    
    # Create the job
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=ContactInfo,
        prompt_template="""
Extract comprehensive contact information from this business card image.

Please identify and extract:
- **Name**: Person's full name
- **Company**: Company or organization name
- **Email**: Email address (if visible)
- **Phone**: Phone number (if visible)
- **Services**: List of services offered (infer from job title/company type)
- **Social Media**: Does the card show social media handles? (true/false)
- **Years in Business**: Estimate based on company presentation (whole number)
- **Revenue**: Estimate annual revenue in millions based on company size/type (decimal)

If specific information isn't visible, make reasonable estimates based on the business type and professional presentation.

Image: {{ image }}
""",
    )
    
    # Add each business card as a separate request
    print("\n📤 Adding business cards to batch job...")
    for i, card_file in enumerate(business_cards, 1):
        file_path = data_dir / card_file
        request_key = f"card_{i}"
        job.add_request(request_key, ImageInput(image=file_path))
        print(f"   ✓ Added {card_file} as '{request_key}'")
    
    # Submit and process
    print("\n🚀 Submitting batch job to Gemini...")
    print("⏳ Extracting contact information (this may take a few moments)...")
    
    try:
        results = list(job.submit().wait().results())
        
        print(f"\n✅ Batch processing completed! Processed {len(results)} business cards.")
        
        # Display results
        print("\n" + "="*70)
        print("📇 CONTACT INFORMATION EXTRACTION RESULTS")
        print("="*70)
        
        successful_extractions = 0
        total_revenue = 0.0
        total_years = 0
        all_services = []
        
        for result in results:
            print(f"\n📋 Business Card: {result.request_key}")
            print("-" * 40)
            
            if result.was_successful:
                contact = result.output
                successful_extractions += 1
                total_revenue += contact.estimated_annual_revenue
                total_years += contact.years_in_business
                all_services.extend(contact.services)
                
                print(f"👤 Name: {contact.name}")
                print(f"🏢 Company: {contact.company}")
                print(f"📧 Email: {contact.email}")
                print(f"📞 Phone: {contact.phone}")
                print(f"🛠️  Services: {', '.join(contact.services)}")
                print(f"📱 Has Social Media: {contact.has_social_media}")
                print(f"📅 Years in Business: {contact.years_in_business}")
                print(f"💰 Est. Annual Revenue: ${contact.estimated_annual_revenue:.1f}M")
                
                # Show data types
                print("\n🔍 Data Types Extracted:")
                print(f"   • name: {type(contact.name).__name__} = '{contact.name}'")
                print(f"   • services: {type(contact.services).__name__} = {contact.services}")
                print(f"   • has_social_media: {type(contact.has_social_media).__name__} = {contact.has_social_media}")
                print(f"   • years_in_business: {type(contact.years_in_business).__name__} = {contact.years_in_business}")
                print(f"   • estimated_annual_revenue: {type(contact.estimated_annual_revenue).__name__} = {contact.estimated_annual_revenue}")
                
                # Token usage
                if result.usage_metadata:
                    tokens = result.usage_metadata.get('totalTokenCount', 'N/A')
                    print(f"🔢 Tokens used: {tokens}")
                
            else:
                print(f"❌ Error: {result.error}")
        
        # Summary statistics
        if successful_extractions > 0:
            print(f"\n" + "="*70)
            print("📊 EXTRACTION SUMMARY")
            print("="*70)
            
            print(f"✅ Successfully processed: {successful_extractions}/{len(results)} cards")
            print(f"💰 Total estimated revenue: ${total_revenue:.1f}M")
            print(f"📈 Average years in business: {total_years / successful_extractions:.1f}")
            
            # Unique services
            unique_services = list(set(all_services))
            print(f"🛠️  Unique services offered: {len(unique_services)}")
            print(f"   Services: {', '.join(unique_services[:5])}{'...' if len(unique_services) > 5 else ''}")
        
        print(f"\n🎉 Contact extraction complete!")
        print("💡 This example demonstrated processing images with mixed data types:")
        print("   • Strings (name, company, email, phone)")
        print("   • Lists (services)")
        print("   • Booleans (has_social_media)")
        print("   • Integers (years_in_business)")
        print("   • Floats (estimated_annual_revenue)")
        
    except Exception as e:
        print(f"\n❌ Error during batch processing: {e}")
        print("💡 Make sure you have set up your Google Cloud credentials and project ID.")


if __name__ == "__main__":
    main()
