#!/usr/bin/env python3
"""
Example 5: Multi-File Input Processing

This example demonstrates PyRTex's ability to process multiple files together
in a single request. We'll combine real estate property data with business card
information to make a simple purchase recommendation.

Key Features Demonstrated:
- Multi-file input processing (text file + image per request)
- Mixed media types in single requests (YAML/JSON + PNG)
- Simple enum output with probability scoring
- Batch processing with multiple file combinations

We'll answer a simple question: "Should this person buy this property?"
"""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from pyrtex import Job


class Recommendation(str, Enum):
    """Simple yes/no recommendation."""
    YES = "recommend"
    NO = "not_recommend"


class MultiFileInput(BaseModel):
    """Input schema for multi-file processing."""
    property_file: str = Field(description="Path to the property data file")
    business_card: str = Field(description="Path to the business card image")


class PurchaseRecommendation(BaseModel):
    """Simple purchase recommendation with probability."""
    
    recommendation: Recommendation = Field(description="Should they buy this property?")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    main_reason: str = Field(description="Primary reason for the recommendation")


def main():
    """Demonstrate multi-file input processing with simple recommendations."""
    
    # Get the data directory
    data_dir = Path(__file__).parent / "data"
    
    # Define property-buyer combinations to analyze
    analysis_pairs = [
        {
            "name": "Tech Professional + Luxury Condo",
            "property": "luxury_condo.yaml",
            "business_card": "business_card_1.png"
        },
        {
            "name": "Business Owner + Suburban House", 
            "property": "suburban_house.yaml",
            "business_card": "business_card_2.png"
        },
        {
            "name": "Finance Exec + Office Building",
            "property": "office_building.json",
            "business_card": "business_card_3.png"
        }
    ]
    
    # Check if files exist
    missing_files = []
    for pair in analysis_pairs:
        property_path = data_dir / pair["property"]
        card_path = data_dir / pair["business_card"]
        if not property_path.exists():
            missing_files.append(pair["property"])
        if not card_path.exists():
            missing_files.append(pair["business_card"])
    
    if missing_files:
        print("❌ Missing required files:")
        for filename in set(missing_files):
            print(f"   • {filename}")
        print("\n💡 Run generate_sample_data.py first to create the sample files.")
        return
    
    print("📁 + �️  Demonstrating multi-file input processing...")
    print("🔄 File combinations to analyze:")
    for pair in analysis_pairs:
        print(f"   • {pair['name']}: {pair['property']} + {pair['business_card']}")
    
    # Create the job
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=PurchaseRecommendation,
        prompt_template="""
You are a real estate advisor. Look at the property details and the business card, then answer a simple question:

**Should this person buy this property?**

Consider:
- Property price vs likely income from the business card
- Property type vs person's likely needs
- Basic affordability and suitability

Give a simple yes/no recommendation with confidence score and main reasoning.

Keep it simple and focused.
""",
    )
    
    # Add each property-buyer combination as a separate request
    print("\n📤 Adding file pairs to batch job...")
    for i, pair in enumerate(analysis_pairs, 1):
        property_path = str(data_dir / pair["property"])
        card_path = str(data_dir / pair["business_card"])
        
        request_key = f"analysis_{i}"
        job.add_request(
            request_key, 
            MultiFileInput(
                property_file=property_path,
                business_card=card_path
            )
        )
        print(f"   ✓ Added {pair['name']} as '{request_key}'")
    
    # Submit and process
    print("\n🚀 Submitting multi-file batch job to Gemini...")
    print("⏳ Processing file combinations (this may take a few moments)...")
    
    try:
        results = list(job.submit().wait().results())
        
        print(f"\n✅ Multi-file processing completed! Analyzed {len(results)} combinations.")
        
        # Display results
        print("\n" + "="*60)
        print("📁 + �️  MULTI-FILE PROCESSING RESULTS")
        print("="*60)
        
        yes_count = 0
        total_confidence = 0.0
        
        for i, result in enumerate(results, 1):
            pair = analysis_pairs[i-1]
            print(f"\n🔍 Analysis {i}: {pair['name']}")
            print("-" * 50)
            
            if result.was_successful:
                recommendation = result.output
                
                # Track statistics
                if recommendation.recommendation == Recommendation.YES:
                    yes_count += 1
                total_confidence += recommendation.confidence
                
                print(f"📄 Property File: {pair['property']}")
                print(f"🖼️  Business Card: {pair['business_card']}")
                print(f"📊 Recommendation: {recommendation.recommendation.upper()}")
                print(f"🎯 Confidence: {recommendation.confidence:.2f}")
                print(f"💡 Main Reason: {recommendation.main_reason}")
                
                # Token usage
                if result.usage_metadata:
                    tokens = result.usage_metadata.get('totalTokenCount', 'N/A')
                    print(f"🔢 Tokens used: {tokens}")
                
            else:
                print(f"❌ Error: {result.error}")
        
        # Summary
        if len(results) > 0:
            print(f"\n" + "="*60)
            print("� PROCESSING SUMMARY")
            print("="*60)
            
            print(f"✅ Successful analyses: {len([r for r in results if r.was_successful])}/{len(results)}")
            print(f"� Positive recommendations: {yes_count}/{len(results)}")
            print(f"🎯 Average confidence: {total_confidence/len(results):.2f}")
        
        print(f"\n🎉 Multi-file processing complete!")
        print("💡 This example demonstrated:")
        print("   • Processing multiple file types together (YAML/JSON + PNG)")
        print("   • Mixed media input in single requests")
        print("   • Simple enum outputs with probability scoring")
        print("   • Clean separation of input files and processing logic")
        
    except Exception as e:
        print(f"\n❌ Error during batch processing: {e}")
        print("💡 Make sure you have set up your Google Cloud credentials and project ID.")


if __name__ == "__main__":
    main()
