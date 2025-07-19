#!/usr/bin/env python3
"""
Example 2: Enum Constraints for Structured Output

This example demonstrates using Python enums to constrain model outputs to 
specific valid values. By defining enum types in your Pydantic schema, you can
ensure the AI model only returns values from predefined categories.

Key Features Demonstrated:
- Python str Enums for constrained categorical outputs
- Batch processing with enum validation
- Schema generation with enum constraints for Gemini function calling
- Mixed field types (enums + primitives) in output schemas

We'll analyze 4 different real estate properties and extract:
- Property type (enum: house, condo, apartment, office, retail, townhouse, land)
- Listing price (int: dollar amount)
- Investment grade (enum: excellent, good, fair, poor)
"""

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from pyrtex import Job


class PropertyType(str, Enum):
    """Enum for property types."""
    HOUSE = "house"
    CONDO = "condo"
    APARTMENT = "apartment"
    OFFICE = "office"
    RETAIL = "retail"
    TOWNHOUSE = "townhouse"
    LAND = "land"


class InvestmentGrade(str, Enum):
    """Enum for investment attractiveness."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    

class FileInput(BaseModel):
    """Input schema for text file processing."""
    file_path: str = Field(description="Path to the text file to process")


class RealEstateAnalysis(BaseModel):
    """Minimized real estate analysis with enum constraints."""
    
    # Basic Property Info (with enum constraint)
    property_type: PropertyType = Field(description="Type of property")
    listing_price: int = Field(description="Listing/asking price in dollars")
    investment_grade: InvestmentGrade = Field(description="Investment attractiveness grade")


def main():
    """Process real estate files to demonstrate enum constraints in output schemas."""
    
    # Get the data directory
    data_dir = Path(__file__).parent / "data"
    
    # Define the real estate files to process
    files_to_process = [
        "luxury_condo.yaml",
        "suburban_house.yaml", 
        "office_building.json",
        "apartment_complex.json"
    ]
    
    # Check if files exist
    missing_files = []
    for filename in files_to_process:
        file_path = data_dir / filename
        if not file_path.exists():
            missing_files.append(filename)
    
    if missing_files:
        print("❌ Missing required real estate files:")
        for filename in missing_files:
            print(f"   • {filename}")
        print("\n💡 Run generate_sample_data.py first to create the sample files.")
        return
    
    print("🏠 Processing real estate files with enum constraints...")
    print("📁 Properties to analyze:")
    for filename in files_to_process:
        print(f"   • {filename}")
    
    # Create the job
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=RealEstateAnalysis,
        prompt_template="""
You are a professional real estate analyst. Analyze the provided property data file and extract key information using the specified enum constraints.

The file contains detailed information about a real estate property. Your task is to extract and standardize the following information:

**Property Type**: Must be one of: house, condo, apartment, office, retail, townhouse, land
**Listing Price**: The asking price in dollars (whole number)
**Investment Grade**: Must be one of: excellent, good, fair, poor

Focus on these three key data points and ensure the property type and investment grade match the allowed enum values exactly.

**Property file to analyze:**
{{ text }}

Please extract concrete data points where available. If information is missing, make reasonable estimates based on the property details provided.
""",
    )
    
    # Add all files to the job as independent requests
    print("\n📤 Adding property files to batch job...")
    for filename in files_to_process:
        file_path = data_dir / filename
        # Use the filename (without extension) as the request key
        property_key = filename.replace('.yaml', '').replace('.json', '')
        job.add_request(property_key, FileInput(file_path=str(file_path)))
        print(f"   ✓ Added {filename} as '{property_key}'")
    
    # Submit and wait for results
    print("\n🚀 Submitting batch job to Gemini...")
    print("⏳ Analyzing properties with enum constraints (this may take a few moments)...")
    
    try:
        results = list(job.submit().wait().results())
        
        print(f"\n✅ Batch analysis completed! Analyzed {len(results)} properties.")
        
        # Display results for each property
        print("\n" + "="*60)
        print("🏘️  REAL ESTATE ANALYSIS WITH ENUM CONSTRAINTS")
        print("="*60)
        
        # Collect all analyses for summary
        all_properties = {}
        total_value = 0
        
        for result in results:
            print(f"\n🏠 Property: {result.request_key}")
            print("-" * 40)
            
            if result.was_successful:
                analysis = result.output
                all_properties[result.request_key] = analysis
                
                total_value += analysis.listing_price
                
                print(f"🏗️  Property Type: {analysis.property_type}")
                print(f"   Type class: {type(analysis.property_type)}")
                print(f"💰 Listing Price: ${analysis.listing_price:,}")
                print(f"⭐ Investment Grade: {analysis.investment_grade}")
                print(f"   Grade class: {type(analysis.investment_grade)}")
                
                # Token usage
                if result.usage_metadata:
                    tokens = result.usage_metadata.get('totalTokenCount', 'N/A')
                    print(f"🔢 Tokens used: {tokens}")
                
            else:
                print(f"❌ Error analyzing property: {result.error}")
        
        # Portfolio summary
        if len(all_properties) > 1:
            print(f"\n" + "="*60)
            print("📊 PORTFOLIO SUMMARY")
            print("="*60)
            
            print(f"🏘️  Total Properties: {len(all_properties)}")
            print(f"💰 Total Portfolio Value: ${total_value:,}")
            
            # Property type distribution (enum values)
            property_types = [prop.property_type for prop in all_properties.values()]
            type_counts = {}
            for ptype in property_types:
                type_str = ptype.value if hasattr(ptype, 'value') else str(ptype)
                type_counts[type_str] = type_counts.get(type_str, 0) + 1
            print(f"🏗️  Property Types: {type_counts}")
            
            # Investment grade distribution (enum values)
            investment_grades = [prop.investment_grade for prop in all_properties.values()]
            grade_counts = {}
            for grade in investment_grades:
                grade_str = grade.value if hasattr(grade, 'value') else str(grade)
                grade_counts[grade_str] = grade_counts.get(grade_str, 0) + 1
            print(f"⭐ Investment Grades: {grade_counts}")
        
        print(f"\n🎉 Real estate analysis complete! Processed {len(results)} properties.")
        print("💡 Each property was analyzed using enum constraints to ensure consistent categorization.")
        print("🔧 Enum types help constrain model outputs to predefined, valid values.")
        print("✨ Benefits of enums: type safety, validation, and guaranteed consistent outputs!")
        
    except Exception as e:
        print(f"\n❌ Error during batch processing: {e}")
        print("💡 Make sure you have set up your Google Cloud credentials and project ID.")


if __name__ == "__main__":
    main()
