#!/usr/bin/env python3
"""
Example 4: Batch Real Estate Analysis with PyRTex

This example demonstrates processing multiple real estate files (YAML and JSON)
as independent requests in a single batch job to extract common property features.

We'll analyze 4 different real estate properties:
- luxury_condo.yaml (San Francisco condo listing)
- suburban_house.yaml (Palo Alto family home)
- office_building.json (Commercial office space)
- apartment_complex.json (Multi-family residential complex)

Each file represents a different property and will be processed independently
to extract standardized real estate information.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from pyrtex import Job


class FileInput(BaseModel):
    """Input schema for text file processing."""
    file_path: str = Field(description="Path to the text file to process")


class RealEstateAnalysis(BaseModel):
    """Standardized real estate analysis output."""
    
    # Basic Property Info
    property_type: str = Field(description="Type of property (condo, house, office, apartment complex, etc.)")
    address: str = Field(description="Full property address")
    city: str = Field(description="City where property is located")
    state: str = Field(description="State where property is located")
    
    # Financial Information
    listing_price: int = Field(description="Listing/asking price in dollars")
    price_per_sqft: float = Field(description="Price per square foot")
    annual_property_taxes: int = Field(description="Annual property taxes in dollars")
    
    # Size and Layout
    total_square_feet: int = Field(description="Total square footage")
    bedrooms: int = Field(description="Number of bedrooms (0 if commercial)")
    bathrooms: float = Field(description="Number of bathrooms (0 if commercial)")
    
    # Property Details
    year_built: int = Field(description="Year the property was built")
    parking_spaces: int = Field(description="Number of parking spaces available")
    
    # Market Information
    days_on_market: int = Field(description="How many days the property has been listed")
    property_condition: str = Field(description="Overall condition (excellent/good/fair/needs work)")
    
    # Investment Metrics
    estimated_monthly_income: int = Field(description="Estimated monthly rental income (0 if owner-occupied)")
    investment_grade: str = Field(description="Investment attractiveness (excellent/good/fair/poor)")
    
    # Key Features (top 3 most important)
    key_feature_1: str = Field(description="Most important property feature")
    key_feature_2: str = Field(description="Second most important property feature") 
    key_feature_3: str = Field(description="Third most important property feature")
    
    # Market Assessment
    market_competitiveness: str = Field(description="How competitive this property is (very competitive/competitive/average/overpriced)")
    target_buyer_type: str = Field(description="Most likely buyer type (investor/family/professional/developer)")


def main():
    """Process real estate files to extract standardized property information."""
    
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
    
    print("🏠 Processing real estate property files...")
    print("📁 Properties to analyze:")
    for filename in files_to_process:
        print(f"   • {filename}")
    
    # Create the job
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=RealEstateAnalysis,
        prompt_template="""
You are a professional real estate analyst. Analyze the provided property data file and extract standardized real estate information.

The file contains detailed information about a real estate property. Your task is to extract and standardize the following information:

**Basic Property Information:**
- Property type (condo, house, office building, apartment complex, etc.)
- Complete address, city, and state
- Total square footage and layout details

**Financial Analysis:**
- Listing/asking price and price per square foot
- Property taxes and other carrying costs
- Estimated rental income potential

**Physical Characteristics:**
- Year built, bedrooms, bathrooms, parking
- Overall condition and key features

**Market Analysis:**
- Time on market and competitiveness
- Target buyer demographic
- Investment potential

**Property file to analyze:**
{{ text }}

Please extract concrete data points where available. If information is missing or unclear, make reasonable estimates based on property type and location. Focus on providing actionable real estate insights.
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
    print("⏳ Analyzing properties (this may take a few moments)...")
    
    try:
        results = list(job.submit().wait().results())
        
        print(f"\n✅ Batch analysis completed! Analyzed {len(results)} properties.")
        
        # Display results for each property
        print("\n" + "="*80)
        print("🏘️  REAL ESTATE PORTFOLIO ANALYSIS")
        print("="*80)
        
        # Collect all analyses for comparison
        all_properties = {}
        total_value = 0
        total_sqft = 0
        
        for result in results:
            print(f"\n🏠 Property: {result.request_key}")
            print("-" * 50)
            
            if result.was_successful:
                analysis = result.output
                all_properties[result.request_key] = analysis
                
                # Add to totals
                total_value += analysis.listing_price
                total_sqft += analysis.total_square_feet
                
                print(f"📍 Address: {analysis.address}, {analysis.city}, {analysis.state}")
                print(f"🏗️  Type: {analysis.property_type} (Built: {analysis.year_built})")
                print(f"� Size: {analysis.total_square_feet:,} sq ft")
                
                if analysis.bedrooms > 0:
                    print(f"�️  Layout: {analysis.bedrooms} bed, {analysis.bathrooms} bath")
                
                print(f"� Price: ${analysis.listing_price:,} (${analysis.price_per_sqft:.0f}/sq ft)")
                print(f"🏛️  Taxes: ${analysis.annual_property_taxes:,}/year")
                print(f"🚗 Parking: {analysis.parking_spaces} spaces")
                print(f"📅 Market: {analysis.days_on_market} days listed")
                print(f"🏥 Condition: {analysis.property_condition}")
                
                if analysis.estimated_monthly_income > 0:
                    annual_income = analysis.estimated_monthly_income * 12
                    cap_rate = (annual_income / analysis.listing_price) * 100
                    print(f"💵 Income: ${analysis.estimated_monthly_income:,}/month (${annual_income:,}/year)")
                    print(f"📊 Cap Rate: {cap_rate:.2f}%")
                
                print(f"⭐ Investment Grade: {analysis.investment_grade}")
                print(f"🎯 Target Buyer: {analysis.target_buyer_type}")
                print(f"🏆 Competitiveness: {analysis.market_competitiveness}")
                
                print("🔑 Key Features:")
                print(f"   1. {analysis.key_feature_1}")
                print(f"   2. {analysis.key_feature_2}")
                print(f"   3. {analysis.key_feature_3}")
                
                # Token usage
                if result.usage_metadata:
                    tokens = result.usage_metadata.get('totalTokenCount', 'N/A')
                    print(f"🔢 Tokens used: {tokens}")
                
            else:
                print(f"❌ Error analyzing property: {result.error}")
        
        # Portfolio summary
        if len(all_properties) > 1:
            print(f"\n" + "="*80)
            print("� PORTFOLIO SUMMARY")
            print("="*80)
            
            print(f"🏘️  Total Properties: {len(all_properties)}")
            print(f"💰 Total Portfolio Value: ${total_value:,}")
            print(f"� Total Square Footage: {total_sqft:,}")
            
            if total_sqft > 0:
                avg_price_psf = total_value / total_sqft
                print(f"📊 Average Price/SqFt: ${avg_price_psf:.0f}")
            
            # Property type distribution
            property_types = [prop.property_type for prop in all_properties.values()]
            type_counts = {ptype: property_types.count(ptype) for ptype in set(property_types)}
            print(f"🏗️  Property Types: {type_counts}")
            
            # Investment grade distribution
            investment_grades = [prop.investment_grade for prop in all_properties.values()]
            grade_counts = {grade: investment_grades.count(grade) for grade in set(investment_grades)}
            print(f"⭐ Investment Grades: {grade_counts}")
            
            # Market competitiveness
            competitiveness = [prop.market_competitiveness for prop in all_properties.values()]
            comp_counts = {comp: competitiveness.count(comp) for comp in set(competitiveness)}
            print(f"🏆 Market Competitiveness: {comp_counts}")
            
            # Calculate potential rental income
            total_monthly_income = sum(prop.estimated_monthly_income for prop in all_properties.values())
            if total_monthly_income > 0:
                annual_income = total_monthly_income * 12
                portfolio_cap_rate = (annual_income / total_value) * 100
                print(f"💵 Total Monthly Income: ${total_monthly_income:,}")
                print(f"📊 Portfolio Cap Rate: {portfolio_cap_rate:.2f}%")
        
        print(f"\n🎉 Real estate analysis complete! Processed {len(results)} independent property files.")
        print("💡 Each file was analyzed separately to extract standardized real estate metrics.")
        
    except Exception as e:
        print(f"\n❌ Error during batch processing: {e}")
        print("💡 Make sure you have set up your Google Cloud credentials and project ID.")


if __name__ == "__main__":
    main()
