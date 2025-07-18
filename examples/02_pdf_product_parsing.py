#!/usr/bin/env python3
"""
Example 2: Product Catalog Image Parsing

This example demonstrates parsing product information from a catalog image.
It extracts structured product data and organizes it using Pydantic models.
"""

from pathlib import Path
from pydantic import BaseModel
from pyrtex import Job

# Define the input schema for image data
class CatalogInput(BaseModel):
    image: Path  # Catalog image file path
    extraction_focus: str  # What specific information to extract
    analysis_type: str  # Type of analysis to perform

# Define the output schema for product information
class Product(BaseModel):
    name: str
    sku: str
    price: float
    features: list[str]
    stock_quantity: int

class ProductCatalog(BaseModel):
    products: list[Product]
    total_products: int
    contact_email: str
    contact_phone: str

def main():
    # Set up the job
    job = Job[ProductCatalog](
        model="gemini-2.0-flash-lite-001",
        output_schema=ProductCatalog,
        prompt_template="""
        Parse the product catalog from this image and extract {{ analysis_type }} information:
        
        Focus on: {{ extraction_focus }}
        
        Please extract:
        - All products with their names, SKUs, prices, features, and stock quantities
        - Parse prices correctly (remove $ and convert to float)
        - Extract features as a list from each product
        - Get contact information from the bottom
        - Count total number of products
        
        Be thorough and extract all visible product details.
        """
    )
    
    # Read product catalog image
    data_dir = Path(__file__).parent / "data"
    catalog_path = data_dir / "product_catalog.png"
    
    if not catalog_path.exists():
        print(f"Image file not found: {catalog_path}")
        print("Please run 'python generate_sample_data.py' first to create sample files.")
        return
    
    # Add the parsing request using Path object directly
    job.add_request("catalog_parsing", CatalogInput(
        image=catalog_path,  # Pass Path object directly
        analysis_type="detailed",
        extraction_focus="product specifications, pricing accuracy, and comprehensive inventory data"
    ))
    
    # Process and get results
    print("Parsing product catalog from image...")
    print(f"Processing: {catalog_path}")
    
    # Submit job, wait for completion, then get results
    results = list(job.submit().wait().results())
    
    # Display results
    for result in results:
        if result.was_successful:
            catalog = result.output
            print(f"\n--- Product Catalog Analysis ---")
            print(f"Total Products Found: {catalog.total_products}")
            print(f"Contact Email: {catalog.contact_email}")
            print(f"Contact Phone: {catalog.contact_phone}")
            print("\nProducts:")
            
            total_value = 0
            for i, product in enumerate(catalog.products, 1):
                print(f"\n  {i}. {product.name} ({product.sku})")
                print(f"     Price: ${product.price:.2f}")
                print(f"     Stock: {product.stock_quantity} units")
                print(f"     Features:")
                for feature in product.features:
                    print(f"       • {feature}")
                total_value += product.price * product.stock_quantity
            
            print(f"\n--- Catalog Statistics ---")
            print(f"Total inventory value: ${total_value:,.2f}")
            print(f"Average price: ${sum(p.price for p in catalog.products) / len(catalog.products):.2f}")
        else:
            print(f"Error: {result.error}")
    
    print(f"\n--- Processing Stats ---")
    print(f"Input: Product catalog image ({catalog_path.name})")
    print(f"Tokens used: {results[0].usage_metadata.get('totalTokenCount', 'N/A') if results else 'N/A'}")

if __name__ == "__main__":
    main()
