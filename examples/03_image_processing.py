#!/usr/bin/env python3
"""
Example 3: Image Processing

Demonstrates extracting structured data from images.
"""

from pathlib import Path
from pydantic import BaseModel
from pyrtex import Job


class Product(BaseModel):
    """Product information extracted from image."""
    name: str
    price: float
    description: str


class ProductList(BaseModel):
    """List of products from catalog image."""
    products: list[Product]
    total_count: int


class ImageInput(BaseModel):
    """Input schema for image processing."""
    image: Path


def main():
    data_dir = Path(__file__).parent / "data"
    image_path = data_dir / "product_catalog.png"
    
    if not image_path.exists():
        print("Sample image not found. Run generate_sample_data.py first.")
        return
    
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=ProductList,
        prompt_template="Extract all products from this catalog image: {{ image }}",
    )
    
    job.add_request("catalog", ImageInput(image=image_path))
    
    for result in job.submit().wait().results():
        if result.was_successful:
            catalog = result.output
            print(f"Found {catalog.total_count} products:")
            for product in catalog.products:
                print(f"- {product.name}: ${product.price}")


if __name__ == "__main__":
    main()
