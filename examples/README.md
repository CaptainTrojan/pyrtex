# PyRTex Examples

This directory contains example scripts demonstrating various use cases for PyRTex with real PDF and image processing.

## Running the Examples

All examples can be run in simulation mode (no GCP credentials required) or with real Vertex AI processing.

### Prerequisites

1. Install pyrtex with dependencies:
```bash
pip install -e .[dev]
```

2. Generate sample files:
```bash
cd examples
python generate_sample_data.py
```

3. For real GCP processing (optional):
```bash
export GOOGLE_PROJECT_ID="your-project-id"
# Ensure you have GCP credentials configured
```

### Example 1: Business Card Text Extraction (Image Processing)

Demonstrates extracting contact information from a business card image using computer vision.

```bash
cd examples
python 01_simple_text_extraction.py
```

**What it does:**
- Processes a business card image (`data/business_card.png`)
- Extracts contact details (name, title, company, email, phone, address)
- Identifies services offered by the company
- Uses Gemini's vision capabilities for image-to-text extraction
- Structures output with detailed Pydantic models
- Follows the submit → wait → results workflow

### Example 2: Product Catalog Parsing (Image Processing)

Shows how to parse product information from a catalog image with multiple products.

```bash
cd examples
python 02_pdf_product_parsing.py
```

**What it does:**
- Analyzes product catalog image (`data/product_catalog.png`)
- Extracts product details (name, SKU, price, features, stock)
- Parses contact information from the catalog
- Calculates inventory statistics and insights
- Demonstrates structured data extraction from complex images

### Example 3: PDF Invoice Data Extraction (Document Processing)

Demonstrates extracting comprehensive structured data from PDF invoices.

```bash
cd examples
python 03_image_description.py
```

**What it does:**
- Processes PDF invoice (`data/sample_invoice.pdf`)
- Extracts company and customer information
- Parses line items with quantities, prices, and totals
- Calculates financial summaries and tax information
- Shows PDF document processing capabilities

## Sample Data

The `data/` directory contains realistic sample files:

- `sample_invoice.pdf` - Professional invoice with multiple line items, tax calculations, and company details
- `product_catalog.png` - Product catalog image with 4 products, pricing, and contact information
- `business_card.png` - Business card with contact details and service offerings

### Generating Fresh Sample Data

To regenerate the sample files with different content:

```bash
cd examples
python generate_sample_data.py
```

This script creates:
- PDF invoices using ReportLab
- Product catalog images using Pillow
- Business card images with contact information

## File Format Support

PyRTex supports various input formats through Gemini's multimodal capabilities:

- **Images**: PNG, JPEG, WebP, GIF
- **Documents**: PDF (converted to base64)
- **Text**: Plain text, structured formats

## Switching to Real Processing

To use real Vertex AI processing instead of simulation mode:

1. Set your GCP project ID:
```bash
export GOOGLE_PROJECT_ID="your-project-id"
```

2. Modify the example scripts by changing:
```python
simulation_mode=True  # Change to False
```

3. Ensure your GCP credentials are properly configured for Vertex AI access.

## PyRTex Workflow

All examples follow the same three-step pattern:

```python
# 1. Create job and add requests
job = Job[OutputSchema](model="...", output_schema=OutputSchema, ...)
job.add_request("key", data)

# 2. Submit and wait for completion
job.submit().wait()

# 3. Get results
for result in job.results():
    if result.was_successful:
        # Process result.output
    else:
        # Handle result.error
```

The key point is that `results()` can only be called after `submit()` and `wait()` have completed. In simulation mode, these operations are instant.

## Output Schema Customization

Each example uses Pydantic models to define the expected output structure:

- **ContactInfo**: Business card data extraction
- **ProductCatalog**: Product information with inventory details  
- **InvoiceData**: Comprehensive invoice parsing with financial calculations

You can modify these schemas to extract different information or change the data format according to your needs.

## Performance Notes

- **Simulation mode**: Instant results with dummy data
- **Real processing**: Typically 10-30 seconds per request depending on file size
- **Token usage**: Images and PDFs consume more tokens than plain text
- **File size limits**: PDFs should be under 10MB for optimal performance
