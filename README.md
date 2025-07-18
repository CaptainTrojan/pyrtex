# PyRTex

A Python library for batch text extraction and processing using Google Cloud Vertex AI.

PyRTex simplifies the process of sending multiple text processing requests to Gemini models and collecting structured results. It provides type-safe output parsing with Pydantic models, efficient batch processing, and seamless GCP integration.

## Features

- **Batch Processing**: Process multiple requests efficiently with Vertex AI
- **Structured Output**: Use Pydantic models for type-safe results
- **Template System**: Jinja2 templates for flexible prompt engineering
- **GCP Integration**: Seamless integration with Vertex AI and BigQuery
- **Error Handling**: Robust error handling and validation
- **Simulation Mode**: Test without incurring GCP costs

## Installation

```bash
git clone https://github.com/CaptainTrojan/pyrtex.git
cd pyrtex
pip install -e .[dev]
```

## Quick Start

```python
from pydantic import BaseModel
from pyrtex import Job

# Define input and output schemas
class TextInput(BaseModel):
    content: str

class TextAnalysis(BaseModel):
    summary: str
    sentiment: str
    key_points: list[str]

# Create a job
job = Job[TextAnalysis](
    model="gemini-2.0-flash-lite-001",
    output_schema=TextAnalysis,
    prompt_template="Analyze this text: {{ content }}",
    simulation_mode=True  # For testing without GCP costs
)

# Add requests (must be Pydantic model instances)
job.add_request("doc1", TextInput(content="Your text here"))

# Submit job, wait for completion, then get results
for result in job.submit().wait().results():
    if result.was_successful:
        print(f"Summary: {result.output.summary}")
    else:
        print(f"Error: {result.error}")
```

## Workflow

PyRTex follows a simple three-step workflow:

1. **Configure & Add Requests**: Set up your job and add data to process
2. **Submit & Wait**: Submit the job to Vertex AI and wait for completion  
3. **Retrieve Results**: Get structured results with error handling

```python
# Step 1: Configure and add requests
job = Job[YourOutputSchema](model="gemini-2.0-flash-lite-001", ...)
job.add_request("key1", YourInputModel(data="value1"))
job.add_request("key2", YourInputModel(data="value2"))

# Step 2: Submit and wait (can be chained)
job.submit().wait()

# Step 3: Get results
for result in job.results():
    if result.was_successful:
        # Process result.output
    else:
        # Handle result.error
```

**Important**: 
- `add_request()` requires Pydantic model instances, not dictionaries
- You must call `submit()` and `wait()` before calling `results()`  
- The methods can be chained for convenience: `job.submit().wait().results()`

## Configuration

For production use, set your GCP project ID:

```bash
export GOOGLE_PROJECT_ID="your-project-id"
```

Then set `simulation_mode=False` to use real Vertex AI processing.

## Examples

The `examples/` directory contains complete working examples that demonstrate PDF and image processing:

### Running Examples

Generate sample files and run examples:

```bash
cd examples

# Generate sample PDF and image files
python generate_sample_data.py

# Example 1: Extract contact info from business card image
python 01_simple_text_extraction.py

# Example 2: Parse products from catalog image  
python 02_pdf_product_parsing.py

# Example 3: Extract structured data from PDF invoice
python 03_image_description.py
```

### Example Use Cases

- **Business Card Processing**: Extract contact information from business card images
- **Product Catalog Analysis**: Parse product details, pricing, and inventory from catalog images
- **Invoice Data Extraction**: Extract structured financial data from PDF invoices
- **Document Processing**: Handle various file formats (PDF, PNG, JPEG) with multimodal AI
- **Batch Processing**: Process multiple documents efficiently

Each example includes sample data files and demonstrates different aspects of PyRTex functionality.

## Documentation

Generate and view the documentation:

```bash
# Install documentation dependencies
pip install -e .[dev]

# Build docs
cd docs
make html

# View docs
open _build/html/index.html
```

## Development

### Running Tests

```bash
# Run all mocked tests (default)
./test_runner.sh

# Run unit tests only
./test_runner.sh --unit

# Run integration tests (mocked)
./test_runner.sh --integration

# Run linting
./test_runner.sh --flake

# Run linting with auto-fix
./test_runner.sh --flake-fix

# For real GCP tests (incurs costs)
./test_runner.sh --real --project-id your-project-id
```

### Code Quality

This project uses:
- **flake8** for linting
- **black** for code formatting  
- **isort** for import sorting
- **pytest** for testing with 100% coverage

## License

MIT License - see [LICENSE](LICENSE) file for details.
