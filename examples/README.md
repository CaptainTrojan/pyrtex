# PyRTex Examples

Simple, clear examples demonstrating PyRTex functionality.

## Quick Start

1. Install dependencies:
```bash
pip install -e .[dev]
```

2. Generate sample data:
```bash
python generate_sample_data.py
```

3. Run any example:
```bash
python 01_basic_text_extraction.py
```

## Examples

### 1. Basic Text Extraction (`01_basic_text_extraction.py`)
Fundamental PyRTex workflow: extract structured data from text input.
- Simple text → structured output
- Basic Pydantic schema
- Core submit → wait → results pattern

### 2. Enum Constraints (`02_enum_constraints.py`)
Using enums to constrain model outputs to specific valid values.
- String enums for categorical data
- Validation and type safety
- Consistent output formatting

### 3. Image Processing (`03_image_processing.py`)
Extracting structured data from images.
- Image file input handling
- Vision model capabilities
- Product catalog parsing

### 4. PDF Processing (`04_pdf_processing.py`)
Processing PDF documents for data extraction.
- PDF file handling
- Document parsing
- Invoice data extraction

### 5. Batch Processing (`05_batch_processing.py`)
Processing multiple inputs in a single batch job.
- Multiple file processing
- Efficient batch operations
- Request key management

### 6. Multi-File Input (`06_multi_file_input.py`)
Processing multiple files together in a single request.
- Combining different files per request
- Cross-file analysis
- Property + business card matching

### 7. Multi-Format Processing (`07_multi_format_processing.py`)
Processing different file types (images, PDFs, text, audio, video) in one batch.
- Mixed file type support
- Images, PDFs, YAML, JSON, audio, video
- Unified data extraction

## Key Concepts

**Basic Workflow:**
```python
# 1. Create job
job = Job(
    model="gemini-2.0-flash-lite-001",
    output_schema=YourSchema,
    prompt_template="Your prompt: {{ input_field }}",
    simulation_mode=True  # For testing
)

# 2. Add requests
job.add_request("key", InputData(field="value"))

# 3. Process
for result in job.submit().wait().results():
    if result.was_successful:
        print(result.output)
```

**Input Types:**
- Text strings
- File paths (Path objects or strings)
- Images (PNG, JPEG, WebP)
- PDFs
- Audio files (WAV, MP3, etc.)
- Video files (MP4, etc.)
- Text files (YAML, JSON, etc.)
- Multiple files per request
- Mixed file types in batch
- Any Pydantic model

**Output Schemas:**
- Use Pydantic models to define structure
- Support enums for constrained values
- Nested models and lists work
- Type validation included

## Simulation Mode

All examples use `simulation_mode=True` by default, which:
- Returns dummy data instantly
- Doesn't require GCP credentials
- Perfect for development and testing

To use real processing, set `simulation_mode=False` and configure GCP credentials.

## File Generation

Run `generate_sample_data.py` to create:
- Sample images (business cards, product catalogs)
- Sample PDFs (invoices)
- Sample text files (property data)

These files are used by the examples and demonstrate realistic use cases.
