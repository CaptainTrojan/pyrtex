# PyRTex

[![CI](https://github.com/CaptainTrojan/pyrtex/actions/workflows/ci.yml/badge.svg)](https://github.com/CaptainTrojan/pyrtex/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/CaptainTrojan/pyrtex/branch/main/graph/badge.svg)](https://codecov.io/gh/CaptainTrojan/pyrtex)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A simple Python library for batch text extraction and processing using Google Cloud Vertex AI.

PyRTex makes it easy to process multiple documents, images, or text snippets with Gemini models and get back structured, type-safe results using Pydantic models.

## ✨ Features

- **🚀 Simple API**: Just 3 steps - configure, submit, get results
- **📦 Batch Processing**: Process multiple inputs efficiently  
- **🔒 Type Safety**: Pydantic models for structured output
- **🎨 Flexible Templates**: Jinja2 templates for prompt engineering
- **☁️ GCP Integration**: Seamless Vertex AI and BigQuery integration
- **🧪 Testing Mode**: Simulate without GCP costs

## 📦 Installation

```bash
git clone https://github.com/CaptainTrojan/pyrtex.git
cd pyrtex
pip install -e .
```

For development:
```bash
pip install -e .[dev]
```

## 🚀 Quick Start

```python
from pydantic import BaseModel
from pyrtex import Job

# Define your data structures
class TextInput(BaseModel):
    content: str

class Analysis(BaseModel):
    summary: str
    sentiment: str
    key_points: list[str]

# Create a job
job = Job[Analysis](
    model="gemini-2.0-flash-lite-001",
    output_schema=Analysis,
    prompt_template="Analyze this text: {{ content }}",
    simulation_mode=True  # Set to False for real processing
)

# Add your data
job.add_request("doc1", TextInput(content="Your text here"))
job.add_request("doc2", TextInput(content="Another document"))

# Process and get results
for result in job.submit().wait().results():
    if result.was_successful:
        print(f"Summary: {result.output.summary}")
        print(f"Sentiment: {result.output.sentiment}")
    else:
        print(f"Error: {result.error}")
```

## 📋 Core Workflow

PyRTex uses a simple 3-step workflow:

### 1. Configure & Add Data
```python
job = Job[YourSchema](model="gemini-2.0-flash-lite-001", ...)
job.add_request("key1", YourModel(data="value1"))
job.add_request("key2", YourModel(data="value2"))
```

### 2. Submit & Wait  
```python
job.submit().wait()  # Can be chained
```

### 3. Get Results
```python
for result in job.results():
    if result.was_successful:
        # Use result.output (typed!)
    else:
        # Handle result.error
```

## ⚙️ Configuration

For production use, set your GCP project:

```bash
export GOOGLE_PROJECT_ID="your-project-id"
```

Then use `simulation_mode=False` for real processing.

## 📚 Examples

The `examples/` directory contains complete working examples:

```bash
cd examples

# Generate sample files
python generate_sample_data.py

# Extract contact info from business cards
python 01_simple_text_extraction.py

# Parse product catalogs  
python 02_pdf_product_parsing.py

# Extract invoice data from PDFs
python 03_image_description.py
```

### Example Use Cases

- **📇 Business Cards**: Extract contact information
- **📄 Documents**: Process PDFs, images (PNG, JPEG)  
- **🛍️ Product Catalogs**: Parse pricing and inventory
- **🧾 Invoices**: Extract structured financial data
- **📊 Batch Processing**: Handle multiple files efficiently

## 🧪 Development

### Running Tests

```bash
# All tests (mocked, safe)
./test_runner.sh

# Specific test types
./test_runner.sh --unit
./test_runner.sh --integration
./test_runner.sh --flake

# Real GCP tests (costs money!)
./test_runner.sh --real --project-id your-project-id
```

Windows users:
```cmd
test_runner.bat --unit
test_runner.bat --flake
```

### Code Quality

- **flake8**: Linting
- **black**: Code formatting  
- **isort**: Import sorting
- **pytest**: Testing with coverage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `./test_runner.sh`
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/CaptainTrojan/pyrtex/issues)
- **Examples**: Check the `examples/` directory
- **Testing**: Use `simulation_mode=True` for development
