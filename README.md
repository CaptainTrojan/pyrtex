# PyRTex

[![CI](https://github.com/CaptainTrojan/pyrtex/actions/workflows/ci.yml/badge.svg)](https://github.com/CaptainTrojan/pyrtex/actions/workflows/ci.yml)
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
- **🔄 Async-Friendly**: Serialize job state & reconnect later (multi-process)

## 📦 Installation

Install from PyPI (recommended):
```bash
pip install pyrtex
```

Or install from source:
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
job = Job(
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

Check out the [examples](https://github.com/CaptainTrojan/pyrtex/tree/main/examples) directory for diverse, concrete usage examples.

## 📋 Core Workflow

PyRTex uses a simple 3-step workflow:

### 1. Configure & Add Data
```python
job = Job(model="gemini-2.0-flash-lite-001", ...)
job.add_request("key1", YourModel(data="value1"))
job.add_request("key2", YourModel(data="value2"))
```

### 2. Submit & Wait  
Can be chained (for synchronous processing)
```python
job.submit().wait()
```
Or separated (classic blocking vs explicit wait):
```python
job.submit()
# do other work
job.wait()
```

### 2b. Asynchronous / Multi-Process Pattern
You can avoid blocking entirely by serializing job state after submission and reconnecting later (different process / machine / scheduled task):
```python
# Process A (submitter)
job.submit()
state_json = job.serialize()
# persist state_json somewhere durable (DB, GCS, S3, queue message)

# Process B (poller / checker)
re_job = Job.reconnect_from_state(state_json)
if not re_job.is_done:
    # job.status can be inspected, then exit / reschedule
    print("Still running:", re_job.status)

# Process C (collector) – run after poller detects completion
re_job = Job.reconnect_from_state(state_json)
if re_job.is_done:
    for r in re_job.results():
        if r.was_successful:
            print(r.request_key, r.output)
```
Why serialize? The serialized state contains:
- Vertex AI batch job resource name
- InfrastructureConfig (project/location/bucket/dataset)
- Session ID (for tracing)
- Instance map (request key ↔ internal instance id ↔ output schema type)
This allows precise result parsing without retaining the original Job object in memory.

See `examples/09_async_reconnect.py` for a CLI demonstration (start / status / results commands).

### 3. Get Results
```python
for result in job.results():
    if result.was_successful:
        # Use result.output (typed!)
    else:
        # Handle result.error
```

## 🔐 Authentication

PyRTex supports multiple authentication methods for Google Cloud Platform. Choose the method that best fits your deployment environment:

### Method 1: Service Account JSON String (Recommended for Production)

Perfect for serverless deployments (AWS Lambda, Google Cloud Functions, etc.):

```python
import json
import os
from pyrtex.config import InfrastructureConfig

# Set via environment variable (most secure)
os.environ["PYRTEX_SERVICE_ACCOUNT_KEY_JSON"] = json.dumps({
    "type": "service_account",
    "project_id": "your-project-id",
    "private_key_id": "key-id",
    "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
    "client_email": "your-service@your-project.iam.gserviceaccount.com",
    "client_id": "123456789",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/your-service%40your-project.iam.gserviceaccount.com"
})

# PyRTex will automatically detect and use the JSON string
job = Job(
    model="gemini-2.0-flash-lite-001",
    output_schema=YourSchema,
    prompt_template="Your prompt"
)
```

Or configure directly:
```python
config = InfrastructureConfig(
    service_account_key_json=json.dumps(service_account_dict)
)
job = Job(..., config=config)
```

### Method 2: Service Account File Path

For traditional server deployments:

```python
import os

# Set via environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/service-account.json"

# PyRTex will automatically detect and use the file
job = Job(
    model="gemini-2.0-flash-lite-001",
    output_schema=YourSchema,
    prompt_template="Your prompt"
)
```

Or configure directly:
```python
config = InfrastructureConfig(
    service_account_key_path="/path/to/service-account.json"
)
job = Job(..., config=config)
```

### Method 3: Application Default Credentials (Development)

For local development and testing:

```bash
# One-time setup
gcloud auth application-default login
```

```python
# No additional configuration needed
job = Job(
    model="gemini-2.0-flash-lite-001",
    output_schema=YourSchema,
    prompt_template="Your prompt"
)
```

### Authentication Priority

PyRTex uses the following priority order:
1. **Service Account JSON string** (`PYRTEX_SERVICE_ACCOUNT_KEY_JSON` or `service_account_key_json`)
2. **Service Account file** (`GOOGLE_APPLICATION_CREDENTIALS` or `service_account_key_path`)  
3. **Application Default Credentials** (gcloud login)

### Required GCP Permissions

When creating a service account for PyRTex, assign these IAM roles:

**Required Roles:**
- **`roles/aiplatform.user`** - Vertex AI access for batch processing
- **`roles/storage.objectAdmin`** - GCS bucket read/write access
- **`roles/bigquery.dataEditor`** - BigQuery dataset read/write access
- **`roles/bigquery.jobUser`** - BigQuery job execution

**Alternative (More Restrictive):**
If you prefer granular permissions, create a custom role with:
```json
{
  "permissions": [
    "aiplatform.batchPredictionJobs.create",
    "aiplatform.batchPredictionJobs.get",
    "aiplatform.batchPredictionJobs.list",
    "aiplatform.models.predict",
    "storage.objects.create",
    "storage.objects.delete", 
    "storage.objects.get",
    "storage.objects.list",
    "bigquery.datasets.create",
    "bigquery.tables.create",
    "bigquery.tables.get",
    "bigquery.tables.getData",
    "bigquery.tables.updateData",
    "bigquery.jobs.create"
  ]
}
```

**Setup via gcloud CLI:**
```bash
# Create service account
gcloud iam service-accounts create pyrtex-service \
    --display-name="PyRTex Service Account"

# Assign roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:pyrtex-service@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:pyrtex-service@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:pyrtex-service@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:pyrtex-service@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/bigquery.jobUser"

# Create and download key
gcloud iam service-accounts keys create pyrtex-key.json \
    --iam-account=pyrtex-service@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

## ⚙️ Configuration

### InfrastructureConfig

Configure GCP resources and authentication:

```python
from pyrtex.config import InfrastructureConfig

config = InfrastructureConfig(
    # Required (set one of these)
    project_id="your-gcp-project-id",                    # GCP Project ID
    
    # Authentication (optional - detected automatically)
    service_account_key_json='{"type": "service_account", ...}',  # JSON string
    service_account_key_path="/path/to/service-account.json",     # File path
    
    # GCP Resources (optional - sensible defaults)
    location="us-central1",                              # Vertex AI location
    gcs_bucket_name="pyrtex-assets-your-project",        # GCS bucket for files
    bq_dataset_id="pyrtex_results",                      # BigQuery dataset
    
    # Data Retention (optional)
    gcs_file_retention_days=1,                           # GCS file cleanup (1-365)
    bq_table_retention_days=1                            # BigQuery table cleanup (1-365)
)

job = Job(..., config=config)
```

**Environment Variables:**
- `GOOGLE_PROJECT_ID` or `PYRTEX_PROJECT_ID` → `project_id`
- `GOOGLE_LOCATION` → `location` 
- `PYRTEX_GCS_BUCKET_NAME` → `gcs_bucket_name`
- `PYRTEX_BQ_DATASET_ID` → `bq_dataset_id`
- `PYRTEX_SERVICE_ACCOUNT_KEY_JSON` → `service_account_key_json`
- `GOOGLE_APPLICATION_CREDENTIALS` → `service_account_key_path`

### GenerationConfig

Fine-tune Gemini model behavior:

```python
from pyrtex.config import GenerationConfig

generation_config = GenerationConfig(
    temperature=0.7,        # Creativity level (0.0-2.0)
    max_output_tokens=2048, # Maximum response length (1-8192)
    top_p=0.95,            # Nucleus sampling (0.0-1.0)
    top_k=40               # Top-k sampling (1-40)
)

job = Job(
    model="gemini-2.0-flash-lite-001",
    output_schema=YourSchema,
    prompt_template="Your prompt",
    generation_config=generation_config
)
```

**Parameters:**
- **`temperature`** (0.0-2.0): Controls randomness. Lower = more focused, higher = more creative
- **`max_output_tokens`** (1-8192): Maximum tokens in response. Adjust based on expected output length
- **`top_p`** (0.0-1.0): Nucleus sampling. Consider tokens with cumulative probability up to top_p
- **`top_k`** (1-40): Top-k sampling. Consider only the k most likely tokens

**Quick Configs:**
```python
# Conservative (factual, consistent)
GenerationConfig(temperature=0.1, top_p=0.8, top_k=10)

# Balanced (default)
GenerationConfig(temperature=0.0, max_output_tokens=2048)

# Creative (diverse, experimental)  
GenerationConfig(temperature=1.2, top_p=0.95, top_k=40)
```

## 🎯 Usage Patterns

### Synchronous (Chained)
```python
for r in job.submit().wait().results():
    ...
```

### Non-Blocking (Explicit Wait)
```python
job.submit()
# do other tasks
job.wait()
for r in job.results():
    ...
```

### Async / Distributed (Serialize + Reconnect)
```python
# Submit phase
a_job = Job(...)
a_job.add_request("k1", Input(...))
a_job.add_request("k2", Input(...))
a_job.submit()
state_json = a_job.serialize()
# persist state_json externally

# Later (polling)
re_job = Job.reconnect_from_state(state_json)
if not re_job.is_done:
    print("Still running")

# Or even waiting (blocking)
re_job.wait()

# Later (collection)
re_job = Job.reconnect_from_state(state_json)
if re_job.is_done:
    for r in re_job.results():
        ...
```

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

### Common Issues

**"Project was not passed and could not be determined from the environment"**
- Solution: Set `GOOGLE_PROJECT_ID` environment variable or use `simulation_mode=True`

**"Failed to initialize GCP clients"**  
- Solution: Run `gcloud auth application-default login` or use simulation mode

**Examples not working**
- Solution: Run `python generate_sample_data.py` first to create sample files
