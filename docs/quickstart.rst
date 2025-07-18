Quick Start Guide
================

Installation
------------

Install PyRTex from source::

    git clone https://github.com/CaptainTrojan/pyrtex.git
    cd pyrtex
    pip install -e .[dev]

Basic Usage
-----------

1. **Define your schemas**::

    from pydantic import BaseModel

    class TextInput(BaseModel):
        content: str

    class TextAnalysis(BaseModel):
        summary: str
        key_points: list[str]
        sentiment: str

2. **Create a job**::

    from pyrtex import Job

    job = Job[TextAnalysis](
        model="gemini-2.0-flash-lite-001",
        output_schema=TextAnalysis,
        prompt_template="Analyze: {{ content }}",
        simulation_mode=True  # For testing
    )

3. **Add requests**::

    # Must use Pydantic model instances, not dictionaries
    job.add_request("doc1", TextInput(content="Your text here"))
    job.add_request("doc2", TextInput(content="Another document"))

4. **Submit, wait, and get results**::

    # Method 1: Chain the calls
    for result in job.submit().wait().results():
        if result.was_successful:
            print(f"Analysis: {result.output.summary}")
        else:
            print(f"Error: {result.error}")
    
    # Method 2: Step by step
    job.submit()  # Submit to Vertex AI
    job.wait()    # Wait for completion
    
    for result in job.results():
        if result.was_successful:
            print(f"Analysis: {result.output.summary}")
        else:
            print(f"Error: {result.error}")

Workflow Notes
--------------

**Important**: You must call ``submit()`` and ``wait()`` before calling ``results()``. 

- ``submit()``: Uploads data and submits the job to Vertex AI
- ``wait()``: Blocks until the job completes
- ``results()``: Retrieves and parses the results

In simulation mode, ``submit()`` and ``wait()`` are no-ops that return immediately.

Configuration
-------------

For production use, configure your GCP project::

    export GOOGLE_PROJECT_ID="your-project-id"

Set simulation_mode=False to use real Vertex AI processing.

Examples
--------

See the ``examples/`` directory for complete working examples:

* Text extraction and analysis
* Product catalog parsing  
* Invoice data extraction
