Configuration Guide
==================

Environment Setup
-----------------

PyRTex requires minimal configuration for most use cases. The primary configuration is setting your Google Cloud project ID.

Basic Configuration
~~~~~~~~~~~~~~~~~~~

For simulation mode (testing without GCP costs)::

    from pyrtex import Job
    
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=YourSchema,
        prompt_template="Your prompt here",
        simulation_mode=True  # No GCP credentials needed
    )

For production use with real Vertex AI::

    # Set environment variable
    export GOOGLE_PROJECT_ID="your-project-id"
    
    # Then in your code
    job = Job(
        model="gemini-2.0-flash-lite-001", 
        output_schema=YourSchema,
        prompt_template="Your prompt here",
        simulation_mode=False  # Uses real Vertex AI
    )

File Format Support
-------------------

PyRTex supports multiple input formats through Gemini's multimodal capabilities:

Image Processing
~~~~~~~~~~~~~~~~

Supported image formats:
- PNG
- JPEG
- WebP
- GIF

Example usage::

    import base64
    
    def encode_image(image_path):
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    image_data = encode_image("path/to/image.png")
    job.add_request("image_analysis", {
        "image": f"data:image/png;base64,{image_data}"
    })

PDF Processing
~~~~~~~~~~~~~~

PDF documents are processed by converting to base64::

    def encode_pdf(pdf_path):
        with open(pdf_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    pdf_data = encode_pdf("document.pdf")
    job.add_request("pdf_extraction", {
        "document": f"data:application/pdf;base64,{pdf_data}"
    })

Text Processing
~~~~~~~~~~~~~~~

Plain text can be passed directly::

    job.add_request("text_analysis", {
        "content": "Your text content here"
    })

Advanced Configuration
----------------------

Generation Config
~~~~~~~~~~~~~~~~~

Customize the model's behavior::

    from pyrtex.config import GenerationConfig
    
    generation_config = GenerationConfig(
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        max_output_tokens=2048
    )
    
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=YourSchema,
        prompt_template="Your prompt",
        generation_config=generation_config
    )

Infrastructure Config
~~~~~~~~~~~~~~~~~~~~~

Control GCP infrastructure settings::

    from pyrtex.config import InfrastructureConfig
    
    infra_config = InfrastructureConfig(
        machine_type="n1-standard-4",
        replica_count=2,
        max_replica_count=5,
        batch_size=100
    )
    
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=YourSchema,
        prompt_template="Your prompt",
        config=infra_config
    )

Template System
---------------

PyRTex uses Jinja2 for flexible prompt templating::

    prompt_template = """
    Analyze the following {{ content_type }}:
    
    {% if content_type == "image" %}
    {{ image_data }}
    {% else %}
    {{ text_content }}
    {% endif %}
    
    Extract the following information:
    {% for field in required_fields %}
    - {{ field }}
    {% endfor %}
    """

Error Handling
--------------

Configure error handling behavior::

    # Process results with error checking
    for result in job.results():
        if result.was_successful:
            # Process successful result
            data = result.output
        else:
            # Handle error
            print(f"Error: {result.error}")
            print(f"Request: {result.request_key}")

Performance Tuning
------------------

File Size Limits
~~~~~~~~~~~~~~~~

- Images: Recommended under 5MB
- PDFs: Recommended under 10MB  
- Text: No practical limit

Batch Processing
~~~~~~~~~~~~~~~~

Process multiple files efficiently::

    # Add multiple requests
    for file_path in file_list:
        job.add_request(file_path.stem, {
            "content": process_file(file_path)
        })
    
    # Submit and wait
    job.submit().wait()
    
    # Collect all results
    results = {r.request_key: r.output for r in job.results() if r.was_successful}

Model Selection
~~~~~~~~~~~~~~~

Choose the appropriate model for your use case:

- ``gemini-2.0-flash-lite-001``: Fast, cost-effective for simple tasks
- ``gemini-1.5-pro``: More capable for complex analysis
- ``gemini-1.5-flash``: Balanced performance and cost
    
    Example::
    
        export GOOGLE_PROJECT_ID="my-project-id"

**GOOGLE_APPLICATION_CREDENTIALS** (optional)
    Path to your GCP service account key file.
    
    Example::
    
        export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

Configuration Classes
---------------------

InfrastructureConfig
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyrtex.config.InfrastructureConfig
   :members:

GenerationConfig  
~~~~~~~~~~~~~~~~

.. autoclass:: pyrtex.config.GenerationConfig
   :members:

Usage Examples
--------------

Custom infrastructure settings::

    from pyrtex.config import InfrastructureConfig
    
    config = InfrastructureConfig(
        region="us-central1",
        machine_type="n1-standard-4",
        max_replica_count=5
    )
    
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=MySchema,
        prompt_template="...",
        config=config
    )

Custom generation settings::

    from pyrtex.config import GenerationConfig
    
    gen_config = GenerationConfig(
        temperature=0.7,
        max_output_tokens=1000,
        top_p=0.8
    )
    
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=MySchema,
        prompt_template="...",
        generation_config=gen_config
    )
