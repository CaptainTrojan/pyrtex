PyRTex Documentation
====================

PyRTex is a Python library for batch text extraction and processing using Google Cloud Vertex AI.
It simplifies the process of sending multiple text processing requests to Gemini models and 
collecting structured results.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   api
   examples
   configuration

Quick Start
-----------

Install PyRTex::

    pip install pyrtex

Basic usage::

    from pydantic import BaseModel
    from pyrtex import Job

    class MyOutput(BaseModel):
        summary: str
        sentiment: str

    job = Job[MyOutput](
        model="gemini-2.0-flash-lite-001",
        output_schema=MyOutput,
        prompt_template="Analyze this text: {{ text }}",
        simulation_mode=True
    )

    job.add_request("sample", {"text": "This is a great product!"})
    results = list(job.results())

Features
--------

* **Batch Processing**: Process multiple requests efficiently
* **Structured Output**: Use Pydantic models for type-safe results
* **GCP Integration**: Seamless integration with Vertex AI
* **Template System**: Jinja2 templates for flexible prompts
* **Error Handling**: Robust error handling and validation
* **Simulation Mode**: Test without GCP costs

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`