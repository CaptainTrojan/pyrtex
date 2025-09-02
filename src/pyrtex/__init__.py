# src/pyrtex/__init__.py

"""
Pyrtex - A Python library for batch text extraction and processing using
Google Cloud Vertex AI.

This library provides a simple interface for submitting batch jobs to Google
Cloud Vertex AI for text extraction, document processing, and structured data
extraction tasks.
"""

from .client import Job
from .config import GenerationConfig, InfrastructureConfig
from .exceptions import ConfigurationError, JobFailedError
from .models import BatchResult, T

__version__ = "0.2.3"
__all__ = [
    "Job",
    "InfrastructureConfig",
    "GenerationConfig",
    "BatchResult",
    "T",
    "ConfigurationError",
    "JobFailedError",
]
