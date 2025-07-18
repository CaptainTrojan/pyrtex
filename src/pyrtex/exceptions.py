# src/pyrtex/exceptions.py

"""
Custom exceptions for the Pyrtex library.
"""


class PyrtexError(Exception):
    """Base exception for all Pyrtex errors."""

    pass


class ConfigurationError(PyrtexError):
    """
    Raised when there is an issue with the configuration.

    This includes problems with:
    - GCP authentication
    - Project ID discovery
    - Missing required configuration values
    """

    pass


class JobFailedError(PyrtexError):
    """
    Raised when a Vertex AI batch job fails.

    This indicates that the job was submitted successfully but failed during execution.
    """

    pass


class ValidationError(PyrtexError):
    """
    Raised when there is an issue with data validation.

    This includes problems with:
    - Invalid Pydantic models
    - Schema validation failures
    - Malformed input data
    """

    pass


class ResultParsingError(PyrtexError):
    """
    Raised when there is an issue parsing results from Vertex AI.

    This includes problems with:
    - Malformed JSON responses
    - Missing expected fields
    - Invalid function call responses
    """

    pass
