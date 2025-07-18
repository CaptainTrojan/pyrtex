# src/pyrtex/models.py

from typing import Any, Dict, Generic, Hashable, Optional, TypeVar

from pydantic import BaseModel

# This defines a generic type that can be any Pydantic model.
# When a user provides their `ProductInfo` model, T becomes `ProductInfo`.
T = TypeVar("T", bound=BaseModel)


class BatchResult(BaseModel, Generic[T]):
    """
    A container for the result of a single request in a batch job.
    """

    # The key the user provided when calling `add_request`.
    request_key: Hashable

    # The parsed output, typed as the user's Pydantic schema.
    # This will be `None` if parsing failed or the model returned an error.
    output: Optional[T] = None

    # Raw JSON response from the model, for debugging.
    raw_response: Optional[Dict[str, Any]] = None

    # Usage metadata from Vertex AI.
    usage_metadata: Optional[Dict[str, Any]] = None

    # A string describing any error that occurred for this specific request.
    error: Optional[str] = None

    @property
    def was_successful(self) -> bool:
        """Returns True if the request succeeded and output was parsed."""
        return self.error is None and self.output is not None
