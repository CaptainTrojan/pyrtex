# src/pyrtex/config.py

from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class InfrastructureConfig(BaseSettings):
    """
    Configuration for GCP resources.

    Pyrtex will use sensible defaults. Use this class only when you need to
    override the default GCS bucket or BigQuery dataset for compliance or
    billing reasons.

    Values can be set via environment variables (e.g., `GOOGLE_PROJECT_ID`).
    """

    # By using pydantic-settings, this can be automatically loaded from env vars
    model_config = SettingsConfigDict(env_prefix="PYRTEX_", extra="ignore")

    # If not provided, pyrtex will attempt to discover it from the environment.
    project_id: Optional[str] = Field(default=None)

    # If not provided, defaults to the same as project_id.
    location: Optional[str] = Field(default="us-central1")

    # If not provided, a default bucket will be created/used.
    # e.g., "pyrtex-assets-[project_id]"
    gcs_bucket_name: Optional[str] = None

    # If not provided, a default dataset will be created/used.
    # e.g., "pyrtex_results"
    bq_dataset_id: Optional[str] = None

    # Resource retention settings (in days)
    gcs_file_retention_days: int = 1
    bq_table_retention_days: int = 1

    def __init__(self, **data):
        # Load from environment variables first
        env_values = {}

        # Check for specific environment variables
        import os

        google_project_id = os.getenv("GOOGLE_PROJECT_ID")
        if google_project_id and "project_id" not in data:
            env_values["project_id"] = google_project_id

        google_location = os.getenv("GOOGLE_LOCATION")
        if google_location and "location" not in data:
            env_values["location"] = google_location

        # Merge environment values with explicit data (explicit takes precedence)
        merged_data = {**env_values, **data}
        super().__init__(**merged_data)


class GenerationConfig(BaseModel):
    """
    Configuration for the model's generation parameters.
    See the Vertex AI documentation for details on each parameter.
    """

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=2048, gt=0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, gt=0)
