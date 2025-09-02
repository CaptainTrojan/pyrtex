# src/pyrtex/config.py

from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class InfrastructureConfig(BaseSettings):
    """
    Configuration for GCP resources and authentication.

    Pyrtex will use sensible defaults. Use this class only when you need to
    override the default GCS bucket or BigQuery dataset for compliance or
    billing reasons, or when you need to specify custom authentication.

    Authentication Options (in order of precedence):
    1. service_account_key_json: JSON string of service account key
    2. service_account_key_path: Path to service account key file
    3. Application Default Credentials (gcloud auth)

    Values can be set via environment variables (e.g., `GOOGLE_PROJECT_ID`,
    `PYRTEX_SERVICE_ACCOUNT_KEY_JSON`, `GOOGLE_APPLICATION_CREDENTIALS`).
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

    # Authentication options (new)
    # Path to service account key file (set via GOOGLE_APPLICATION_CREDENTIALS)
    service_account_key_path: Optional[str] = Field(default=None)
    # Service account key as JSON string (set via PYRTEX_SERVICE_ACCOUNT_KEY_JSON)
    service_account_key_json: Optional[str] = Field(default=None)

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

        # Check for service account authentication
        service_account_key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if (
            service_account_key_path
            and "service_account_key_path" not in data
            and self._is_service_account_file(service_account_key_path)
        ):
            env_values["service_account_key_path"] = service_account_key_path

        service_account_key_json = os.getenv("PYRTEX_SERVICE_ACCOUNT_KEY_JSON")
        if service_account_key_json and "service_account_key_json" not in data:
            env_values["service_account_key_json"] = service_account_key_json

        # Merge environment values with explicit data (explicit takes precedence)
        merged_data = {**env_values, **data}
        super().__init__(**merged_data)

    def _is_service_account_file(self, file_path: str) -> bool:
        """Check if a file is a service account key file (not user ADC file)."""
        try:
            import json

            with open(file_path, "r") as f:
                data = json.load(f)

            # Service account files have these fields, user ADC files don't
            required_fields = {"type", "client_email", "private_key", "token_uri"}
            return (
                required_fields.issubset(data.keys())
                and data.get("type") == "service_account"
            )
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            return False


class ThinkingConfig(BaseModel):
    """
    Configuration for the model's thinking parameters.
    """

    thinking_budget: int = Field(default=0, ge=-1)  # By default, thinking is off


class GenerationConfig(BaseModel):
    """
    Configuration for the model's generation parameters.
    See the Vertex AI documentation for details on each parameter.
    """

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: Optional[int] = Field(default=None, gt=0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, gt=0)
    thinking_config: ThinkingConfig = Field(default_factory=ThinkingConfig)
