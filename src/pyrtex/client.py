# src/pyrtex/client.py

import json
import logging
import mimetypes
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Generic, Hashable, Iterator, List, Optional, Type, Union

import google.cloud.aiplatform as aiplatform
import google.cloud.bigquery as bigquery
import google.cloud.storage as storage
import jinja2
from google.api_core.exceptions import NotFound
from google.cloud.aiplatform_v1.types import JobState
from pydantic import BaseModel

from .config import GenerationConfig, InfrastructureConfig
from .exceptions import ConfigurationError
from .models import BatchResult, T

logger = logging.getLogger(__name__)


class Job(Generic[T]):
    """
    Manages the configuration, submission, and result retrieval for a
    Vertex AI Batch Prediction Job.

    The generic type parameter T should match the output_schema for type safety:

    Example:
        job = Job[ContactInfo](
            model="gemini-2.0-flash-lite-001",
            output_schema=ContactInfo,  # Must match the generic type T
            prompt_template="Extract contact info: {{ content }}"
        )

    The generic type T provides static type checking for the results,
    while output_schema is used at runtime for validation and schema generation.

    Warning: This class is not thread-safe. Do not share Job instances
    across multiple threads without proper synchronization.
    """

    def __init__(
        self,
        model: str,
        output_schema: Type[T],
        prompt_template: str,
        generation_config: Optional[GenerationConfig] = None,
        config: Optional[InfrastructureConfig] = None,
        simulation_mode: bool = False,
    ):
        self.model = model
        self.output_schema = output_schema
        self.prompt_template = prompt_template
        self.generation_config = generation_config or GenerationConfig()
        self.config = config or InfrastructureConfig()
        self.simulation_mode = simulation_mode

        # Validate schema for problematic enum values
        self._validate_enum_values()

        self._session_id: str = uuid.uuid4().hex[:10]
        self._requests: List[tuple[Hashable, BaseModel]] = []
        self._instance_map: Dict[str, Hashable] = {}
        self._batch_job: Optional[aiplatform.BatchPredictionJob] = None

        self._jinja_env = jinja2.Environment()
        self._initialize_gcp()

    def _initialize_gcp(self):
        """Initializes GCP clients and resolves final configuration."""
        try:
            self._storage_client = storage.Client(project=self.config.project_id)
            self._bigquery_client = bigquery.Client(project=self.config.project_id)
            aiplatform.init(
                project=self.config.project_id, location=self.config.location
            )
            self._resolve_infra_config()
            project_id = self.config.project_id
            location = self.config.location
            logger.info(
                f"Pyrtex initialized for project '{project_id}' in '{location}'."
            )
        except Exception as e:
            msg1 = "Failed to initialize GCP clients. "
            msg2 = "Please ensure you are authenticated. "
            msg3 = "Run 'gcloud auth application-default login' in your terminal. "
            raise ConfigurationError(msg1 + msg2 + msg3 + f"Original error: {e}") from e

    def _resolve_infra_config(self):
        """Fills in missing infrastructure config values with sensible defaults."""
        if not self.config.project_id:
            self.config.project_id = self._storage_client.project
            if not self.config.project_id:
                msg1 = "Could not automatically discover GCP Project ID. "
                msg2 = "Please set the GOOGLE_PROJECT_ID environment variable "
                msg3 = "or pass it in InfrastructureConfig."
                raise ConfigurationError(msg1 + msg2 + msg3)
        if not self.config.gcs_bucket_name:
            project_id = self.config.project_id
            self.config.gcs_bucket_name = f"pyrtex-assets-{project_id}"
            bucket_name = self.config.gcs_bucket_name
            logger.info(f"GCS bucket not specified, using default: '{bucket_name}'")
        if not self.config.bq_dataset_id:
            self.config.bq_dataset_id = "pyrtex_results"
            dataset_id = self.config.bq_dataset_id
            logger.info(
                f"BigQuery dataset not specified, using default: '{dataset_id}'"
            )

    def _setup_cloud_resources(self):
        """
        Ensures the GCS bucket and BigQuery dataset exist and are configured
        correctly.
        """
        logger.info("Verifying and setting up cloud resources...")
        try:
            bucket = self._storage_client.get_bucket(self.config.gcs_bucket_name)
        except NotFound:
            bucket_name = self.config.gcs_bucket_name
            location = self.config.location
            logger.info(f"Creating GCS bucket '{bucket_name}' in {location}...")
            bucket = self._storage_client.create_bucket(
                self.config.gcs_bucket_name, location=self.config.location
            )
        bucket.clear_lifecyle_rules()
        bucket.add_lifecycle_delete_rule(age=self.config.gcs_file_retention_days)
        bucket.patch()
        logger.info("GCS bucket is ready.")
        dataset_id_full = f"{self.config.project_id}.{self.config.bq_dataset_id}"
        try:
            dataset = self._bigquery_client.get_dataset(self.config.bq_dataset_id)
        except NotFound:
            location = self.config.location
            logger.info(
                f"Creating BigQuery dataset '{dataset_id_full}' in {location}..."
            )
            dataset_ref = bigquery.Dataset(dataset_id_full)
            dataset_ref.location = self.config.location
            dataset = self._bigquery_client.create_dataset(dataset_ref)
        dataset.default_table_expiration_ms = (
            self.config.bq_table_retention_days * 24 * 60 * 60 * 1000
        )
        self._bigquery_client.update_dataset(dataset, ["default_table_expiration_ms"])
        logger.info("BigQuery dataset is ready.")

    def add_request(self, request_key: Hashable, data: BaseModel) -> "Job[T]":
        """Adds a single, structured request to the batch."""
        if self._batch_job is not None:
            raise RuntimeError("Cannot add requests after job has been submitted.")

        # Check for duplicate request keys
        existing_keys = {key for key, _ in self._requests}
        if request_key in existing_keys:
            msg = f"Request key '{request_key}' already exists. "
            msg += "Use a unique key for each request."
            raise ValueError(msg)

        self._requests.append((request_key, data))
        return self

    def _upload_file_to_gcs(
        self, source: Union[str, bytes, Path], gcs_path: str
    ) -> tuple[str, str]:
        """Uploads a local file or bytes to GCS and returns its URI and mime type."""
        bucket = self._storage_client.bucket(self.config.gcs_bucket_name)
        blob = bucket.blob(gcs_path)

        # Improved MIME type detection with only Gemini-supported types
        if isinstance(source, bytes):
            # For bytes, we can't detect extension, so default to text/plain
            mime_type = "text/plain"
        else:
            source_path = Path(source)
            ext = source_path.suffix.lower()

            # Map file extensions to Gemini-supported MIME types only
            # Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini
            gemini_supported_types = {
                # Text files - all map to text/plain
                ".txt": "text/plain",
                ".yaml": "text/plain",
                ".yml": "text/plain",
                ".json": "text/plain",  # JSON files are text, not application/json
                ".xml": "text/plain",
                ".csv": "text/plain",
                ".tsv": "text/plain",
                ".md": "text/plain",
                ".rst": "text/plain",
                ".log": "text/plain",
                ".ini": "text/plain",
                ".cfg": "text/plain",
                ".conf": "text/plain",
                ".py": "text/plain",
                ".js": "text/plain",
                ".css": "text/plain",
                ".html": "text/plain",
                ".htm": "text/plain",
                ".sql": "text/plain",
                ".sh": "text/plain",
                ".bat": "text/plain",
                ".ps1": "text/plain",
                # PDF files
                ".pdf": "application/pdf",
                # Image files
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                # Audio files
                ".mp3": "audio/mp3",
                ".mpeg": "audio/mpeg",
                ".wav": "audio/wav",
                # Video files
                ".mov": "video/mov",
                ".mp4": "video/mp4",
                ".mpeg": "video/mpeg",
                ".mpg": "video/mpg",
                ".avi": "video/avi",
                ".wmv": "video/wmv",
                ".flv": "video/flv",
            }

            # Use our mapping if available, otherwise default to text/plain for unknown extensions
            mime_type = gemini_supported_types.get(ext, "text/plain")

        if isinstance(source, bytes):
            blob.upload_from_string(source, content_type=mime_type)
        else:
            blob.upload_from_filename(str(source), content_type=mime_type)

        return f"gs://{self.config.gcs_bucket_name}/{gcs_path}", mime_type

    def _get_flattened_schema(self) -> dict:
        """Generate a flattened JSON schema without $ref references for BigQuery compatibility."""
        schema = self.output_schema.model_json_schema()

        # If there are no $defs, return as-is
        if "$defs" not in schema:
            return schema

        # Flatten the schema by inlining all $ref references
        defs = schema.pop("$defs", {})

        def resolve_refs(obj):
            if isinstance(obj, dict):
                if "$ref" in obj:
                    ref_path = obj["$ref"]
                    if ref_path.startswith("#/$defs/"):
                        def_name = ref_path.replace("#/$defs/", "")
                        if def_name in defs:
                            # Get the resolved definition
                            resolved = resolve_refs(defs[def_name].copy())
                            # Preserve any properties from the original object (like description)
                            original_props = {
                                k: v for k, v in obj.items() if k != "$ref"
                            }
                            resolved.update(original_props)
                            return resolved
                        else:
                            # If ref not found, return the ref as-is (shouldn't happen)
                            return obj
                    else:
                        return obj
                else:
                    # Recursively resolve refs in all dictionary values
                    return {k: resolve_refs(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                # Recursively resolve refs in all list items
                return [resolve_refs(item) for item in obj]
            else:
                # Return primitive values as-is
                return obj

        return resolve_refs(schema)

    def _create_jsonl_payload(self) -> str:
        """Processes all requests into a JSONL string, uploading files as needed."""
        jsonl_lines = []
        gcs_session_folder = f"batch-inputs/{self._session_id}"

        for i, (request_key, data_model) in enumerate(self._requests):
            instance_id = f"req_{i:05d}_{uuid.uuid4().hex[:8]}"
            self._instance_map[instance_id] = request_key

            parts = []
            template_context = {}
            data_dict = data_model.model_dump()

            for field_name, value in data_dict.items():
                # More explicit file detection logic
                is_file_data = isinstance(value, (bytes, Path)) or (
                    isinstance(value, str)
                    and len(value) > 0
                    and Path(value).exists()
                    and Path(value).is_file()
                )

                if is_file_data:
                    # This is file data. Upload it.
                    if isinstance(value, Path):
                        filename = value.name
                    elif isinstance(value, str):
                        filename = Path(value).name
                    else:
                        filename = field_name

                    gcs_path = f"{gcs_session_folder}/{instance_id}/{filename}"
                    gcs_uri, mime_type = self._upload_file_to_gcs(value, gcs_path)
                    parts.append(
                        {"file_data": {"mime_type": mime_type, "file_uri": gcs_uri}}
                    )
                else:
                    # This is text data for the prompt template.
                    template_context[field_name] = value

            # Render the prompt template with the text data
            template = self._jinja_env.from_string(self.prompt_template)
            rendered_prompt = template.render(template_context)
            parts.append({"text": rendered_prompt})

            # Assemble the final JSONL line for this request
            instance_payload = {
                "id": instance_id,
                "request": {
                    "contents": [{"role": "user", "parts": parts}],
                    "generation_config": self.generation_config.model_dump(
                        exclude_none=True
                    ),
                    "tools": [
                        {
                            "function_declarations": [
                                {
                                    "name": "extract_info",
                                    "description": (
                                        "Extracts structured information "
                                        "based on the schema."
                                    ),
                                    "parameters": (self._get_flattened_schema()),
                                }
                            ]
                        }
                    ],
                    "tool_config": {"function_calling_config": {"mode": "any"}},
                },
            }
            jsonl_lines.append(json.dumps(instance_payload))

        return "\n".join(jsonl_lines)

    def submit(self, dry_run: bool = False) -> "Job[T]":
        """Constructs and submits the batch job."""
        if not self._requests:
            raise RuntimeError(
                "Cannot submit a job with no requests. Use .add_request() first."
            )

        if self.simulation_mode:
            logger.info("Simulation mode enabled. Skipping job submission.")
            # In simulation mode, create a mock job object instead of a string
            from unittest.mock import Mock

            self._batch_job = Mock()
            self._batch_job.state = JobState.JOB_STATE_SUCCEEDED
            self._batch_job.resource_name = (
                f"simulation://pyrtex-job-{self._session_id}"
            )
            self._batch_job.name = f"simulation-job-{self._session_id}"
            return self

        logger.info(
            f"Preparing job '{self._session_id}' with {len(self._requests)} requests..."
        )
        self._setup_cloud_resources()

        jsonl_payload = self._create_jsonl_payload()

        if dry_run:
            print("--- DRY RUN OUTPUT ---")
            print("Generated JSONL Payload (first 3 lines):")
            for line in jsonl_payload.splitlines()[:3]:
                print(json.dumps(json.loads(line), indent=2))
            print("----------------------")
            print("Dry run enabled. Job was not submitted.", file=sys.stderr)
            return self

        # Upload the generated payload to GCS
        gcs_session_folder = f"batch-inputs/{self._session_id}"
        gcs_path = f"{gcs_session_folder}/input.jsonl"
        gcs_uri, _ = self._upload_file_to_gcs(jsonl_payload.encode("utf-8"), gcs_path)
        logger.info(f"Uploaded JSONL payload to {gcs_uri}")

        # Submit the job
        job_display_name = f"pyrtex-job-{self._session_id}"
        project_id = self.config.project_id
        dataset_id = self.config.bq_dataset_id
        session_id = self._session_id
        bq_destination_prefix = (
            f"bq://{project_id}.{dataset_id}.batch_predictions_{session_id}"
        )

        model_resource_name = self.model
        if "/" not in model_resource_name:
            model_resource_name = f"publishers/google/models/{self.model}"

        self._batch_job = aiplatform.BatchPredictionJob.submit(
            job_display_name=job_display_name,
            model_name=model_resource_name,
            instances_format="jsonl",
            predictions_format="bigquery",
            gcs_source=gcs_uri,
            bigquery_destination_prefix=bq_destination_prefix,
            location=self.config.location,
            project=self.config.project_id,
        )

        logger.info(f"Batch job submitted: {self._batch_job.resource_name}")
        location = self.config.location
        batch_job_name = self._batch_job.name
        project_id = self.config.project_id
        console_url = (
            f"https://console.cloud.google.com/vertex-ai/locations/{location}/"
            f"batch-predictions/{batch_job_name}?project={project_id}"
        )
        logger.info(f"View in console: {console_url}")
        return self

    def wait(self) -> "Job[T]":
        """Waits for the submitted batch job to complete."""
        if self.simulation_mode:
            logger.info("Simulation mode enabled. Skipping wait.")
            return self

        if not self._batch_job:
            logger.warning("No job submitted, nothing to wait for.")
            return self

        logger.info("Waiting for job to complete...")
        self._batch_job.wait_for_completion()
        logger.info("Job completed!")
        return self

    def _process_usage_metadata(
        self, usage_metadata: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Process usage metadata to extract token counts from complex structures."""
        if not usage_metadata:
            return usage_metadata

        processed = usage_metadata.copy()

        # Extract token counts from candidatesTokensDetails
        if "candidatesTokensDetails" in processed and isinstance(
            processed["candidatesTokensDetails"], list
        ):
            if (
                len(processed["candidatesTokensDetails"]) > 0
                and "tokenCount" in processed["candidatesTokensDetails"][0]
            ):
                processed["candidatesTokensDetails"] = processed[
                    "candidatesTokensDetails"
                ][0]["tokenCount"]

        # Extract token counts from promptTokensDetails
        if "promptTokensDetails" in processed and isinstance(
            processed["promptTokensDetails"], list
        ):
            if (
                len(processed["promptTokensDetails"]) > 0
                and "tokenCount" in processed["promptTokensDetails"][0]
            ):
                processed["promptTokensDetails"] = processed["promptTokensDetails"][0][
                    "tokenCount"
                ]

        return processed

    def results(self) -> Iterator[BatchResult[T]]:
        """
        Retrieves results from the completed job, parsing them into the
        output schema.
        """
        if self.simulation_mode:
            yield from self._generate_dummy_results()
            return

        if not self._batch_job:
            raise RuntimeError(
                "Cannot get results for a job that has not been submitted."
            )

        # Check if job is completed successfully
        if self._batch_job.state != JobState.JOB_STATE_SUCCEEDED:
            job_state = self._batch_job.state
            msg = (
                "Cannot get results for a job that has not completed successfully. "
                f"Job state: {job_state}"
            )
            raise RuntimeError(msg)

        output_table = self._batch_job.output_info.bigquery_output_table.replace(
            "bq://", ""
        )
        logger.info(f"Querying results from BigQuery table: `{output_table}`")
        # Include status column to check for errors
        query = f"SELECT id, response, status FROM `{output_table}`"

        try:
            query_job = self._bigquery_client.query(query)
            for row in query_job.result():
                instance_id = row.id
                request_key = self._instance_map.get(instance_id)

                result_args = {
                    "request_key": request_key,
                    "raw_response": {},
                    "usage_metadata": None,
                }

                # Check status first for errors
                if hasattr(row, "status") and row.status and row.status != "{}":
                    # Parse status for error information
                    try:
                        status_dict = (
                            json.loads(row.status)
                            if isinstance(row.status, str)
                            else row.status
                        )
                        if "error" in status_dict:
                            error_info = status_dict["error"]
                            if isinstance(error_info, dict):
                                error_msg = error_info.get("message", str(error_info))
                                error_code = error_info.get("code", "")
                                result_args["error"] = (
                                    f"API Error {error_code}: {error_msg}"
                                )
                            else:
                                result_args["error"] = f"API Error: {error_info}"
                        else:
                            result_args["error"] = (
                                f"Request failed with status: {row.status}"
                            )
                    except (json.JSONDecodeError, TypeError):
                        result_args["error"] = (
                            f"Request failed with status: {row.status}"
                        )

                    yield BatchResult[T](**result_args)
                    continue

                # Process successful response
                if not row.response:
                    result_args["error"] = "Empty response from API"
                    yield BatchResult[T](**result_args)
                    continue

                # Check if response is already a dict or needs to be parsed
                if isinstance(row.response, dict):
                    response_dict = row.response
                else:
                    try:
                        response_dict = json.loads(row.response)
                    except json.JSONDecodeError as e:
                        result_args["error"] = f"Failed to parse response JSON: {e}"
                        yield BatchResult[T](**result_args)
                        continue

                result_args["raw_response"] = response_dict

                # Process usage metadata to extract token counts
                usage_metadata = response_dict.get("usageMetadata")
                processed_usage_metadata = self._process_usage_metadata(usage_metadata)
                result_args["usage_metadata"] = processed_usage_metadata

                try:
                    # Extract the function call arguments which contain
                    # the structured data
                    part = response_dict["candidates"][0]["content"]["parts"][0]
                    if "functionCall" not in part:
                        raise KeyError("Model did not return a function call.")

                    args = part["functionCall"]["args"]
                    parsed_output = self.output_schema.model_validate(args)
                    result_args["output"] = parsed_output

                except (KeyError, IndexError, TypeError) as e:
                    # Check if there's an error in the response
                    if "error" in response_dict:
                        error_info = response_dict["error"]
                        if isinstance(error_info, dict):
                            error_msg = error_info.get("message", str(error_info))
                            error_code = error_info.get("code", "")
                            result_args["error"] = (
                                f"Response Error {error_code}: {error_msg}"
                            )
                        else:
                            result_args["error"] = f"Response Error: {error_info}"
                    else:
                        result_args["error"] = f"Failed to parse model output: {e}"
                except Exception as e:
                    result_args["error"] = f"Validation error: {e}"

                yield BatchResult[T](**result_args)

        except Exception as e:
            raise RuntimeError(
                f"Error querying or parsing BigQuery results: {e}"
            ) from e

    def _generate_dummy_results(self) -> Iterator[BatchResult[T]]:
        """Generates dummy results for simulation mode."""
        for request_key, data_model in self._requests:
            # Generate dummy output data based on the schema
            dummy_output = self._create_dummy_output()

            # Create a dummy raw response
            raw_response = {
                "content": {"parts": [{"text": "dummy response"}]},
                "note": "This is a dummy response generated in simulation mode",
            }

            # Create usage metadata
            usage_metadata = {
                "promptTokenCount": 0,
                "candidatesTokenCount": 0,
                "totalTokenCount": 0,
            }

            yield BatchResult(
                request_key=request_key,
                output=dummy_output,
                raw_response=raw_response,
                usage_metadata=usage_metadata,
            )

    def _create_dummy_output(self) -> T:
        """Creates a dummy output instance based on the output schema."""
        from datetime import datetime
        from typing import get_args, get_origin

        from pydantic_core import PydanticUndefined

        # Get the schema fields
        schema_fields = self.output_schema.model_fields
        dummy_data = {}

        for field_name, field_info in schema_fields.items():
            # Check if field has a default value (but not PydanticUndefined)
            if (
                field_info.default is not None
                and field_info.default != PydanticUndefined
                and field_info.default != field_info.default_factory
            ):
                dummy_data[field_name] = field_info.default
            elif field_info.default_factory is not None:
                dummy_data[field_name] = field_info.default_factory()
            else:
                # Generate dummy data based on field type annotation
                field_type = field_info.annotation

                # Handle generic types like Optional[str], list[str], dict[str, str]
                origin = get_origin(field_type)
                args = get_args(field_type)

                if origin is Union:
                    # Get the first non-None type from Union (like Optional[str])
                    field_type = next(
                        (arg for arg in args if arg is not type(None)), str
                    )
                    origin = get_origin(field_type)
                    args = get_args(field_type)

                if origin is list or field_type is list:
                    dummy_data[field_name] = [f"dummy_{field_name}_item"]
                elif origin is dict or field_type is dict:
                    dummy_data[field_name] = {
                        f"dummy_{field_name}_key": f"dummy_{field_name}_value"
                    }
                elif field_type == str:
                    dummy_data[field_name] = f"dummy_{field_name}"
                elif field_type == int:
                    dummy_data[field_name] = 42
                elif field_type == float:
                    dummy_data[field_name] = 3.14
                elif field_type == bool:
                    dummy_data[field_name] = True
                elif field_type == datetime:
                    dummy_data[field_name] = datetime.now()
                else:
                    # For complex types, use a string representation
                    dummy_data[field_name] = f"dummy_{field_name}"

        return self.output_schema(**dummy_data)

    def _validate_enum_values(self):
        """Validates that enum values don't conflict with JSON boolean interpretation."""
        from enum import Enum
        from typing import get_args, get_origin

        # Problematic enum values that can be interpreted as booleans
        PROBLEMATIC_VALUES = {
            "yes",
            "no",
            "true",
            "false",
            "Yes",
            "No",
            "True",
            "False",
            "YES",
            "NO",
            "TRUE",
            "FALSE",
            "y",
            "n",
            "Y",
            "N",
            "1",
            "0",
        }

        def check_field_enums(field_type, field_name: str):
            """Recursively check field types for problematic enum values."""
            # Handle Union types (like Optional[Enum])
            origin = get_origin(field_type)
            if origin is not None:
                args = get_args(field_type)
                for arg in args:
                    if arg is not type(None):  # Skip NoneType
                        check_field_enums(arg, field_name)
                return

            # Check if this is an enum class
            if isinstance(field_type, type) and issubclass(field_type, Enum):
                for enum_member in field_type:
                    enum_value = enum_member.value
                    if isinstance(enum_value, str) and enum_value.lower() in {
                        v.lower() for v in PROBLEMATIC_VALUES
                    }:
                        raise ValueError(
                            f"Enum value '{enum_value}' in field '{field_name}' of enum '{field_type.__name__}' "
                            f"conflicts with JSON boolean interpretation. "
                            f"Problematic values: {sorted(PROBLEMATIC_VALUES)}. "
                            f"Consider using values like 'recommend'/'not_recommend' instead of 'yes'/'no'."
                        )

        # Check all fields in the output schema
        schema_fields = self.output_schema.model_fields
        for field_name, field_info in schema_fields.items():
            field_type = field_info.annotation
            check_field_enums(field_type, field_name)
