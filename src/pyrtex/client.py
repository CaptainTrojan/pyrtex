# src/pyrtex/client.py

import json
import logging
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
    Manages the configuration, submission, and retrieval for a
    Vertex AI Batch Prediction Job.

    The generic type parameter T should match the global output_schema for type safety.
    You can override the schema for individual requests using .add_request().

    Example (Single Schema):
        job = Job[ContactInfo](
            model="gemini-1.5-flash-001",
            output_schema=ContactInfo,
            prompt_template="Extract contact info: {{ content }}"
        )
        job.add_request("req1", MyInput(content="..."))
        # Type checkers will know that results are of type BatchResult[ContactInfo]
        # (Note: the return type hint is now BatchResult[Any] for flexibility)

    Example (Multiple Schemas and Configs):
        # Set defaults for the job
        job = Job[ContactInfo](model="...", output_schema=ContactInfo, ...)
        # Use the default schema and config for this request
        job.add_request("req1", MyInput(content="..."))
        # Override the schema for another request
        job.add_request("req2", MyInput(content="..."), output_schema=CompanyInfo)
        # Override the generation config for another request
        custom_config = GenerationConfig(temperature=0.9, max_output_tokens=1000)
        job.add_request("req3", MyInput(content="..."), generation_config=custom_config)

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

        # Validate the global schema for problematic enum values
        self._validate_schema(self.output_schema)

        self._session_id: str = uuid.uuid4().hex[:10]
        self._requests: List[
            tuple[
                Hashable,
                BaseModel,
                Optional[Type[BaseModel]],
                Optional[str],
                Optional[GenerationConfig],
            ]
        ] = []
        self._instance_map: Dict[str, tuple[Hashable, Type[BaseModel]]] = {}
        self._batch_job: Optional[aiplatform.BatchPredictionJob] = None

        self._jinja_env = jinja2.Environment()
        self._initialize_gcp()

    def _initialize_gcp(self):
        """Initialize GCP clients with flexible authentication and resolve config."""
        if self.simulation_mode:
            logger.info("Simulation mode enabled. Using mock GCP clients.")
            return

        try:
            credentials = self._get_credentials()

            # Initialize clients with credentials
            self._storage_client = storage.Client(
                project=self.config.project_id, credentials=credentials
            )
            self._bigquery_client = bigquery.Client(
                project=self.config.project_id, credentials=credentials
            )
            aiplatform.init(
                project=self.config.project_id,
                location=self.config.location,
                credentials=credentials,
            )

            self._resolve_infra_config()
            project_id = self.config.project_id
            location = self.config.location
            logger.info(
                f"Pyrtex initialized for project '{project_id}' in '{location}'."
            )
        except Exception as e:
            self._handle_authentication_error(e)

    def _get_credentials(self):
        """Get Google Cloud credentials using various methods."""
        import os

        # Method 1: Service account key JSON string (environment variable)
        if self.config.service_account_key_json:
            logger.info("Using service account credentials from JSON string")
            return self._credentials_from_json_string(
                self.config.service_account_key_json
            )

        # Method 2: Service account key file path (explicit service account file)
        elif (
            self.config.service_account_key_path
            and os.path.exists(self.config.service_account_key_path)
            and self._is_service_account_file(self.config.service_account_key_path)
        ):
            logger.info(
                f"Using service account credentials from file: "
                f"{self.config.service_account_key_path}"
            )
            return self._credentials_from_file(self.config.service_account_key_path)

        # Method 3: Application Default Credentials
        # (covers both gcloud user login and service accounts)
        else:
            logger.info("Using Application Default Credentials")
            return self._credentials_from_adc()

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

    def _credentials_from_json_string(self, json_string: str):
        """Create credentials from JSON string."""
        import json

        from google.oauth2 import service_account

        try:
            key_info = json.loads(json_string)
            credentials = service_account.Credentials.from_service_account_info(
                key_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

            # Auto-discover project ID if not set
            if not self.config.project_id and hasattr(credentials, "project_id"):
                self.config.project_id = credentials.project_id

            return credentials
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in service account key: {e}")
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load service account from JSON string: {e}"
            )

    def _credentials_from_file(self, file_path: str):
        """Create credentials from service account file."""
        from google.oauth2 import service_account

        try:
            credentials = service_account.Credentials.from_service_account_file(
                file_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

            # Auto-discover project ID if not set
            if not self.config.project_id and hasattr(credentials, "project_id"):
                self.config.project_id = credentials.project_id

            return credentials
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load service account from file '{file_path}': {e}"
            )

    def _credentials_from_adc(self):
        """Create credentials using Application Default Credentials."""
        import google.auth

        try:
            credentials, discovered_project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

            # Auto-discover project ID if not set
            if not self.config.project_id and discovered_project:
                self.config.project_id = discovered_project

            return credentials
        except Exception as e:
            raise ConfigurationError(
                f"Failed to get Application Default Credentials: {e}"
            )

    def _handle_authentication_error(self, error: Exception):
        """Provide helpful error messages for authentication failures."""
        msg1 = "Failed to initialize GCP clients. "

        # Check if any authentication method was attempted
        if self.config.service_account_key_json:
            msg2 = (
                "Issue with service account JSON string. "
                "Please verify the JSON is valid. "
            )
        elif self.config.service_account_key_path:
            msg2 = (
                f"Issue with service account file "
                f"'{self.config.service_account_key_path}'. "
                f"Please verify the file exists and is valid. "
            )
        else:
            msg2 = "No authentication method configured. Try one of these:\n"
            msg2 += "  1) Set PYRTEX_SERVICE_ACCOUNT_KEY_JSON env var\n"
            msg2 += "  2) Set GOOGLE_APPLICATION_CREDENTIALS env var\n"
            msg2 += "  3) Run 'gcloud auth application-default login'\n"
            msg2 += "  4) Use simulation_mode=True for testing\n"

        # Add specific help for common ADC issues
        error_str = str(error).lower()
        if "application default credentials" in error_str or "adc" in error_str:
            msg2 += "\n💡 ADC Troubleshooting:\n"
            msg2 += "  - Run 'gcloud auth application-default login'\n"
            msg2 += "  - Set project ID (PYRTEX_PROJECT_ID or GOOGLE_PROJECT_ID)\n"
            msg2 += "  - Check required permissions in your GCP project\n"

        raise ConfigurationError(msg1 + msg2 + f"\nOriginal error: {error}") from error

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

    def add_request(
        self,
        request_key: Hashable,
        data: BaseModel,
        output_schema: Optional[Type[BaseModel]] = None,
        prompt_template: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> "Job[T]":
        """
        Adds a single, structured request to the batch.

        Args:
            request_key: A unique, hashable identifier for this request.
            data: A Pydantic model instance containing the input data for the prompt.
            output_schema: (Optional) A Pydantic model to use as the output schema
                for this specific request, overriding the job's default schema.
            prompt_template: (Optional) A string to use as the prompt template for
                this request, overriding the job's default prompt template.
            generation_config: (Optional) A GenerationConfig to use for this specific
                request, overriding the job's default generation config.
        """
        if self._batch_job is not None:
            raise RuntimeError("Cannot add requests after job has been submitted.")

        # Check for duplicate request keys
        if any(key == request_key for key, _, _, _, _ in self._requests):
            msg = f"Request key '{request_key}' already exists. "
            msg += "Use a unique key for each request."
            raise ValueError(msg)

        # If an override schema is provided, validate it
        if output_schema:
            self._validate_schema(output_schema)

        self._requests.append(
            (request_key, data, output_schema, prompt_template, generation_config)
        )
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
            gemini_supported_types = {
                ".txt": "text/plain",
                ".yaml": "text/plain",
                ".yml": "text/plain",
                ".json": "text/plain",
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
                ".pdf": "application/pdf",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".mp3": "audio/mp3",
                ".mpeg": "audio/mpeg",
                ".wav": "audio/wav",
                ".mov": "video/mov",
                ".mp4": "video/mp4",
                ".mpv": "video/mpeg",
                ".mpg": "video/mpg",
                ".avi": "video/avi",
                ".wmv": "video/wmv",
                ".flv": "video/flv",
            }
            mime_type = gemini_supported_types.get(ext, "text/plain")

        if isinstance(source, bytes):
            blob.upload_from_string(source, content_type=mime_type)
        else:
            blob.upload_from_filename(str(source), content_type=mime_type)

        return f"gs://{self.config.gcs_bucket_name}/{gcs_path}", mime_type

    def _get_flattened_schema(
        self, schema_to_flatten: Optional[Type[BaseModel]] = None
    ) -> dict:
        """Generate a flattened JSON schema without $ref references.

        Backwards compatibility: prior versions exposed this helper without
        requiring an argument and implicitly operated on the job's global
        ``output_schema``. Tests (and potentially user code) still call it
        with no argument, so ``schema_to_flatten`` is optional and defaults
        to ``self.output_schema`` when omitted.
        """
        schema_to_flatten = schema_to_flatten or self.output_schema
        schema = schema_to_flatten.model_json_schema()

        if "$defs" not in schema:
            return schema

        defs = schema.pop("$defs", {})

        def resolve_refs(obj):
            if isinstance(obj, dict):
                if "$ref" in obj:
                    ref_path = obj["$ref"]
                    if ref_path.startswith("#/$defs/"):
                        def_name = ref_path.replace("#/$defs/", "")
                        if def_name in defs:
                            resolved = resolve_refs(defs[def_name].copy())
                            original_props = {
                                k: v for k, v in obj.items() if k != "$ref"
                            }
                            resolved.update(original_props)
                            return resolved
                        else:
                            return obj
                    else:
                        return obj
                else:
                    return {k: resolve_refs(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_refs(item) for item in obj]
            else:
                return obj

        return resolve_refs(schema)

    def _create_jsonl_payload(self) -> str:
        """Processes all requests into a JSONL string, uploading files as needed."""
        jsonl_lines = []
        gcs_session_folder = f"batch-inputs/{self._session_id}"

        for i, (
            request_key,
            data_model,
            override_schema,
            override_prompt,
            override_generation_config,
        ) in enumerate(self._requests):
            instance_id = f"req_{i:05d}_{uuid.uuid4().hex[:8]}"

            # Determine which schema to use and store it for result parsing
            schema_to_use = override_schema or self.output_schema
            self._instance_map[instance_id] = (request_key, schema_to_use)

            parts = []
            template_context = {}
            data_dict = data_model.model_dump()

            for field_name, value in data_dict.items():
                if isinstance(value, (bytes, Path)):
                    if isinstance(value, Path):
                        filename = value.name
                    else:
                        filename = field_name

                    gcs_path = f"{gcs_session_folder}/{instance_id}/{filename}"
                    gcs_uri, mime_type = self._upload_file_to_gcs(value, gcs_path)
                    parts.append(
                        {"file_data": {"mime_type": mime_type, "file_uri": gcs_uri}}
                    )
                else:
                    template_context[field_name] = value

            # Use per-request prompt if provided, else job-level prompt
            prompt_to_use = override_prompt or self.prompt_template
            template = self._jinja_env.from_string(prompt_to_use)
            rendered_prompt = template.render(template_context)
            parts.append({"text": rendered_prompt})

            # Use per-request generation config if provided, else job-level config
            generation_config_to_use = (
                override_generation_config or self.generation_config
            )
            request_gen_config = generation_config_to_use.model_dump(exclude_none=True)

            if (
                self.model == "gemini-2.5-pro"
                and request_gen_config["thinking_config"]["thinking_budget"] == 0
            ):
                # Can't disable thinking for this specific model. We'll limit it
                # at least
                request_gen_config["thinking_config"]["thinking_budget"] = 128

            request_gen_config["response_mime_type"] = "application/json"
            request_gen_config["response_schema"] = self._get_flattened_schema(
                schema_to_use
            )

            instance_payload = {
                "id": instance_id,
                "request": {
                    "contents": [{"role": "user", "parts": parts}],
                    "generation_config": request_gen_config,
                },
            }
            jsonl_lines.append(json.dumps(instance_payload))

        return "\n".join(jsonl_lines)

    def _create_pydantic_model_from_schema(
        self, schema: Dict[str, Any]
    ) -> Type[BaseModel]:
        """Dynamically creates a Pydantic model class from a JSON schema dictionary."""
        from pydantic import create_model

        model_name = schema.get("title", "DynamicModel")
        fields = {}
        for prop_name, prop_schema in schema.get("properties", {}).items():
            field_type = Any
            if prop_schema.get("type") == "integer":
                field_type = int
            elif prop_schema.get("type") == "number":
                field_type = float
            elif prop_schema.get("type") == "boolean":
                field_type = bool
            elif prop_schema.get("type") == "array":
                field_type = List
            elif prop_schema.get("type") == "object":
                field_type = Dict
            elif prop_schema.get("type") == "string":
                field_type = str
            fields[prop_name] = (field_type, ...)
        return create_model(model_name, **fields)

    def submit(self, dry_run: bool = False) -> "Job[T]":
        """Constructs and submits the batch job."""
        if not self._requests:
            raise RuntimeError(
                "Cannot submit a job with no requests. Use .add_request() first."
            )

        if self.simulation_mode:
            logger.info("Simulation mode enabled. Skipping job submission.")
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

        gcs_session_folder = f"batch-inputs/{self._session_id}"
        gcs_path = f"{gcs_session_folder}/input.jsonl"
        gcs_uri, _ = self._upload_file_to_gcs(jsonl_payload.encode("utf-8"), gcs_path)
        logger.info(f"Uploaded JSONL payload to {gcs_uri}")

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

    def results(self) -> Iterator[BatchResult[Any]]:
        """
        Retrieves results from the completed job, parsing them into the
        appropriate output schema for each request.

        Note: The return type is Iterator[BatchResult[Any]] because a single job
        can contain requests with different output schemas. You may need to perform
        a runtime type check (e.g., `isinstance()`) on the `output` attribute.
        """
        if self.simulation_mode:
            yield from self._generate_dummy_results()
            return

        if not self._batch_job:
            raise RuntimeError(
                "Cannot get results for a job that has not been submitted."
            )

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
        query = f"SELECT id, response, status FROM `{output_table}`"

        try:
            query_job = self._bigquery_client.query(query)
            for row in query_job.result():
                instance_id = row.id
                lookup_result = self._instance_map.get(instance_id)

                if not lookup_result:
                    logger.warning(
                        f"Could not find request data for instance ID '{instance_id}'"
                        ". Skipping."
                    )
                    continue

                # Expect new tuple-based mapping (request_key, schema)
                request_key, schema_used = lookup_result  # type: ignore[misc]

                result_args = {
                    "request_key": request_key,
                    "raw_response": {},
                    "usage_metadata": None,
                }

                if hasattr(row, "status") and row.status and row.status != "{}":
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

                    yield BatchResult[Any](**result_args)
                    continue

                if not row.response:
                    result_args["error"] = "Empty response from API"
                    yield BatchResult[Any](**result_args)
                    continue

                if isinstance(row.response, dict):
                    response_dict = row.response
                else:
                    try:
                        response_dict = json.loads(row.response)
                    except json.JSONDecodeError as e:
                        result_args["error"] = f"Failed to parse response JSON: {e}"
                        yield BatchResult[Any](**result_args)
                        continue

                result_args["raw_response"] = response_dict

                usage_metadata = response_dict.get("usageMetadata")
                processed_usage_metadata = self._process_usage_metadata(usage_metadata)
                result_args["usage_metadata"] = processed_usage_metadata

                try:
                    # In JSON mode, the output is in the 'text' of the first part
                    part = response_dict["candidates"][0]["content"]["parts"][0]
                    if "text" not in part:
                        raise KeyError("Model response did not contain a 'text' part.")

                    # The text part contains the JSON string, so we need to parse it
                    json_output = json.loads(part["text"])

                    # Now validate the parsed JSON against the Pydantic schema
                    parsed_output = schema_used.model_validate(json_output)
                    result_args["output"] = parsed_output

                except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
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

                yield BatchResult[Any](**result_args)

        except Exception as e:
            raise RuntimeError(
                f"Error querying or parsing BigQuery results: {e}"
            ) from e

    def _generate_dummy_results(self) -> Iterator[BatchResult[Any]]:
        """Generates dummy results for simulation mode."""
        for request_key, _, override_schema, _, _ in self._requests:
            schema_to_mock = override_schema or self.output_schema
            dummy_output = self._create_dummy_output(schema_to_mock)

            raw_response = {
                "content": {"parts": [{"text": "dummy response"}]},
                "note": "This is a dummy response generated in simulation mode",
            }

            usage_metadata = {
                "promptTokenCount": 0,
                "candidatesTokenCount": 0,
                "thoughtsTokenCount": 0,
                "totalTokenCount": 0,
            }

            yield BatchResult(
                request_key=request_key,
                output=dummy_output,
                raw_response=raw_response,
                usage_metadata=usage_metadata,
            )

    def _create_dummy_output(self, schema_to_mock: Type[BaseModel] = None) -> BaseModel:
        """Creates a dummy output instance based on the provided schema."""
        from datetime import datetime
        from typing import get_args, get_origin

        from pydantic_core import PydanticUndefined

        schema_fields = (
            schema_to_mock.model_fields
            if schema_to_mock
            else self.output_schema.model_fields
        )
        dummy_data = {}

        for field_name, field_info in schema_fields.items():
            if (
                field_info.default is not None
                and field_info.default != PydanticUndefined
                and field_info.default != field_info.default_factory
            ):
                dummy_data[field_name] = field_info.default
            elif field_info.default_factory is not None:
                dummy_data[field_name] = field_info.default_factory()
            else:
                field_type = field_info.annotation
                origin = get_origin(field_type)
                args = get_args(field_type)

                if origin is Union:
                    field_type = next(
                        (arg for arg in args if arg is not type(None)), str
                    )
                    origin = get_origin(field_type)

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
                    dummy_data[field_name] = f"dummy_{field_name}"

        return schema_to_mock(**dummy_data)

    def _validate_schema(self, schema_to_validate: Type[BaseModel]):
        """
        Validates that a Pydantic schema is compatible with Vertex AI's
        function calling requirements. This check is recursive.

        Specifically, it prohibits:
        1.  Union and Optional types, which generate `anyOf` in JSON Schema
            and are not supported by Vertex AI.
        2.  Enum values that can be misinterpreted as booleans (e.g., "yes", "no").
        """
        from enum import Enum
        from typing import Any, Union, get_args, get_origin

        from pydantic import BaseModel

        # Using a set for efficient lookup of visited models to prevent infinite
        # recursion in case of self-referencing models.
        visited_models = set()

        def _recursive_validate(model_to_check: Type[BaseModel], parent_path: str = ""):
            """Inner function to recursively validate nested models."""
            if model_to_check in visited_models:
                return
            visited_models.add(model_to_check)

            for field_name, field_info in model_to_check.model_fields.items():
                current_path = (
                    f"{parent_path}.{field_name}" if parent_path else field_name
                )
                _check_field_type(field_info.annotation, current_path)

        def _check_field_type(field_type: Any, field_path: str):
            """
            Recursively checks a type annotation for compatibility issues
            within a field.
            """
            origin = get_origin(field_type)
            args = get_args(field_type)

            # --- 1. Prohibit Union and Optional types ---
            if origin is Union:
                # Identify if it's Optional[T] (i.e., Union[T, None]) for a clearer
                # error message
                is_optional = len(args) == 2 and args[1] is type(None)
                error_type = "Optional" if is_optional else "Union"

                raise ValueError(
                    f"Field '{field_path}' uses an {error_type} type, "
                    "which is not supported by Vertex AI function calling "
                    "as it produces 'anyOf' in the schema. "
                    "Consider making the field required and providing a default value "
                    "(e.g., an empty string '' or an empty list []) "
                    "instead of None."
                )

            # --- 2. Recurse into generic container types ---
            if origin in (list, dict):
                if origin is list and args:
                    # For list[T], check T
                    _check_field_type(args[0], f"{field_path}[]")
                elif origin is dict and args:
                    # For dict[K, V], keys must be strings and we check V
                    if args[0] is not str:
                        raise ValueError(
                            f"Dictionary keys in '{field_path}' must be of type str "
                            f"for JSON compatibility."
                        )
                    _check_field_type(args[1], f"{field_path}.value")
                return  # Stop processing after handling the container

            # Use the origin if it exists (like `list` from `list[str]`),
            # otherwise use the type itself.
            type_to_check = origin or field_type

            # --- 3. Check for problematic Enum values ---
            if isinstance(type_to_check, type) and issubclass(type_to_check, Enum):
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
                for member in type_to_check:
                    if isinstance(member.value, str) and member.value.lower() in {
                        v.lower() for v in PROBLEMATIC_VALUES
                    }:
                        raise ValueError(
                            f"Enum value '{member.value}' in '{field_path}' of enum "
                            f"'{type_to_check.__name__}' conflicts with JSON boolean "
                            f"interpretation. Consider using different values (e.g., "
                            f"'approved'/'rejected' instead of 'yes'/'no')."
                        )

            # --- 4. Recurse into nested Pydantic models ---
            if isinstance(type_to_check, type) and issubclass(type_to_check, BaseModel):
                _recursive_validate(type_to_check, field_path)

        # Start the validation process from the top-level schema
        _recursive_validate(schema_to_validate)

    def serialize(self) -> str:
        """
        Serializes the job's state to a JSON string, embedding schema definitions.
        This state contains all necessary information to reconnect to the
        job from another process to check its status and retrieve results.
        Raises:
            RuntimeError: If the job has not been submitted yet.
        Returns:
            A JSON string representing the job's state.
        """
        if not self._batch_job:
            raise RuntimeError("Cannot serialize a job that has not been submitted.")

        serializable_instance_map = {
            instance_id: (
                request_key,
                self._get_flattened_schema(schema),  # Store schema dict
            )
            for instance_id, (request_key, schema) in self._instance_map.items()
        }

        state = {
            "batch_job_resource_name": self._batch_job.resource_name,
            "session_id": self._session_id,
            "model": self.model,
            "infrastructure_config": self.config.model_dump(mode="json"),
            "instance_map": serializable_instance_map,
        }
        return json.dumps(state)

    @classmethod
    def reconnect_from_state(cls, state_json: str) -> "Job[T]":
        """
        Reconnects to a job, dynamically recreating schemas from the state file.
        Args:
            state_json: The JSON string generated by the .serialize() method.
        Returns:
            A new Job instance linked to the existing cloud job.
        """
        state_data = json.loads(state_json)
        config = InfrastructureConfig(**state_data["infrastructure_config"])

        reconnected_job = cls(
            model=state_data["model"],
            output_schema=BaseModel,
            prompt_template="",
            config=config,
        )
        reconnected_job._initialize_gcp()
        reconnected_job._batch_job = aiplatform.BatchPredictionJob(
            state_data["batch_job_resource_name"]
        )
        reconnected_job._session_id = state_data["session_id"]

        # Recreate models from schema dicts instead of importing
        reconnected_job._instance_map = {}
        for instance_id, (req_key, schema_dict) in state_data["instance_map"].items():
            schema_class = reconnected_job._create_pydantic_model_from_schema(
                schema_dict
            )
            reconnected_job._instance_map[instance_id] = (req_key, schema_class)

        reconnected_job._requests = []
        return reconnected_job

    @property
    def status(self) -> Optional[JobState]:
        """Returns the current state of the Vertex AI job."""
        if not self._batch_job:
            return None
        return self._batch_job.state

    @property
    def is_done(self) -> bool:
        """Returns True if the job has finished (succeeded, failed, or cancelled)."""
        current_status = self.status
        if not current_status:
            return False
        return current_status in [
            JobState.JOB_STATE_SUCCEEDED,
            JobState.JOB_STATE_FAILED,
            JobState.JOB_STATE_CANCELLED,
            JobState.JOB_STATE_EXPIRED,
        ]
