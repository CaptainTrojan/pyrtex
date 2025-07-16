# src/pyrtex/client.py

import uuid
import json
import logging
import mimetypes
from pathlib import Path
from typing import Type, Optional, Hashable, Iterator, List, Any, Dict, Union, Generic

from pydantic import BaseModel
import jinja2
import google.cloud.aiplatform as aiplatform
import google.cloud.storage as storage
import google.cloud.bigquery as bigquery
from google.api_core.exceptions import NotFound

from .config import InfrastructureConfig, GenerationConfig
from .models import BatchResult, T
from .exceptions import ConfigurationError, JobFailedError

logger = logging.getLogger(__name__)


class Job(Generic[T]):
    """
    Manages the configuration, submission, and result retrieval for a
    Vertex AI Batch Prediction Job.
    """

    def __init__(
        self,
        output_schema: Type[T],
        prompt_template: str,
        model: str = "gemini-1.5-flash-001",
        generation_config: Optional[GenerationConfig] = None,
        config: Optional[InfrastructureConfig] = None,
    ):
        self.output_schema = output_schema
        self.prompt_template = prompt_template
        self.model = model
        self.generation_config = generation_config or GenerationConfig()
        self.config = config or InfrastructureConfig()

        self._session_id: str = uuid.uuid4().hex[:10]
        self._requests: List[tuple[Hashable, BaseModel]] = []
        self._instance_map: Dict[str, Hashable] = {}
        self._batch_job: Optional[aiplatform.BatchPredictionJob] = None
        self._results_cache: Optional[List[BatchResult[T]]] = None
        
        self._jinja_env = jinja2.Environment()
        self._initialize_gcp()
        
    # ... (keep _initialize_gcp, _resolve_infra_config, _setup_cloud_resources) ...
    def _initialize_gcp(self):
        """Initializes GCP clients and resolves final configuration."""
        try:
            self._storage_client = storage.Client(project=self.config.project_id)
            self._bigquery_client = bigquery.Client(project=self.config.project_id)
            aiplatform.init(project=self.config.project_id, location=self.config.location)
            self._resolve_infra_config()
            logger.info(f"Pyrtex initialized for project '{self.config.project_id}' in '{self.config.location}'.")
        except Exception as e:
            raise ConfigurationError(
                "Failed to initialize GCP clients. Please ensure you are authenticated. "
                "Run 'gcloud auth application-default login' in your terminal. "
                f"Original error: {e}"
            ) from e

    def _resolve_infra_config(self):
        """Fills in missing infrastructure config values with sensible defaults."""
        if not self.config.project_id:
            self.config.project_id = self._storage_client.project
            if not self.config.project_id:
                raise ConfigurationError(
                    "Could not automatically discover GCP Project ID. "
                    "Please set the GOOGLE_PROJECT_ID environment variable or pass it in InfrastructureConfig."
                )
        if not self.config.gcs_bucket_name:
            self.config.gcs_bucket_name = f"pyrtex-assets-{self.config.project_id}"
            logger.info(f"GCS bucket not specified, using default: '{self.config.gcs_bucket_name}'")
        if not self.config.bq_dataset_id:
            self.config.bq_dataset_id = "pyrtex_results"
            logger.info(f"BigQuery dataset not specified, using default: '{self.config.bq_dataset_id}'")

    def _setup_cloud_resources(self):
        """Ensures the GCS bucket and BigQuery dataset exist and are configured correctly."""
        logger.info("Verifying and setting up cloud resources...")
        try:
            bucket = self._storage_client.get_bucket(self.config.gcs_bucket_name)
        except NotFound:
            logger.info(f"Creating GCS bucket '{self.config.gcs_bucket_name}' in {self.config.location}...")
            bucket = self._storage_client.create_bucket(self.config.gcs_bucket_name, location=self.config.location)
        bucket.clear_lifecycle_rules()
        bucket.add_lifecycle_delete_rule(age=1)
        bucket.patch()
        logger.info("GCS bucket is ready.")
        dataset_id_full = f"{self.config.project_id}.{self.config.bq_dataset_id}"
        try:
            dataset = self._bigquery_client.get_dataset(self.config.bq_dataset_id)
        except NotFound:
            logger.info(f"Creating BigQuery dataset '{dataset_id_full}' in {self.config.location}...")
            dataset_ref = bigquery.Dataset(dataset_id_full)
            dataset_ref.location = self.config.location
            dataset = self._bigquery_client.create_dataset(dataset_ref)
        dataset.default_table_expiration_ms = 24 * 60 * 60 * 1000
        self._bigquery_client.update_dataset(dataset, ["default_table_expiration_ms"])
        logger.info("BigQuery dataset is ready.")


    def add_request(self, request_key: Hashable, data: BaseModel) -> 'Job[T]':
        """Adds a single, structured request to the batch."""
        if self._batch_job:
            raise RuntimeError("Cannot add requests after job has been submitted.")
        self._requests.append((request_key, data))
        return self

    def _upload_file_to_gcs(self, source: Union[str, bytes, Path], gcs_path: str) -> str:
        """Uploads a local file or bytes to GCS and returns its URI."""
        bucket = self._storage_client.bucket(self.config.gcs_bucket_name)
        blob = bucket.blob(gcs_path)
        mime_type, _ = mimetypes.guess_type(str(source))
        mime_type = mime_type or "application/octet-stream"

        if isinstance(source, bytes):
            blob.upload_from_string(source, content_type=mime_type)
        else:
            blob.upload_from_filename(str(source), content_type=mime_type)
        
        return f"gs://{self.config.gcs_bucket_name}/{gcs_path}", mime_type

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
                if isinstance(value, (bytes, Path)) or (isinstance(value, str) and Path(value).is_file()):
                    # This is file data. Upload it.
                    gcs_path = f"{gcs_session_folder}/{instance_id}/{Path(value).name if not isinstance(value, bytes) else field_name}"
                    gcs_uri, mime_type = self._upload_file_to_gcs(value, gcs_path)
                    parts.append({"file_data": {"mime_type": mime_type, "file_uri": gcs_uri}})
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
                    "generation_config": self.generation_config.model_dump(exclude_none=True),
                    "tools": [{"function_declarations": [
                        {
                            "name": "extract_info", 
                            "description": "Extracts structured information based on the schema.",
                            "parameters": self.output_schema.model_json_schema()
                        }
                    ]}],
                    "tool_config": {"function_calling_config": {"mode": "any"}}
                },
            }
            jsonl_lines.append(json.dumps(instance_payload))

        return "\n".join(jsonl_lines)

    def submit(self, dry_run: bool = False) -> 'Job[T]':
        """Constructs and submits the batch job."""
        if not self._requests:
            raise RuntimeError("Cannot submit a job with no requests. Use .add_request() first.")
        
        logger.info(f"Preparing job '{self._session_id}' with {len(self._requests)} requests...")
        self._setup_cloud_resources()

        jsonl_payload = self._create_jsonl_payload()
        
        if dry_run:
            print("--- DRY RUN OUTPUT ---")
            print("Generated JSONL Payload (first 3 lines):")
            for line in jsonl_payload.splitlines()[:3]:
                print(json.dumps(json.loads(line), indent=2))
            print("----------------------")
            logger.info("Dry run enabled. Job was not submitted.")
            return self

        # Upload the generated payload to GCS
        gcs_session_folder = f"batch-inputs/{self._session_id}"
        gcs_path = f"{gcs_session_folder}/input.jsonl"
        gcs_uri, _ = self._upload_file_to_gcs(jsonl_payload.encode('utf-8'), gcs_path)
        logger.info(f"Uploaded JSONL payload to {gcs_uri}")

        # Submit the job
        job_display_name = f"pyrtex-job-{self._session_id}"
        bq_destination_prefix = f"bq://{self.config.project_id}.{self.config.bq_dataset_id}"
        
        model_resource_name = self.model
        if not "/" in model_resource_name:
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
        logger.info(f"View in console: https://console.cloud.google.com/vertex-ai/locations/{self.config.location}/batch-predictions/{self._batch_job.name}?project={self.config.project_id}")
        return self

    def wait(self) -> 'Job[T]':
        """Waits for the submitted batch job to complete."""
        # --- Placeholder ---
        print("--- WAIT LOGIC NOT YET IMPLEMENTED ---")
        if not self._batch_job:
            print("No job submitted, nothing to wait for.")
            return self
        logger.info(f"Waiting for job to complete...")
        # self._batch_job.wait()
        logger.info("Job completed!")
        return self

    def results(self) -> Iterator[BatchResult[T]]:
        """Retrieves the results from the completed job."""
        # --- Placeholder ---
        print("--- RESULTS LOGIC NOT YET IMPLEMENTED ---")
        if not self._batch_job:
             raise RuntimeError("Cannot get results for a job that has not been submitted or completed.")
        if self._results_cache:
            yield from self._results_cache
            return
        logger.info("Fetching results...")
        self._results_cache = []
        yield from self._results_cache