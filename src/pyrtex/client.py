# src/pyrtex/client.py

import uuid
import json
import logging
import mimetypes
import sys
from pathlib import Path
from typing import Type, Optional, Hashable, Iterator, List, Any, Dict, Union, Generic

from pydantic import BaseModel, ValidationError
import jinja2
import google.cloud.aiplatform as aiplatform
import google.cloud.storage as storage
import google.cloud.bigquery as bigquery
from google.api_core.exceptions import NotFound
from google.cloud.aiplatform_v1.types import JobState

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

        self._session_id: str = uuid.uuid4().hex[:10]
        self._requests: List[tuple[Hashable, BaseModel]] = []
        self._instance_map: Dict[str, Hashable] = {}
        self._batch_job: Optional[aiplatform.BatchPredictionJob] = None
        self._results_cache: Optional[List[BatchResult[T]]] = None
        
        self._jinja_env = jinja2.Environment()
        self._initialize_gcp()
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
        if self._batch_job is not None:
            raise RuntimeError("Cannot add requests after job has been submitted.")
        self._requests.append((request_key, data))
        return self

    def _upload_file_to_gcs(self, source: Union[str, bytes, Path], gcs_path: str) -> tuple[str, str]:
        """Uploads a local file or bytes to GCS and returns its URI and mime type."""
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
                if isinstance(value, (bytes, Path)) or (isinstance(value, str) and Path(value).exists()):
                    # This is file data. Upload it.
                    if isinstance(value, Path):
                        filename = value.name
                    elif isinstance(value, str):
                        filename = Path(value).name
                    else:
                        filename = field_name
                    
                    gcs_path = f"{gcs_session_folder}/{instance_id}/{filename}"
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
        
        if self.simulation_mode:
            logger.info("Simulation mode enabled. Skipping job submission.")
            # In simulation mode, we mark the job as submitted but don't create a real batch job
            self._batch_job = "simulation_mode_marker"
            return self
        
        logger.info(f"Preparing job '{self._session_id}' with {len(self._requests)} requests...")
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
        gcs_uri, _ = self._upload_file_to_gcs(jsonl_payload.encode('utf-8'), gcs_path)
        logger.info(f"Uploaded JSONL payload to {gcs_uri}")

        # Submit the job
        job_display_name = f"pyrtex-job-{self._session_id}"
        bq_destination_prefix = f"bq://{self.config.project_id}.{self.config.bq_dataset_id}.batch_predictions_{self._session_id}"
        
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
        if self.simulation_mode:
            logger.info("Simulation mode enabled. Skipping wait.")
            return self
            
        if not self._batch_job:
            logger.warning("No job submitted, nothing to wait for.")
            return self
            
        logger.info(f"Waiting for job to complete...")
        self._batch_job.wait_for_completion()
        logger.info("Job completed!")
        return self

    def _process_usage_metadata(self, usage_metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Process usage metadata to extract token counts from complex structures."""
        if not usage_metadata:
            return usage_metadata
            
        processed = usage_metadata.copy()
        
        # Extract token counts from candidatesTokensDetails
        if "candidatesTokensDetails" in processed and isinstance(processed["candidatesTokensDetails"], list):
            if len(processed["candidatesTokensDetails"]) > 0 and "tokenCount" in processed["candidatesTokensDetails"][0]:
                processed["candidatesTokensDetails"] = processed["candidatesTokensDetails"][0]["tokenCount"]
        
        # Extract token counts from promptTokensDetails
        if "promptTokensDetails" in processed and isinstance(processed["promptTokensDetails"], list):
            if len(processed["promptTokensDetails"]) > 0 and "tokenCount" in processed["promptTokensDetails"][0]:
                processed["promptTokensDetails"] = processed["promptTokensDetails"][0]["tokenCount"]
        
        return processed

    def results(self) -> Iterator[BatchResult[T]]:
        """Retrieves results from the completed job, parsing them into the output schema."""
        if self.simulation_mode:
            yield from self._generate_dummy_results()
            return
            
        if self._results_cache is not None:
            yield from self._results_cache
            return
            
        if not self._batch_job:
            raise RuntimeError("Cannot get results for a job that has not been submitted.")
            
        # Check if job is completed successfully
        if self._batch_job.state != JobState.JOB_STATE_SUCCEEDED:
            raise RuntimeError(f"Cannot get results for a job that has not completed successfully. Job state: {self._batch_job.state}")

        self._results_cache = []
        output_table = self._batch_job.output_info.bigquery_output_table.replace("bq://", "")
        logger.info(f"Querying results from BigQuery table: `{output_table}`")
        query = f"SELECT id, response FROM `{output_table}`"
        
        try:
            query_job = self._bigquery_client.query(query)
            for row in query_job.result():
                instance_id = row.id
                request_key = self._instance_map.get(instance_id)
                
                # Check if response is already a dict or needs to be parsed
                if isinstance(row.response, dict):
                    response_dict = row.response
                else:
                    response_dict = json.loads(row.response)

                # Process usage metadata to extract token counts
                usage_metadata = response_dict.get("usageMetadata")
                processed_usage_metadata = self._process_usage_metadata(usage_metadata)
                
                result_args = {
                    "request_key": request_key,
                    "raw_response": response_dict,
                    "usage_metadata": processed_usage_metadata,
                }
                
                try:
                    # Extract the function call arguments which contain the structured data
                    part = response_dict['candidates'][0]['content']['parts'][0]
                    if 'functionCall' not in part:
                        raise KeyError("Model did not return a function call.")
                    
                    args = part['functionCall']['args']
                    parsed_output = self.output_schema.model_validate(args)
                    result_args["output"] = parsed_output
                    
                except (KeyError, IndexError, TypeError) as e:
                    result_args["error"] = f"Failed to parse model output: {e}"
                except Exception as e:
                    result_args["error"] = f"Validation error: {e}"
                
                self._results_cache.append(BatchResult[T](**result_args))

        except Exception as e:
            raise RuntimeError(f"Error querying or parsing BigQuery results: {e}") from e

        logger.info(f"Successfully fetched and parsed {len(self._results_cache)} results.")
        yield from self._results_cache

    def _generate_dummy_results(self) -> Iterator[BatchResult[T]]:
        """Generates dummy results for simulation mode."""
        for request_key, data_model in self._requests:
            # Generate dummy output data based on the schema
            dummy_output = self._create_dummy_output()
            
            # Create a dummy raw response
            raw_response = {
                "content": {"parts": [{"text": "dummy response"}]},
                "note": "This is a dummy response generated in simulation mode"
            }
            
            # Create usage metadata
            usage_metadata = {
                "promptTokenCount": 0,
                "candidatesTokenCount": 0,
                "totalTokenCount": 0
            }
            
            yield BatchResult(
                request_key=request_key,
                output=dummy_output,
                raw_response=raw_response,
                usage_metadata=usage_metadata
            )

    def _create_dummy_output(self) -> T:
        """Creates a dummy output instance based on the output schema."""
        from pydantic_core import PydanticUndefined
        from typing import get_origin, get_args
        from datetime import datetime
        
        # Get the schema fields
        schema_fields = self.output_schema.model_fields
        dummy_data = {}
        
        for field_name, field_info in schema_fields.items():
            # Check if field has a default value (but not PydanticUndefined)
            if (field_info.default is not None and
                field_info.default != PydanticUndefined and
                field_info.default != field_info.default_factory):
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
                    field_type = next((arg for arg in args if arg is not type(None)), str)
                    origin = get_origin(field_type)
                    args = get_args(field_type)
                
                if origin is list or field_type is list:
                    dummy_data[field_name] = [f"dummy_{field_name}_item"]
                elif origin is dict or field_type is dict:
                    dummy_data[field_name] = {f"dummy_{field_name}_key": f"dummy_{field_name}_value"}
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