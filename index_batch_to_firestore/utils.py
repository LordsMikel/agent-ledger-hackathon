"""Vertex Batch execution and Cloud Storage output parsing.

Author: Miguel Medina Cantos
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import PurePosixPath
from time import monotonic, perf_counter, sleep
from typing import Any, Mapping

from config.settings import Settings
from convert_batch.utils import PreparedBatch, parse_gcs_uri
from src.agents.extractor.agent import ExtractionError, InvoiceData


@dataclass(frozen=True, slots=True)
class VertexBatchResult:
    """Parsed model outputs and item errors keyed by project-relative source path."""

    job_name: str
    output_directory: str
    invoices: Mapping[str, InvoiceData]
    errors: Mapping[str, str]


def _endpoint_for_location(location: str) -> str:
    if location == "global":
        return "aiplatform.googleapis.com"
    return f"{location}-aiplatform.googleapis.com"


def _publisher_model_name(model: str) -> str:
    if model.startswith(("projects/", "publishers/")):
        return model
    return f"publishers/google/models/{model}"


def _job_state_name(job: Any) -> str:
    try:
        from google.cloud import aiplatform_v1
    except ImportError as error:
        raise RuntimeError("google-cloud-aiplatform is required for Vertex Batch.") from error
    return aiplatform_v1.types.JobState(job.state).name


def _error_text(value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        code = value.get("code")
        message = value.get("message") or value.get("details")
        if code in (None, 0, "0") and not message:
            return None
        return f"code={code}: {message}" if code is not None else str(message)
    code = getattr(value, "code", None)
    message = getattr(value, "message", None) or getattr(value, "details", None)
    if code in (None, 0) and not message:
        return None
    return f"code={code}: {message}" if code is not None else str(message)


def _record_key(record: Mapping[str, Any]) -> str | None:
    direct = record.get("key")
    if direct:
        return str(direct)
    instance = record.get("instance")
    if isinstance(instance, Mapping):
        value = instance.get("key") or instance.get("source_path")
        if value:
            return str(value)
    value = record.get("source_path")
    return str(value) if value else None


def _response_payload(record: Mapping[str, Any]) -> Any:
    for field_name in ("response", "prediction"):
        value = record.get(field_name)
        if value is not None:
            if isinstance(value, Mapping) and value.get("response") is not None:
                return value["response"]
            return value
    return None


def _response_text(payload: Any) -> str | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, Mapping):
        return None
    direct = payload.get("text")
    if direct:
        return str(direct)
    text_parts: list[str] = []
    for candidate in payload.get("candidates") or []:
        if not isinstance(candidate, Mapping):
            continue
        content = candidate.get("content") or {}
        if not isinstance(content, Mapping):
            continue
        for part in content.get("parts") or []:
            if isinstance(part, Mapping) and part.get("text"):
                text_parts.append(str(part["text"]))
    return "".join(text_parts) or None


def _parse_invoice(payload: Any, source_path: str) -> InvoiceData:
    response_text = _response_text(payload)
    if not response_text:
        raise ExtractionError(f"Vertex Batch returned no response text for {source_path}.")
    try:
        value = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ExtractionError(
            f"Vertex Batch returned invalid invoice JSON for {source_path}."
        ) from error
    if not isinstance(value, dict):
        raise ExtractionError(
            f"Vertex Batch invoice response must be an object for {source_path}."
        )
    return InvoiceData.from_mapping(value)


class VertexGcsBatchExecutor:
    """Submit prepared JSONL to Vertex and parse its GCS prediction files."""

    def __init__(self, settings: Settings, logger: Any) -> None:
        self._settings = settings
        self._logger = logger
        self._job_client: Any | None = None
        self._storage_client: Any | None = None

    def _get_job_client(self):
        if self._job_client is not None:
            return self._job_client
        try:
            from google.api_core.client_options import ClientOptions
            from google.cloud import aiplatform_v1
        except ImportError as error:
            raise RuntimeError("google-cloud-aiplatform is required for Vertex Batch.") from error
        self._job_client = aiplatform_v1.JobServiceClient(
            client_options=ClientOptions(
                api_endpoint=_endpoint_for_location(self._settings.google_cloud_location)
            )
        )
        return self._job_client

    def _get_storage_client(self):
        if self._storage_client is not None:
            return self._storage_client
        try:
            from google.cloud import storage
        except ImportError as error:
            raise RuntimeError(
                "google-cloud-storage is required to read Vertex Batch results."
            ) from error
        self._storage_client = storage.Client(project=self._settings.firestore_project_id)
        return self._storage_client

    def execute(self, batch: PreparedBatch) -> VertexBatchResult:
        """Run one prepared batch and return results mapped through Vertex key_field."""

        try:
            from google.cloud import aiplatform_v1
            from google.cloud.aiplatform_v1.types import GcsDestination, GcsSource
        except ImportError as error:
            raise RuntimeError("google-cloud-aiplatform is required for Vertex Batch.") from error

        submission_started = perf_counter()
        job_spec = aiplatform_v1.types.BatchPredictionJob(
            display_name=(
                f"invoice-{batch.batch_number:03d}-"
                f"{PurePosixPath(batch.gcs_input_uri).stem[-20:]}"
            ),
            model=_publisher_model_name(self._settings.gemini_model),
            input_config=aiplatform_v1.types.BatchPredictionJob.InputConfig(
                instances_format="jsonl",
                gcs_source=GcsSource(uris=[batch.gcs_input_uri]),
            ),
            instance_config=aiplatform_v1.types.BatchPredictionJob.InstanceConfig(
                instance_type="object",
                key_field="key",
            ),
            output_config=aiplatform_v1.types.BatchPredictionJob.OutputConfig(
                predictions_format="jsonl",
                gcs_destination=GcsDestination(
                    output_uri_prefix=batch.gcs_output_prefix
                ),
            ),
        )
        parent = (
            f"projects/{self._settings.firestore_project_id}/locations/"
            f"{self._settings.google_cloud_location}"
        )
        job = self._get_job_client().create_batch_prediction_job(
            parent=parent,
            batch_prediction_job=job_spec,
        )
        self._logger.info(
            f"Submitted Vertex Batch job {job.name} in "
            f"{perf_counter() - submission_started:.2f}s."
        )
        completed_job = self._wait_for_job(job)
        output_directory = completed_job.output_info.gcs_output_directory
        if not output_directory:
            raise ExtractionError(
                f"Vertex Batch job {job.name} succeeded without a GCS output directory."
            )
        return self._read_results(batch, job.name, output_directory)

    def _wait_for_job(self, job: Any) -> Any:
        success_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}
        failure_states = {
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_PAUSED",
            "JOB_STATE_EXPIRED",
        }
        deadline = monotonic() + self._settings.gemini_batch_timeout_seconds
        started = monotonic()
        last_state: str | None = None
        last_heartbeat = 0.0
        current = job
        while True:
            state = _job_state_name(current)
            elapsed = monotonic() - started
            if state != last_state or elapsed - last_heartbeat >= 60:
                self._logger.info(
                    f"Vertex Batch job {job.name} state={state} elapsed={elapsed:.1f}s."
                )
                last_state = state
                last_heartbeat = elapsed
            if state in success_states:
                return current
            if state in failure_states:
                raise ExtractionError(
                    f"Vertex Batch job {job.name} ended in {state}: {current.error}"
                )
            if monotonic() >= deadline:
                raise ExtractionError(f"Vertex Batch job timed out: {job.name}")
            sleep(self._settings.gemini_batch_poll_seconds)
            current = self._get_job_client().get_batch_prediction_job(name=job.name)

    def _read_results(
        self,
        batch: PreparedBatch,
        job_name: str,
        output_directory: str,
    ) -> VertexBatchResult:
        parsing_started = perf_counter()
        bucket_name, prefix = parse_gcs_uri(output_directory)
        blobs = sorted(
            self._get_storage_client().list_blobs(bucket_name, prefix=prefix),
            key=lambda blob: blob.name,
        )
        prediction_blobs = [
            blob
            for blob in blobs
            if PurePosixPath(blob.name).name.startswith(("prediction", "predictions"))
            and blob.name.endswith(".jsonl")
        ]
        error_blobs = [
            blob
            for blob in blobs
            if PurePosixPath(blob.name).name.startswith(("error", "errors"))
            and blob.name.endswith(".jsonl")
        ]
        if not prediction_blobs and not error_blobs:
            raise ExtractionError(
                f"No Vertex prediction JSONL files found under {output_directory}."
            )

        records: list[Mapping[str, Any]] = []
        for blob in prediction_blobs:
            records.extend(self._download_records(blob))
        error_records: list[Mapping[str, Any]] = []
        for blob in error_blobs:
            error_records.extend(self._download_records(blob))

        expected_keys = [image.source_path for image in batch.images]
        keyed_records: dict[str, Mapping[str, Any]] = {}
        unkeyed_records: list[Mapping[str, Any]] = []
        for record in records:
            key = _record_key(record)
            if key:
                keyed_records[key] = record
            else:
                unkeyed_records.append(record)
        remaining_keys = [key for key in expected_keys if key not in keyed_records]
        if unkeyed_records:
            if len(unkeyed_records) != len(remaining_keys):
                raise ExtractionError(
                    f"Cannot map {len(unkeyed_records)} unkeyed Vertex responses to "
                    f"{len(remaining_keys)} remaining invoices."
                )
            self._logger.warning(
                "Vertex output omitted key_field; falling back to response order for this batch."
            )
            keyed_records.update(zip(remaining_keys, unkeyed_records, strict=True))

        invoices: dict[str, InvoiceData] = {}
        errors: dict[str, str] = {}
        for key, record in keyed_records.items():
            status_error = _error_text(record.get("status") or record.get("error"))
            if status_error:
                errors[key] = status_error
                continue
            try:
                invoices[key] = _parse_invoice(_response_payload(record), key)
            except Exception as error:
                errors[key] = str(error)

        for record in error_records:
            key = _record_key(record)
            if key:
                errors[key] = _error_text(record.get("error") or record.get("status")) or str(
                    record
                )

        for key in expected_keys:
            if key not in invoices and key not in errors:
                errors[key] = "Vertex Batch returned no prediction for this invoice."

        self._logger.info(
            f"Downloaded and parsed Vertex output in {perf_counter() - parsing_started:.2f}s | "
            f"predictions={len(invoices)} errors={len(errors)} files="
            f"{len(prediction_blobs) + len(error_blobs)}."
        )
        return VertexBatchResult(
            job_name=job_name,
            output_directory=output_directory,
            invoices=invoices,
            errors=errors,
        )

    @staticmethod
    def _download_records(blob: Any) -> list[Mapping[str, Any]]:
        records: list[Mapping[str, Any]] = []
        payload = blob.download_as_text(encoding="utf-8")
        for line_number, line in enumerate(payload.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ExtractionError(
                    f"Invalid JSONL in gs://{blob.bucket.name}/{blob.name} "
                    f"at line {line_number}."
                ) from error
            if not isinstance(value, dict):
                raise ExtractionError(
                    f"Vertex output line {line_number} in {blob.name} is not an object."
                )
            records.append(value)
        return records
