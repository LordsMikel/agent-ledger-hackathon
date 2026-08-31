"""Gemini multimodal invoice extraction and image discovery.

Author: Miguel Medina Cantos
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import threading
import time
from time import perf_counter
from typing import Any, Iterator, Mapping, Protocol, Sequence
import uuid

from config.settings import Settings, configure_vertex_ai_environment, get_logger


SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"})
EXTRACTION_PROMPT = """Analyze this invoice image and return one JSON object only.
Use exactly these keys: supplier_name, invoice_number, invoice_date, currency,
subtotal, tax, total, raw_text. Use null when a field cannot be read. Monetary
values must be strings preserving their printed decimal precision. invoice_date
should use YYYY-MM-DD when the image contains enough information. raw_text must
contain concise searchable invoice text, not commentary. Never infer financial
values that are not visible in the image.
"""


class ImageValidationError(ValueError):
    """Raised when an input path is missing, unsupported, or not a valid image."""


class ExtractionError(RuntimeError):
    """Raised when Gemini cannot return a valid structured invoice."""


class InvoiceExtractor(Protocol):
    """Port implemented by multimodal invoice extractors."""

    @property
    def model_name(self) -> str:
        """Return the extraction model identifier."""

    def extract(self, image_path: Path) -> "InvoiceData":
        """Extract structured invoice data from one validated image."""

    def extract_batch(self, image_paths: Sequence[Path]) -> list["ExtractionOutcome"]:
        """Extract a bounded group of images with the official Gemini Batch API."""


@dataclass(frozen=True, slots=True)
class InvoiceData:
    """Structured fields extracted from an invoice image."""

    supplier_name: str | None = None
    invoice_number: str | None = None
    invoice_date: str | None = None
    currency: str | None = None
    subtotal: str | None = None
    tax: str | None = None
    total: str | None = None
    raw_text: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InvoiceData":
        """Build an invoice while discarding unknown model response keys."""

        field_names = cls.__dataclass_fields__.keys()
        normalized: dict[str, str | None] = {}
        for field_name in field_names:
            raw_value = value.get(field_name)
            normalized[field_name] = None if raw_value is None else str(raw_value).strip() or None
        invoice = cls(**normalized)
        if not invoice.search_text:
            raise ExtractionError("Gemini returned an invoice without searchable content.")
        return invoice

    @property
    def search_text(self) -> str:
        """Return stable text used by the multilingual embedding model."""

        labelled_values = (
            ("Supplier", self.supplier_name),
            ("Invoice", self.invoice_number),
            ("Date", self.invoice_date),
            ("Currency", self.currency),
            ("Subtotal", self.subtotal),
            ("Tax", self.tax),
            ("Total", self.total),
            ("Text", self.raw_text),
        )
        return "\n".join(f"{label}: {value}" for label, value in labelled_values if value)

    def to_dict(self) -> dict[str, str | None]:
        """Return JSON-serializable invoice fields."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExtractionOutcome:
    """Successful invoice data or a recoverable error for one batch item."""

    image_path: Path
    invoice: InvoiceData | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Return whether this item contains extracted invoice data."""

        return self.invoice is not None and self.error is None


def discover_images(input_dir: Path) -> Iterator[Path]:
    """Yield supported image files in deterministic order without reading content."""

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Image input directory does not exist: {input_dir}")
    with os.scandir(input_dir) as entries:
        paths = sorted(
            (
                Path(entry.path)
                for entry in entries
                if entry.is_file() and Path(entry.name).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ),
            key=lambda path: path.name.casefold(),
        )
    yield from paths


def validate_image(image_path: Path, *, max_bytes: int) -> str:
    """Validate one image with Pillow and return its trusted MIME type."""

    if not image_path.is_file():
        raise ImageValidationError(f"Image file does not exist: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ImageValidationError(f"Unsupported image extension: {image_path.suffix}")
    size = image_path.stat().st_size
    if size <= 0:
        raise ImageValidationError(f"Image file is empty: {image_path}")
    if size > max_bytes:
        raise ImageValidationError(
            f"Image exceeds the configured {max_bytes}-byte inline request limit: {image_path}"
        )
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as error:
        raise RuntimeError("Pillow is required to validate invoice images.") from error
    try:
        with Image.open(image_path) as image:
            image.verify()
            detected_format = image.format
    except (OSError, UnidentifiedImageError) as error:
        raise ImageValidationError(f"Invalid image file: {image_path}") from error
    mime_type = Image.MIME.get(detected_format or "") or mimetypes.guess_type(image_path.name)[0]
    if not mime_type or not mime_type.startswith("image/"):
        raise ImageValidationError(f"Could not determine image MIME type: {image_path}")
    return mime_type


def sha256_file(image_path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash one file incrementally so image content is never duplicated in memory."""

    digest = hashlib.sha256()
    with image_path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class GeminiInvoiceExtractor:
    """Extract invoices with one lazily initialized Google Gen AI client."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any | None = None
        self._client_lock = threading.Lock()
        self._logger = get_logger()

    @property
    def model_name(self) -> str:
        """Return the configured Gemini model identifier."""

        return self._settings.gemini_model

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                try:
                    from google import genai
                except ImportError as error:
                    raise RuntimeError(
                        "google-genai is required for multimodal invoice extraction."
                    ) from error
                self._client = genai.Client(
                    vertexai=True,
                    project=self._settings.firestore_project_id,
                    location=self._settings.google_cloud_location,
                )
        return self._client

    def extract(self, image_path: Path) -> InvoiceData:
        """Validate and extract one image using Gemini structured JSON output."""

        mime_type = validate_image(image_path, max_bytes=self._settings.max_image_bytes)
        try:
            from google.genai import types
        except ImportError as error:
            raise RuntimeError("google-genai is required for invoice extraction.") from error
        image_bytes = image_path.read_bytes()
        try:
            response = self._get_client().models.generate_content(
                model=self._settings.gemini_model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    EXTRACTION_PROMPT,
                ],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            response_text = response.text
        except Exception as error:
            raise ExtractionError(
                f"Gemini extraction failed for {image_path.name}: {error}"
            ) from error
        return self._parse_invoice_response(response_text, image_path)

    def extract_batch(self, image_paths: Sequence[Path]) -> list[ExtractionOutcome]:
        """Extract one official Gemini batch containing at most 10 MiB of images."""

        paths = list(image_paths)
        if not paths:
            return []
        total_bytes = sum(path.stat().st_size for path in paths)
        if len(paths) > self._settings.gemini_batch_size:
            raise ExtractionError(
                f"Gemini batch contains {len(paths)} images; maximum is "
                f"{self._settings.gemini_batch_size}."
            )
        if total_bytes > self._settings.gemini_batch_max_bytes:
            raise ExtractionError(
                f"Gemini batch contains {total_bytes} image bytes; maximum is "
                f"{self._settings.gemini_batch_max_bytes}."
            )
        if not self._settings.gemini_batch_enabled:
            return [self._extract_individual(path) for path in paths]
        try:
            return self._extract_official_batch(paths)
        except Exception as error:
            if self._settings.gemini_batch_fallback:
                return [self._extract_individual(path) for path in paths]
            return [ExtractionOutcome(image_path=path, error=str(error)) for path in paths]

    def _extract_official_batch(self, paths: Sequence[Path]) -> list[ExtractionOutcome]:
        preparation_started = perf_counter()
        requests: list[dict[str, Any]] = []
        for path in paths:
            mime_type = validate_image(path, max_bytes=self._settings.max_image_bytes)
            encoded_image = base64.b64encode(path.read_bytes()).decode("ascii")
            requests.append(
                {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "inline_data": {
                                        "mime_type": mime_type,
                                        "data": encoded_image,
                                    }
                                },
                                {"text": EXTRACTION_PROMPT},
                            ],
                        }
                    ],
                    "config": {"response_mime_type": "application/json"},
                }
            )

        self._logger.info(
            f"Prepared {len(paths)} Gemini Batch requests in "
            f"{perf_counter() - preparation_started:.2f}s."
        )
        submission_started = perf_counter()
        job = self._get_client().batches.create(
            model=self._settings.gemini_model,
            src=requests,
            config={"display_name": f"invoice-extraction-{uuid.uuid4().hex[:12]}"},
        )
        self._logger.info(
            f"Submitted Gemini Batch job {job.name} in "
            f"{perf_counter() - submission_started:.2f}s."
        )
        wait_started = perf_counter()
        completed_job = self._wait_for_batch(job)
        self._logger.info(
            f"Gemini Batch job {job.name} completed in "
            f"{perf_counter() - wait_started:.2f}s."
        )
        parsing_started = perf_counter()
        responses = list(getattr(completed_job.dest, "inlined_responses", None) or [])
        if len(responses) != len(paths):
            raise ExtractionError(
                f"Gemini batch returned {len(responses)} responses for {len(paths)} images."
            )

        outcomes: list[ExtractionOutcome] = []
        for path, inline_response in zip(paths, responses, strict=True):
            response_error = getattr(inline_response, "error", None)
            response = getattr(inline_response, "response", None)
            if response_error or response is None:
                if self._settings.gemini_batch_fallback:
                    outcomes.append(self._extract_individual(path))
                else:
                    outcomes.append(
                        ExtractionOutcome(
                            image_path=path,
                            error=f"Gemini batch item failed: {response_error}",
                        )
                    )
                continue
            try:
                response_text = self._response_text(response)
                outcomes.append(
                    ExtractionOutcome(
                        image_path=path,
                        invoice=self._parse_invoice_response(response_text, path),
                    )
                )
            except Exception as error:
                if self._settings.gemini_batch_fallback:
                    outcomes.append(self._extract_individual(path))
                else:
                    outcomes.append(ExtractionOutcome(image_path=path, error=str(error)))
        self._logger.info(
            f"Parsed {len(responses)} Gemini Batch responses in "
            f"{perf_counter() - parsing_started:.2f}s."
        )
        return outcomes

    def _wait_for_batch(self, job: Any) -> Any:
        terminal_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }
        deadline = time.monotonic() + self._settings.gemini_batch_timeout_seconds
        current_job = job
        last_state: str | None = None
        while self._job_state(current_job) not in terminal_states:
            state = self._job_state(current_job)
            if state != last_state:
                self._logger.info(f"Gemini Batch job {job.name} state={state}.")
                last_state = state
            if time.monotonic() >= deadline:
                raise ExtractionError(f"Gemini batch job timed out: {job.name}")
            time.sleep(self._settings.gemini_batch_poll_seconds)
            current_job = self._get_client().batches.get(name=job.name)
        state = self._job_state(current_job)
        if state != last_state:
            self._logger.info(f"Gemini Batch job {job.name} state={state}.")
        if state != "JOB_STATE_SUCCEEDED":
            raise ExtractionError(
                f"Gemini batch job ended in {state}: {getattr(current_job, 'error', None)}"
            )
        return current_job

    @staticmethod
    def _job_state(job: Any) -> str:
        state = getattr(job, "state", None)
        return str(getattr(state, "name", state))

    @staticmethod
    def _response_text(response: Any) -> str | None:
        direct_text = getattr(response, "text", None)
        if direct_text:
            return str(direct_text)
        text_parts: list[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                if getattr(part, "text", None):
                    text_parts.append(str(part.text))
        return "".join(text_parts) or None

    def _extract_individual(self, image_path: Path) -> ExtractionOutcome:
        try:
            return ExtractionOutcome(image_path=image_path, invoice=self.extract(image_path))
        except Exception as error:
            return ExtractionOutcome(image_path=image_path, error=str(error))

    @staticmethod
    def _parse_invoice_response(response_text: str | None, image_path: Path) -> InvoiceData:
        if not response_text:
            raise ExtractionError(f"Gemini returned an empty response for {image_path.name}.")
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError as error:
            raise ExtractionError(f"Gemini returned invalid JSON for {image_path.name}.") from error
        if not isinstance(payload, dict):
            raise ExtractionError(f"Gemini response must be a JSON object for {image_path.name}.")
        return InvoiceData.from_mapping(payload)


def build_adk_agent(extractor: InvoiceExtractor, *, model_name: str) -> Any:
    """Build the specialized Google ADK extraction agent."""

    configure_vertex_ai_environment()
    try:
        from google.adk.agents import Agent
    except ImportError as error:
        raise RuntimeError("google-adk is required to build the extraction agent.") from error

    def extract_invoice(image_path: str) -> dict[str, str | None]:
        """Extract structured financial fields from one local invoice image."""

        return extractor.extract(Path(image_path)).to_dict()

    return Agent(
        name="invoice_extractor",
        model=model_name,
        description="Reads invoice images and returns grounded structured data.",
        instruction=(
            "Extract only information visible in the supplied invoice image. "
            "Use the extract_invoice tool and never invent missing financial values."
        ),
        tools=[extract_invoice],
    )
