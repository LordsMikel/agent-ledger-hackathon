"""Shared manifest, JSONL, and Cloud Storage helpers for invoice batches.

Author: Miguel Medina Cantos
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from config.settings import ConfigurationError, Settings
from src.agents.extractor.agent import (
    EXTRACTION_PROMPT,
    discover_images,
    sha256_file,
    validate_image,
)


EXPECTED_IMAGE_COUNT = 112
IMAGES_PER_BATCH = 10
MAX_BATCH_BYTES = 10 * 1024 * 1024
MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """Stable identity and integrity metadata for one invoice image."""

    source_path: str
    image_identifier: str
    content_hash: str
    size_bytes: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PreparedImage":
        return cls(
            source_path=str(value["source_path"]),
            image_identifier=str(value["image_identifier"]),
            content_hash=str(value["content_hash"]),
            size_bytes=int(value["size_bytes"]),
        )


@dataclass(frozen=True, slots=True)
class PreparedBatch:
    """One ten-image-or-smaller JSONL input prepared for Vertex Batch."""

    batch_number: int
    image_count: int
    total_image_bytes: int
    local_jsonl: str
    gcs_input_uri: str
    gcs_output_prefix: str
    images: tuple[PreparedImage, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PreparedBatch":
        return cls(
            batch_number=int(value["batch_number"]),
            image_count=int(value["image_count"]),
            total_image_bytes=int(value["total_image_bytes"]),
            local_jsonl=str(value["local_jsonl"]),
            gcs_input_uri=str(value["gcs_input_uri"]),
            gcs_output_prefix=str(value["gcs_output_prefix"]),
            images=tuple(
                PreparedImage.from_dict(image) for image in value.get("images", [])
            ),
        )


@dataclass(frozen=True, slots=True)
class BatchManifest:
    """Contract handed from JSONL preparation to Firestore indexing."""

    schema_version: int
    dataset_id: str
    created_at: str
    project_id: str
    location: str
    model: str
    input_directory: str
    gcs_base_uri: str
    image_count: int
    total_image_bytes: int
    images_per_batch: int
    max_batch_bytes: int
    batches: tuple[PreparedBatch, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BatchManifest":
        return cls(
            schema_version=int(value["schema_version"]),
            dataset_id=str(value["dataset_id"]),
            created_at=str(value["created_at"]),
            project_id=str(value["project_id"]),
            location=str(value["location"]),
            model=str(value["model"]),
            input_directory=str(value["input_directory"]),
            gcs_base_uri=str(value["gcs_base_uri"]),
            image_count=int(value["image_count"]),
            total_image_bytes=int(value["total_image_bytes"]),
            images_per_batch=int(value["images_per_batch"]),
            max_batch_bytes=int(value["max_batch_bytes"]),
            batches=tuple(
                PreparedBatch.from_dict(batch) for batch in value.get("batches", [])
            ),
        )


def manifest_path(settings: Settings) -> Path:
    """Return the stable handoff path used by both application steps."""

    return settings.output_dir / "vertex_batches" / "manifest.json"


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Split one gs:// URI into bucket and normalized object prefix."""

    if not uri.startswith("gs://"):
        raise ConfigurationError(f"Cloud Storage URI must start with gs://: {uri!r}")
    remainder = uri[5:]
    bucket_name, separator, object_name = remainder.partition("/")
    if not bucket_name:
        raise ConfigurationError(f"Cloud Storage URI is missing a bucket name: {uri!r}")
    return bucket_name, object_name.strip("/") if separator else ""


def join_gcs_uri(base_uri: str, *parts: str) -> str:
    """Join path components without changing the Cloud Storage bucket."""

    bucket_name, base_prefix = parse_gcs_uri(base_uri)
    components = [base_prefix, *(part.strip("/") for part in parts)]
    object_name = "/".join(component for component in components if component)
    return f"gs://{bucket_name}/{object_name}" if object_name else f"gs://{bucket_name}"


def discover_exact_input(settings: Settings) -> list[Path]:
    """Return exactly the prepared 112-image project dataset in sorted order."""

    expected_input = (settings.app_root / "input").resolve()
    if settings.input_dir.resolve() != expected_input:
        raise ConfigurationError(
            f"Refusing to read images outside {expected_input}; configured input is "
            f"{settings.input_dir.resolve()}."
        )
    paths = list(discover_images(settings.input_dir))
    if len(paths) != EXPECTED_IMAGE_COUNT:
        raise ConfigurationError(
            f"Expected exactly {EXPECTED_IMAGE_COUNT} project invoice images in "
            f"{settings.input_dir}; found {len(paths)}."
        )
    return paths


def partition_images(paths: Sequence[Path]) -> list[tuple[Path, ...]]:
    """Partition all images by count with a defensive raw-byte ceiling."""

    batches: list[tuple[Path, ...]] = []
    current: list[Path] = []
    current_bytes = 0
    for path in paths:
        image_bytes = path.stat().st_size
        if image_bytes > MAX_BATCH_BYTES:
            raise ConfigurationError(
                f"Image {path.name} is {image_bytes} bytes and cannot fit in the "
                f"{MAX_BATCH_BYTES}-byte safety cap."
            )
        if current and (
            len(current) >= IMAGES_PER_BATCH
            or current_bytes + image_bytes > MAX_BATCH_BYTES
        ):
            batches.append(tuple(current))
            current = []
            current_bytes = 0
        current.append(path)
        current_bytes += image_bytes
    if current:
        batches.append(tuple(current))
    return batches


def prepare_image_metadata(settings: Settings, paths: Iterable[Path]) -> list[PreparedImage]:
    """Validate and hash images once before writing any JSONL or GCS object."""

    images: list[PreparedImage] = []
    for path in paths:
        validate_image(path, max_bytes=settings.max_image_bytes)
        images.append(
            PreparedImage(
                source_path=path.relative_to(settings.app_root).as_posix(),
                image_identifier=path.name,
                content_hash=sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    return images


def make_dataset_id(settings: Settings, images: Sequence[PreparedImage]) -> str:
    """Create a deterministic dataset ID from content and extraction semantics."""

    payload = {
        "model": settings.gemini_model,
        "prompt": EXTRACTION_PROMPT,
        "images": [
            {
                "source_path": image.source_path,
                "content_hash": image.content_hash,
            }
            for image in images
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def write_batch_jsonl(
    settings: Settings,
    images: Sequence[PreparedImage],
    destination: Path,
) -> None:
    """Write Vertex GenerateContent requests with a stable key per invoice."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for image in images:
            image_path = settings.app_root / image.source_path
            mime_type = validate_image(image_path, max_bytes=settings.max_image_bytes)
            encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
            record = {
                "key": image.source_path,
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": mime_type,
                                        "data": encoded_image,
                                    }
                                },
                                {"text": EXTRACTION_PROMPT},
                            ],
                        }
                    ],
                    "generationConfig": {"responseMimeType": "application/json"},
                },
            }
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    temporary.replace(destination)


def write_manifest(manifest: BatchManifest, destination: Path) -> None:
    """Atomically publish the local handoff manifest."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest.to_dict(), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(destination)


def load_manifest(path: Path) -> BatchManifest:
    """Load and minimally validate the prepared batch handoff contract."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Batch manifest does not exist: {path}. Run convert_batch.main first."
        )
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Batch manifest must contain one JSON object: {path}")
    manifest = BatchManifest.from_dict(value)
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        raise ConfigurationError(
            f"Unsupported manifest schema {manifest.schema_version}; expected "
            f"{MANIFEST_SCHEMA_VERSION}."
        )
    return manifest


def build_manifest(
    settings: Settings,
    *,
    images: Sequence[PreparedImage],
    grouped_images: Sequence[Sequence[PreparedImage]],
    dataset_id: str,
) -> BatchManifest:
    """Create the complete immutable contract before any upload starts."""

    local_root = settings.output_dir / "vertex_batches" / dataset_id
    batches: list[PreparedBatch] = []
    for batch_number, batch_images in enumerate(grouped_images, start=1):
        batch_name = f"batch-{batch_number:03d}"
        local_jsonl = local_root / "inputs" / f"{batch_name}.jsonl"
        batches.append(
            PreparedBatch(
                batch_number=batch_number,
                image_count=len(batch_images),
                total_image_bytes=sum(image.size_bytes for image in batch_images),
                local_jsonl=local_jsonl.relative_to(settings.app_root).as_posix(),
                gcs_input_uri=join_gcs_uri(
                    settings.gemini_batch_gcs_uri,
                    dataset_id,
                    "inputs",
                    f"{batch_name}.jsonl",
                ),
                gcs_output_prefix=join_gcs_uri(
                    settings.gemini_batch_gcs_uri,
                    dataset_id,
                    "outputs",
                    batch_name,
                ),
                images=tuple(batch_images),
            )
        )
    return BatchManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        dataset_id=dataset_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        project_id=settings.firestore_project_id or "",
        location=settings.google_cloud_location,
        model=settings.gemini_model,
        input_directory=str(settings.input_dir),
        gcs_base_uri=settings.gemini_batch_gcs_uri,
        image_count=len(images),
        total_image_bytes=sum(image.size_bytes for image in images),
        images_per_batch=IMAGES_PER_BATCH,
        max_batch_bytes=MAX_BATCH_BYTES,
        batches=tuple(batches),
    )
