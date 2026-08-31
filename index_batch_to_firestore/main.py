"""Run prepared Vertex batches and persist extracted invoices in Firestore.

The preparation step owns image-to-JSONL conversion and Cloud Storage uploads.
This step owns Vertex Batch execution, output parsing, MiniLM embeddings, and
idempotent Firestore writes.

Author: Miguel Medina Cantos
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

from loguru import logger

from config.settings import ConfigurationError, Settings
from convert_batch.utils import (
    EXPECTED_IMAGE_COUNT,
    IMAGES_PER_BATCH,
    MAX_BATCH_BYTES,
    BatchManifest,
    PreparedBatch,
    PreparedImage,
    discover_exact_input,
    join_gcs_uri,
    load_manifest,
    manifest_path,
)
from index_batch_to_firestore.utils import VertexGcsBatchExecutor
from src.agents.extractor.agent import InvoiceData, sha256_file, validate_image
from src.embeddings.vector_index import (
    FirestoreVectorIndex,
    IndexStatus,
    VectorDocument,
    make_document_id,
)
from src.embeddings.vector_search import SentenceTransformerEmbeddingService


EXPECTED_FIRESTORE_DATABASE = "agent-test1-100"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the prepared 112-invoice Vertex batches ten images at a time, "
            "embed successful extractions, and upsert them into Firestore."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Prepared manifest path. Defaults to "
            "output/vertex_batches/manifest.json."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess and overwrite invoices that are already current.",
    )
    return parser


def _configure_logging(settings: Settings) -> Path:
    log_directory = settings.output_dir / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    log_path = log_directory / f"index-batch-{timestamp}.log"
    logger.add(
        log_path,
        level=settings.log_level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS ZZ} | {elapsed} | {level:<8} | "
            "{message}"
        ),
        encoding="utf-8",
    )
    return log_path


def _validate_settings(settings: Settings) -> None:
    if not settings.insert_into_index:
        raise ConfigurationError(
            "Batch indexing is disabled. Set INSERT_INTO_INDEX=true explicitly."
        )
    settings.validate(require_cloud=True)
    if settings.firestore_database != EXPECTED_FIRESTORE_DATABASE:
        raise ConfigurationError(
            f"This indexer requires FIRESTORE_DATABASE={EXPECTED_FIRESTORE_DATABASE}; "
            f"received {settings.firestore_database!r}."
        )


def _validate_manifest(settings: Settings, manifest: BatchManifest) -> None:
    if manifest.project_id != settings.firestore_project_id:
        raise ConfigurationError(
            "Prepared manifest project does not match GOOGLE_CLOUD_PROJECT: "
            f"{manifest.project_id!r} != {settings.firestore_project_id!r}."
        )
    if manifest.location != settings.google_cloud_location:
        raise ConfigurationError(
            "Prepared manifest location does not match GOOGLE_CLOUD_LOCATION: "
            f"{manifest.location!r} != {settings.google_cloud_location!r}."
        )
    if manifest.model != settings.gemini_model:
        raise ConfigurationError(
            "Prepared manifest model does not match GEMINI_MODEL: "
            f"{manifest.model!r} != {settings.gemini_model!r}."
        )
    if manifest.gcs_base_uri.rstrip("/") != settings.gemini_batch_gcs_uri.rstrip("/"):
        raise ConfigurationError(
            "Prepared manifest bucket prefix does not match GEMINI_BATCH_GCS_URI."
        )
    if manifest.image_count != EXPECTED_IMAGE_COUNT:
        raise ConfigurationError(
            f"Manifest contains {manifest.image_count} images; expected exactly "
            f"{EXPECTED_IMAGE_COUNT}."
        )
    if manifest.images_per_batch != IMAGES_PER_BATCH:
        raise ConfigurationError(
            f"Manifest batch size limit is {manifest.images_per_batch}; expected "
            f"{IMAGES_PER_BATCH}."
        )
    if manifest.max_batch_bytes != MAX_BATCH_BYTES:
        raise ConfigurationError(
            f"Manifest byte limit is {manifest.max_batch_bytes}; expected "
            f"{MAX_BATCH_BYTES}."
        )
    expected_batch_count = (EXPECTED_IMAGE_COUNT + IMAGES_PER_BATCH - 1) // IMAGES_PER_BATCH
    if len(manifest.batches) != expected_batch_count:
        raise ConfigurationError(
            f"Manifest contains {len(manifest.batches)} batches; expected "
            f"{expected_batch_count}."
        )
    flattened = [image for batch in manifest.batches for image in batch.images]
    if len(flattened) != EXPECTED_IMAGE_COUNT:
        raise ConfigurationError(
            f"Manifest batch contents contain {len(flattened)} images; expected "
            f"{EXPECTED_IMAGE_COUNT}."
        )
    if len({image.source_path for image in flattened}) != EXPECTED_IMAGE_COUNT:
        raise ConfigurationError("Manifest contains duplicate invoice source paths.")
    expected_sizes = [IMAGES_PER_BATCH] * (expected_batch_count - 1)
    expected_sizes.append(EXPECTED_IMAGE_COUNT % IMAGES_PER_BATCH or IMAGES_PER_BATCH)
    actual_sizes = [batch.image_count for batch in manifest.batches]
    if actual_sizes != expected_sizes:
        raise ConfigurationError(
            f"Manifest batch sizes are {actual_sizes}; expected {expected_sizes}."
        )
    for batch_number, batch in enumerate(manifest.batches, start=1):
        if batch.batch_number != batch_number:
            raise ConfigurationError(
                f"Manifest batch position {batch_number} has number "
                f"{batch.batch_number}."
            )
        if batch.image_count != len(batch.images):
            raise ConfigurationError(
                f"Batch {batch.batch_number} declares {batch.image_count} images but "
                f"contains {len(batch.images)}."
            )
        actual_batch_bytes = sum(image.size_bytes for image in batch.images)
        if batch.total_image_bytes != actual_batch_bytes:
            raise ConfigurationError(
                f"Batch {batch.batch_number} byte total does not match its images."
            )
        if batch.total_image_bytes > MAX_BATCH_BYTES:
            raise ConfigurationError(
                f"Batch {batch.batch_number} exceeds the {MAX_BATCH_BYTES}-byte limit."
            )
        batch_name = f"batch-{batch.batch_number:03d}"
        expected_input_uri = join_gcs_uri(
            manifest.gcs_base_uri,
            manifest.dataset_id,
            "inputs",
            f"{batch_name}.jsonl",
        )
        expected_output_prefix = join_gcs_uri(
            manifest.gcs_base_uri,
            manifest.dataset_id,
            "outputs",
            batch_name,
        )
        if batch.gcs_input_uri != expected_input_uri:
            raise ConfigurationError(
                f"Batch {batch.batch_number} has an unexpected GCS input URI."
            )
        if batch.gcs_output_prefix != expected_output_prefix:
            raise ConfigurationError(
                f"Batch {batch.batch_number} has an unexpected GCS output prefix."
            )
    if manifest.total_image_bytes != sum(image.size_bytes for image in flattened):
        raise ConfigurationError("Manifest byte total does not match its images.")


def _verify_local_images(
    settings: Settings,
    manifest: BatchManifest,
) -> dict[str, PreparedImage]:
    verification_started = perf_counter()
    discovered = discover_exact_input(settings)
    discovered_sources = {
        path.relative_to(settings.app_root).as_posix() for path in discovered
    }
    manifest_images = {
        image.source_path: image
        for batch in manifest.batches
        for image in batch.images
    }
    if set(manifest_images) != discovered_sources:
        raise ConfigurationError(
            "Prepared manifest sources do not exactly match the 112 project images. "
            "Run convert_batch.main again."
        )
    for source_path, image in manifest_images.items():
        image_path = settings.app_root / source_path
        validate_image(image_path, max_bytes=settings.max_image_bytes)
        actual_size = image_path.stat().st_size
        if actual_size != image.size_bytes:
            raise ConfigurationError(
                f"Image size changed after batch preparation: {source_path}. "
                "Run convert_batch.main again."
            )
        if sha256_file(image_path) != image.content_hash:
            raise ConfigurationError(
                f"Image content changed after batch preparation: {source_path}. "
                "Run convert_batch.main again."
            )
    logger.info(
        f"Verified {len(manifest_images)} local images against the manifest in "
        f"{perf_counter() - verification_started:.2f}s."
    )
    return manifest_images


def _is_current(
    status: IndexStatus | None,
    image: PreparedImage,
    settings: Settings,
) -> bool:
    return bool(
        status
        and status.processing_status == "processed"
        and status.content_hash == image.content_hash
        and status.embedding_model == settings.embedding_model
        and status.extraction_model == settings.gemini_model
    )


def _write_extraction(
    settings: Settings,
    image: PreparedImage,
    invoice: InvoiceData,
) -> None:
    document_id = make_document_id(image.source_path)
    destination = settings.output_dir / "extracted" / f"{document_id}.json"
    temporary = destination.with_suffix(".json.tmp")
    payload = {
        "image_identifier": image.image_identifier,
        "source_path": image.source_path,
        "content_hash": image.content_hash,
        "extraction_model": settings.gemini_model,
        "invoice": invoice.to_dict(),
        "search_text": invoice.search_text,
    }
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(destination)


def _build_documents(
    *,
    settings: Settings,
    images: Sequence[PreparedImage],
    invoices: dict[str, InvoiceData],
    vectors: Sequence[list[float]],
) -> list[VectorDocument]:
    return [
        VectorDocument(
            document_id=make_document_id(image.source_path),
            image_identifier=image.image_identifier,
            source_path=image.source_path,
            content_hash=image.content_hash,
            extraction_model=settings.gemini_model,
            embedding_model=settings.embedding_model,
            embedding=vector,
            invoice=invoices[image.source_path].to_dict(),
            search_text=invoices[image.source_path].search_text,
        )
        for image, vector in zip(images, vectors, strict=True)
    ]


def _index_batch(
    *,
    settings: Settings,
    batch: PreparedBatch,
    pending: Sequence[PreparedImage],
    executor: VertexGcsBatchExecutor,
    embeddings: SentenceTransformerEmbeddingService,
    vector_index: FirestoreVectorIndex,
) -> tuple[int, int, str, str]:
    result = executor.execute(batch)
    pending_sources = {image.source_path for image in pending}
    failed = {
        source_path: message
        for source_path, message in result.errors.items()
        if source_path in pending_sources
    }
    successful_images = [
        image
        for image in pending
        if image.source_path in result.invoices and image.source_path not in failed
    ]
    for image in pending:
        if image.source_path not in result.invoices and image.source_path not in failed:
            failed[image.source_path] = "Vertex Batch returned no usable invoice."

    for image in successful_images:
        _write_extraction(settings, image, result.invoices[image.source_path])

    if successful_images:
        embedding_started = perf_counter()
        vectors = embeddings.embed_texts(
            [result.invoices[image.source_path].search_text for image in successful_images]
        )
        logger.info(
            f"Batch {batch.batch_number} MiniLM embeddings finished in "
            f"{perf_counter() - embedding_started:.2f}s for "
            f"{len(successful_images)} invoices."
        )
        documents = _build_documents(
            settings=settings,
            images=successful_images,
            invoices=dict(result.invoices),
            vectors=vectors,
        )
        firestore_started = perf_counter()
        vector_index.upsert_many(documents)
        logger.info(
            f"Batch {batch.batch_number} Firestore upsert finished in "
            f"{perf_counter() - firestore_started:.2f}s for {len(documents)} invoices."
        )

    for source_path, message in sorted(failed.items()):
        logger.error(
            f"Batch {batch.batch_number} failed item {source_path}: {message}"
        )
    return (
        len(successful_images),
        len(failed),
        result.job_name,
        result.output_directory,
    )


def run(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    settings = Settings.from_env()
    _validate_settings(settings)
    settings.prepare_output()
    log_path = _configure_logging(settings)
    logger.info(f"Persistent Loguru output: {log_path}.")
    active_manifest_path = (
        arguments.manifest.resolve()
        if arguments.manifest is not None
        else manifest_path(settings)
    )
    manifest = load_manifest(active_manifest_path)
    _validate_manifest(settings, manifest)
    manifest_images = _verify_local_images(settings, manifest)
    operation_started = perf_counter()

    vector_index = FirestoreVectorIndex(settings)
    lookup_started = perf_counter()
    document_ids = {
        source_path: make_document_id(source_path) for source_path in manifest_images
    }
    statuses = vector_index.lookup_statuses(list(document_ids.values()))
    logger.info(
        f"Firestore currentness preflight finished in "
        f"{perf_counter() - lookup_started:.2f}s for {len(document_ids)} invoices."
    )
    current_sources = {
        source_path
        for source_path, image in manifest_images.items()
        if _is_current(statuses.get(document_ids[source_path]), image, settings)
    }
    if arguments.force:
        current_sources.clear()

    initial_skipped = len(current_sources)
    initial_pending = manifest.image_count - initial_skipped
    logger.info(
        f"Step 2 target | database={settings.firestore_database} | "
        f"collection={settings.firestore_collection} | current={initial_skipped} "
        f"pending={initial_pending}."
    )
    if initial_pending == 0:
        elapsed = perf_counter() - operation_started
        logger.success(
            f"All {manifest.image_count} invoices are already current. Vertex Batch, "
            f"MiniLM, and Firestore writes were skipped in {elapsed:.2f}s."
        )
        print(
            json.dumps(
                {
                    "status": "already_current",
                    "image_count": manifest.image_count,
                    "processed": 0,
                    "skipped": initial_skipped,
                    "failed": 0,
                    "submitted_jobs": 0,
                    "firestore_database": settings.firestore_database,
                    "elapsed_seconds": round(elapsed, 3),
                    "log_path": str(log_path),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    executor = VertexGcsBatchExecutor(settings, logger)
    embeddings = SentenceTransformerEmbeddingService(
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        device=settings.device,
        normalize=settings.normalize_embeddings,
    )
    processed = 0
    failed = 0
    submitted_jobs: list[str] = []
    output_directories: list[str] = []

    for batch in manifest.batches:
        batch_started = perf_counter()
        pending = [
            image for image in batch.images if image.source_path not in current_sources
        ]
        batch_skipped = batch.image_count - len(pending)
        if not pending:
            remaining = initial_pending - processed - failed
            logger.info(
                f"Batch {batch.batch_number}/{len(manifest.batches)} skipped | "
                f"already_current={batch_skipped} | remaining_images={remaining} "
                f"remaining_batches={len(manifest.batches) - batch.batch_number}."
            )
            continue

        try:
            batch_processed, batch_failed, job_name, output_directory = _index_batch(
                settings=settings,
                batch=batch,
                pending=pending,
                executor=executor,
                embeddings=embeddings,
                vector_index=vector_index,
            )
        except Exception as error:
            logger.error(
                f"Batch {batch.batch_number}/{len(manifest.batches)} aborted after "
                f"{perf_counter() - batch_started:.2f}s | error={error} | "
                f"remaining_images={initial_pending - processed}."
            )
            raise
        processed += batch_processed
        failed += batch_failed
        submitted_jobs.append(job_name)
        output_directories.append(output_directory)
        remaining = initial_pending - processed
        remaining_batches = len(manifest.batches) - batch.batch_number
        if batch_failed:
            remaining_batches += 1
        logger.info(
            f"Batch {batch.batch_number}/{len(manifest.batches)} finished in "
            f"{perf_counter() - batch_started:.2f}s | requested={len(pending)} "
            f"processed={batch_processed} skipped={batch_skipped} "
            f"failed={batch_failed} | total_processed={processed} "
            f"remaining_images={remaining} "
            f"remaining_batches={remaining_batches}."
        )
        if batch_failed:
            raise RuntimeError(
                f"Batch {batch.batch_number}/{len(manifest.batches)} had "
                f"{batch_failed} failed invoice(s). Successful invoices were saved; "
                "rerun without --force to retry only records that are not current."
            )

    elapsed = perf_counter() - operation_started
    logger.success(
        f"Step 2 completed in {elapsed:.2f}s | processed={processed} "
        f"skipped={initial_skipped} failed={failed} jobs={len(submitted_jobs)}."
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "dataset_id": manifest.dataset_id,
                "image_count": manifest.image_count,
                "processed": processed,
                "skipped": initial_skipped,
                "failed": failed,
                "submitted_jobs": len(submitted_jobs),
                "job_names": submitted_jobs,
                "vertex_output_directories": output_directories,
                "firestore_database": settings.firestore_database,
                "elapsed_seconds": round(elapsed, 3),
                "log_path": str(log_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except (ConfigurationError, FileNotFoundError, RuntimeError, ValueError) as error:
        logger.error(str(error))
        raise SystemExit(2) from error
    except KeyboardInterrupt:
        logger.info("Vertex Batch indexing interrupted by the user.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
