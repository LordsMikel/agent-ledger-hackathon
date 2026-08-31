"""Prepare and upload the complete 112-invoice Vertex Batch dataset.

This step never calls Gemini and never reads or writes Firestore. It produces
deterministic ten-image JSONL files plus the manifest consumed by step two.

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
    BatchManifest,
    build_manifest,
    discover_exact_input,
    make_dataset_id,
    manifest_path,
    parse_gcs_uri,
    partition_images,
    prepare_image_metadata,
    write_batch_jsonl,
    write_manifest,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert exactly 112 project invoice images into ten-image Vertex Batch "
            "JSONL inputs and upload them to Cloud Storage."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite JSONL objects that already exist in Cloud Storage.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Create local JSONL files and manifest without uploading to Cloud Storage.",
    )
    return parser


def _configure_logging(settings: Settings) -> Path:
    log_directory = settings.output_dir / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    log_path = log_directory / f"convert-batch-{timestamp}.log"
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


def _storage_client(settings: Settings):
    try:
        from google.cloud import storage
    except ImportError as error:
        raise RuntimeError("google-cloud-storage is required to upload batch JSONL.") from error
    return storage.Client(project=settings.firestore_project_id)


def _upload_jsonl(
    *,
    client,
    local_path: Path,
    gcs_uri: str,
    force: bool,
) -> bool:
    bucket_name, object_name = parse_gcs_uri(gcs_uri)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    if not force and blob.exists(client=client):
        return False
    blob.upload_from_filename(str(local_path), content_type="application/jsonl")
    return True


def _prepare_manifest(settings: Settings) -> BatchManifest:
    discovery_started = perf_counter()
    paths = discover_exact_input(settings)
    path_batches = partition_images(paths)
    images = prepare_image_metadata(settings, paths)
    images_by_path = {image.source_path: image for image in images}
    grouped_images = [
        [images_by_path[path.relative_to(settings.app_root).as_posix()] for path in batch]
        for batch in path_batches
    ]
    dataset_id = make_dataset_id(settings, images)
    manifest = build_manifest(
        settings,
        images=images,
        grouped_images=grouped_images,
        dataset_id=dataset_id,
    )
    logger.info(
        f"Validated {manifest.image_count} images "
        f"({manifest.total_image_bytes / 1024 / 1024:.2f} MiB) and prepared "
        f"{len(manifest.batches)} batch definitions in "
        f"{perf_counter() - discovery_started:.2f}s."
    )
    return manifest


def run(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    settings = Settings.from_env()
    settings.validate(require_cloud=True)
    settings.prepare_output()
    log_path = _configure_logging(settings)
    logger.info(f"Persistent Loguru output: {log_path}.")
    logger.info(
        f"Step 1 target | input={settings.input_dir} | "
        f"GCS={settings.gemini_batch_gcs_uri}."
    )

    operation_started = perf_counter()
    manifest = _prepare_manifest(settings)
    storage_client = None if arguments.local_only else _storage_client(settings)
    uploaded = 0
    reused = 0
    prepared_images = 0

    for batch in manifest.batches:
        batch_started = perf_counter()
        local_path = settings.app_root / batch.local_jsonl
        write_batch_jsonl(settings, batch.images, local_path)
        upload_status = "local-only"
        if storage_client is not None:
            if _upload_jsonl(
                client=storage_client,
                local_path=local_path,
                gcs_uri=batch.gcs_input_uri,
                force=arguments.force,
            ):
                uploaded += 1
                upload_status = "uploaded"
            else:
                reused += 1
                upload_status = "already-present"
        prepared_images += batch.image_count
        remaining_images = manifest.image_count - prepared_images
        remaining_batches = len(manifest.batches) - batch.batch_number
        logger.info(
            f"Batch {batch.batch_number}/{len(manifest.batches)} prepared in "
            f"{perf_counter() - batch_started:.2f}s | images={batch.image_count} "
            f"size={batch.total_image_bytes / 1024 / 1024:.2f} MiB | "
            f"GCS={upload_status} | remaining_images={remaining_images} "
            f"remaining_batches={remaining_batches}."
        )

    dataset_manifest_path = (
        settings.output_dir / "vertex_batches" / manifest.dataset_id / "manifest.json"
    )
    write_manifest(manifest, dataset_manifest_path)
    write_manifest(manifest, manifest_path(settings))
    total_seconds = perf_counter() - operation_started
    logger.success(
        f"Step 1 completed in {total_seconds:.2f}s | images={manifest.image_count} "
        f"batches={len(manifest.batches)} uploaded={uploaded} reused={reused}."
    )
    output = {
        "status": "prepared_local_only" if arguments.local_only else "prepared_and_uploaded",
        "dataset_id": manifest.dataset_id,
        "image_count": manifest.image_count,
        "batch_count": len(manifest.batches),
        "batch_sizes": [batch.image_count for batch in manifest.batches],
        "uploaded": uploaded,
        "reused": reused,
        "manifest": str(manifest_path(settings)),
        "gcs_base_uri": manifest.gcs_base_uri,
        "elapsed_seconds": round(total_seconds, 3),
        "log_path": str(log_path),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except (ConfigurationError, FileNotFoundError, RuntimeError, ValueError) as error:
        logger.error(str(error))
        raise SystemExit(2) from error
    except KeyboardInterrupt:
        logger.info("Vertex Batch preparation interrupted by the user.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
