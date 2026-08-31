"""Firestore vector storage, idempotency checks, and nearest-neighbor search.

Author: Miguel Medina Cantos
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath
import shlex
import threading
from typing import Any, Mapping, Protocol, Sequence

from config.settings import FIRESTORE_MAX_RESULTS, Settings


@dataclass(frozen=True, slots=True)
class IndexStatus:
    """Minimal persisted state used to decide whether work can be skipped."""

    content_hash: str
    embedding_model: str
    extraction_model: str
    processing_status: str


@dataclass(frozen=True, slots=True)
class VectorDocument:
    """Invoice record and vector ready for persistence."""

    document_id: str
    image_identifier: str
    source_path: str
    content_hash: str
    extraction_model: str
    embedding_model: str
    embedding: list[float]
    invoice: Mapping[str, str | None]
    search_text: str


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One nearest-neighbor invoice result returned by the index."""

    document_id: str
    distance: float | None
    image_identifier: str
    source_path: str
    invoice: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible result representation."""

        return {
            "document_id": self.document_id,
            "distance": self.distance,
            "image_identifier": self.image_identifier,
            "source_path": self.source_path,
            "invoice": dict(self.invoice),
        }


@dataclass(frozen=True, slots=True)
class InvoiceRecord:
    """Structured invoice metadata loaded without embeddings or long OCR text."""

    document_id: str
    image_identifier: str
    source_path: str
    invoice: Mapping[str, Any]


class VectorIndex(Protocol):
    """Application port for a persistent vector index."""

    def lookup_statuses(self, document_ids: Sequence[str]) -> dict[str, IndexStatus]:
        """Return existing statuses for the requested deterministic document IDs."""

    def upsert_many(self, documents: Sequence[VectorDocument]) -> None:
        """Atomically upsert a bounded group of vector documents."""

    def list_invoices(self, *, limit: int) -> list[InvoiceRecord]:
        """Return structured invoice records without vector or OCR payloads."""

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
        distance_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Return nearest indexed invoices ordered by the configured distance."""


def make_document_id(relative_path: str) -> str:
    """Create a Firestore-safe stable identifier from a normalized relative path."""

    normalized = PurePosixPath(relative_path).as_posix().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


class FirestoreVectorIndex:
    """Persist normalized vectors and query KNN results in Cloud Firestore."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client
        self._client_lock = threading.Lock()

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                try:
                    from google.cloud import firestore
                except ImportError as error:
                    raise RuntimeError(
                        "google-cloud-firestore is required for vector persistence."
                    ) from error
                self._client = firestore.Client(
                    project=self._settings.firestore_project_id,
                    database=self._settings.firestore_database,
                )
        return self._client

    def lookup_statuses(self, document_ids: Sequence[str]) -> dict[str, IndexStatus]:
        """Read only the documents required for one indexing batch."""

        if not document_ids:
            return {}
        collection = self._get_client().collection(self._settings.firestore_collection)
        references = [collection.document(document_id) for document_id in document_ids]
        statuses: dict[str, IndexStatus] = {}
        for snapshot in self._get_client().get_all(references):
            if not snapshot.exists:
                continue
            data = snapshot.to_dict() or {}
            statuses[snapshot.id] = IndexStatus(
                content_hash=str(data.get("content_hash", "")),
                embedding_model=str(data.get("embedding_model", "")),
                extraction_model=str(data.get("extraction_model", "")),
                processing_status=str(data.get("processing_status", "")),
            )
        return statuses

    def upsert_many(self, documents: Sequence[VectorDocument]) -> None:
        """Write vectors in Firestore batches below the platform operation limit."""

        if not documents:
            return
        try:
            from google.cloud import firestore
            from google.cloud.firestore_v1.vector import Vector
        except ImportError as error:
            raise RuntimeError(
                "google-cloud-firestore is required for vector persistence."
            ) from error
        client = self._get_client()
        collection = client.collection(self._settings.firestore_collection)
        for offset in range(0, len(documents), 450):
            batch = client.batch()
            for document in documents[offset : offset + 450]:
                if len(document.embedding) != self._settings.embedding_dimension:
                    raise ValueError(
                        f"Vector for {document.document_id} has dimension "
                        f"{len(document.embedding)}; expected "
                        f"{self._settings.embedding_dimension}."
                    )
                payload = {
                    "image_identifier": document.image_identifier,
                    "source_path": document.source_path,
                    "content_hash": document.content_hash,
                    "processing_status": "processed",
                    "extraction_model": document.extraction_model,
                    "embedding_model": document.embedding_model,
                    "embedding_dimension": len(document.embedding),
                    "invoice": dict(document.invoice),
                    "search_text": document.search_text,
                    self._settings.firestore_vector_field: Vector(document.embedding),
                    "updated_at": firestore.SERVER_TIMESTAMP,
                }
                batch.set(collection.document(document.document_id), payload, merge=True)
            batch.commit()

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
        distance_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Execute a Firestore nearest-neighbor query and include calculated distance."""

        if len(query_vector) != self._settings.embedding_dimension:
            raise ValueError(
                f"Query vector has dimension {len(query_vector)}; expected "
                f"{self._settings.embedding_dimension}."
            )
        if limit <= 0 or limit > FIRESTORE_MAX_RESULTS:
            raise ValueError(f"Search limit must be between 1 and {FIRESTORE_MAX_RESULTS}.")
        try:
            from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
            from google.cloud.firestore_v1.vector import Vector
        except ImportError as error:
            raise RuntimeError(
                "google-cloud-firestore is required for vector search."
            ) from error
        distance_measure = getattr(DistanceMeasure, self._settings.distance_measure)
        collection = self._get_client().collection(self._settings.firestore_collection)
        query_options: dict[str, Any] = {
            "vector_field": self._settings.firestore_vector_field,
            "query_vector": Vector(list(query_vector)),
            "distance_measure": distance_measure,
            "limit": limit,
            "distance_result_field": "vector_distance",
        }
        if distance_threshold is not None:
            query_options["distance_threshold"] = distance_threshold
        snapshots = collection.find_nearest(**query_options).stream()
        results: list[SearchResult] = []
        for snapshot in snapshots:
            data = snapshot.to_dict() or {}
            raw_distance = data.get("vector_distance")
            results.append(
                SearchResult(
                    document_id=snapshot.id,
                    distance=float(raw_distance) if raw_distance is not None else None,
                    image_identifier=str(data.get("image_identifier", "")),
                    source_path=str(data.get("source_path", "")),
                    invoice=data.get("invoice") or {},
                )
            )
        return results

    def list_invoices(self, *, limit: int) -> list[InvoiceRecord]:
        """Read compact invoice metadata once for global and aggregate questions."""

        if limit <= 0 or limit > FIRESTORE_MAX_RESULTS:
            raise ValueError(f"List limit must be between 1 and {FIRESTORE_MAX_RESULTS}.")
        collection = self._get_client().collection(self._settings.firestore_collection)
        snapshots = (
            collection.select(
                [
                    "image_identifier",
                    "source_path",
                    "invoice.invoice_number",
                    "invoice.supplier_name",
                    "invoice.invoice_date",
                    "invoice.currency",
                    "invoice.subtotal",
                    "invoice.tax",
                    "invoice.total",
                ]
            )
            .order_by("image_identifier")
            .limit(limit)
            .stream()
        )
        records: list[InvoiceRecord] = []
        for snapshot in snapshots:
            data = snapshot.to_dict() or {}
            records.append(
                InvoiceRecord(
                    document_id=snapshot.id,
                    image_identifier=str(data.get("image_identifier", "")),
                    source_path=str(data.get("source_path", "")),
                    invoice=data.get("invoice") or {},
                )
            )
        return records


def build_vector_index_command(settings: Settings) -> str:
    """Return the documented gcloud command for the required flat vector index."""

    settings.validate(require_cloud=True, require_input=False)
    vector_config = (
        f"field-path={settings.firestore_vector_field},"
        f"vector-config={{\"dimension\":\"{settings.embedding_dimension}\",\"flat\": \"{{}}\"}}"
    )
    arguments = [
        "gcloud",
        "firestore",
        "indexes",
        "composite",
        "create",
        f"--collection-group={settings.firestore_collection}",
        "--query-scope=COLLECTION",
        f"--field-config={vector_config}",
        f"--database={settings.firestore_database}",
        f"--project={settings.firestore_project_id}",
    ]
    return " ".join(shlex.quote(argument) for argument in arguments)
