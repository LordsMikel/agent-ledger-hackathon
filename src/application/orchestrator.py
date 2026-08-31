"""Application use cases coordinating extraction, embedding, indexing, and search.

Author: Miguel Medina Cantos
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import islice
import json
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Sequence

from config.settings import ConfigurationError, FIRESTORE_MAX_RESULTS, Settings, get_logger
from src.agents.extractor.agent import (
    InvoiceData,
    InvoiceExtractor,
    discover_images,
    sha256_file,
    validate_image,
)
from src.embeddings.vector_index import (
    IndexStatus,
    InvoiceRecord,
    SearchResult,
    VectorDocument,
    VectorIndex,
    make_document_id,
)
from src.embeddings.vector_search import EmbeddingService


@dataclass(frozen=True, slots=True)
class IndexingError:
    """One recoverable file-level indexing failure."""

    source_path: str
    message: str


@dataclass(slots=True)
class IndexingSummary:
    """Counters and bounded error details for one indexing execution."""

    discovered: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    selected_bytes: int = 0
    errors: list[IndexingError] = field(default_factory=list)

    def record_error(self, source_path: str, error: Exception) -> None:
        """Record a failure without retaining unbounded error state."""

        self.failed += 1
        if len(self.errors) < 100:
            self.errors.append(IndexingError(source_path=source_path, message=str(error)))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible summary."""

        return {
            "discovered": self.discovered,
            "processed": self.processed,
            "skipped": self.skipped,
            "failed": self.failed,
            "selected_bytes": self.selected_bytes,
            "errors": [asdict(error) for error in self.errors],
        }


@dataclass(frozen=True, slots=True)
class _Candidate:
    path: Path
    relative_path: str
    document_id: str
    content_hash: str


def _select_single_batch(
    image_paths: Iterable[Path],
    *,
    max_bytes: int,
    max_images: int,
) -> tuple[list[Path], int]:
    """Select the sorted prefix that fits one bounded Gemini batch, then stop."""

    selected: list[Path] = []
    total_bytes = 0
    for path in image_paths:
        if len(selected) >= max_images:
            break
        image_bytes = path.stat().st_size
        if total_bytes + image_bytes > max_bytes:
            break
        selected.append(path)
        total_bytes += image_bytes
    return selected, total_bytes


class AgentLedgerOrchestrator:
    """Coordinate bounded pipeline batches while keeping infrastructure behind ports."""

    def __init__(
        self,
        *,
        settings: Settings,
        extractor: InvoiceExtractor,
        embeddings: EmbeddingService,
        vector_index: VectorIndex,
    ) -> None:
        self._settings = settings
        self._extractor = extractor
        self._embeddings = embeddings
        self._vector_index = vector_index
        self._logger = get_logger()

    def index_images(
        self,
        *,
        force: bool = False,
        limit: int | None = None,
        offset: int = 0,
        progress: Callable[[IndexingSummary], None] | None = None,
    ) -> IndexingSummary:
        """Discover, extract, embed, and persist invoices with duplicate prevention."""

        indexing_started = perf_counter()
        if not self._settings.insert_into_index:
            raise ConfigurationError(
                "Image ingestion is disabled. Set INSERT_INTO_INDEX=true to index images."
            )
        self._settings.validate()
        self._settings.prepare_output()
        if limit is not None and limit <= 0:
            raise ValueError("Indexing limit must be greater than zero.")
        if offset < 0:
            raise ValueError("Indexing offset cannot be negative.")
        summary = IndexingSummary()
        max_images = self._settings.gemini_batch_size
        if limit is not None:
            max_images = min(max_images, limit)
        eligible_paths = islice(
            discover_images(self._settings.input_dir),
            offset,
            None,
        )
        selected_paths, selected_bytes = _select_single_batch(
            eligible_paths,
            max_bytes=self._settings.gemini_batch_max_bytes,
            max_images=max_images,
        )
        summary.selected_bytes = selected_bytes
        validation_started = perf_counter()
        candidates: list[_Candidate] = []
        for path in selected_paths:
            summary.discovered += 1
            relative_path = path.relative_to(self._settings.app_root).as_posix()
            try:
                validate_image(path, max_bytes=self._settings.max_image_bytes)
                candidates.append(
                    _Candidate(
                        path=path,
                        relative_path=relative_path,
                        document_id=make_document_id(relative_path),
                        content_hash=sha256_file(path),
                    )
                )
            except Exception as error:
                summary.record_error(relative_path, error)
        self._logger.info(
            f"Image validation and hashing finished in "
            f"{perf_counter() - validation_started:.2f}s for {len(selected_paths)} images."
        )
        if candidates:
            self._process_candidates(candidates, summary=summary, force=force)
        if progress:
            progress(summary)
        self._logger.info(
            f"Indexing batch finished in {perf_counter() - indexing_started:.2f}s: "
            f"processed={summary.processed}, skipped={summary.skipped}, "
            f"failed={summary.failed}."
        )
        return summary

    def _process_candidates(
        self,
        candidates: Sequence[_Candidate],
        *,
        summary: IndexingSummary,
        force: bool,
    ) -> None:
        lookup_started = perf_counter()
        statuses = self._vector_index.lookup_statuses(
            [candidate.document_id for candidate in candidates]
        )
        self._logger.info(
            f"Firestore status lookup finished in {perf_counter() - lookup_started:.2f}s "
            f"for {len(candidates)} images."
        )
        pending: list[_Candidate] = []
        for candidate in candidates:
            status = statuses.get(candidate.document_id)
            if not force and self._is_current(status, candidate):
                summary.skipped += 1
                continue
            pending.append(candidate)
        extracted: list[tuple[_Candidate, InvoiceData]] = []
        if pending:
            extraction_started = perf_counter()
            failures_before_extraction = summary.failed
            batch_method = getattr(self._extractor, "extract_batch", None)
            if callable(batch_method):
                try:
                    outcomes = batch_method([candidate.path for candidate in pending])
                    if len(outcomes) != len(pending):
                        raise RuntimeError(
                            "Invoice extractor returned an unexpected batch result count."
                        )
                    for candidate, outcome in zip(pending, outcomes, strict=True):
                        if outcome.succeeded and outcome.invoice is not None:
                            extracted.append((candidate, outcome.invoice))
                            self._write_extraction(candidate, outcome.invoice)
                        else:
                            summary.record_error(
                                candidate.relative_path,
                                RuntimeError(outcome.error or "Gemini batch item failed."),
                            )
                except Exception as error:
                    for candidate in pending:
                        summary.record_error(candidate.relative_path, error)
            else:
                for candidate in pending:
                    try:
                        invoice = self._extractor.extract(candidate.path)
                        extracted.append((candidate, invoice))
                        self._write_extraction(candidate, invoice)
                    except Exception as error:
                        summary.record_error(candidate.relative_path, error)
            self._logger.info(
                f"Gemini extraction and JSON persistence finished in "
                f"{perf_counter() - extraction_started:.2f}s: requested={len(pending)}, "
                f"extracted={len(extracted)}, "
                f"failed={summary.failed - failures_before_extraction}."
            )
        if not extracted:
            return
        embedding_started = perf_counter()
        try:
            vectors = self._embeddings.embed_texts(
                [invoice.search_text for _, invoice in extracted]
            )
        except Exception as error:
            for candidate, _ in extracted:
                summary.record_error(candidate.relative_path, error)
            self._logger.error(
                f"MiniLM embedding generation failed after "
                f"{perf_counter() - embedding_started:.2f}s: {error}"
            )
            return
        self._logger.info(
            f"MiniLM embedding generation finished in "
            f"{perf_counter() - embedding_started:.2f}s for {len(vectors)} invoices."
        )

        documents = [
            VectorDocument(
                document_id=candidate.document_id,
                image_identifier=candidate.path.name,
                source_path=candidate.relative_path,
                content_hash=candidate.content_hash,
                extraction_model=self._extractor.model_name,
                embedding_model=self._embeddings.model_name,
                embedding=vector,
                invoice=invoice.to_dict(),
                search_text=invoice.search_text,
            )
            for (candidate, invoice), vector in zip(extracted, vectors, strict=True)
        ]
        firestore_started = perf_counter()
        try:
            self._vector_index.upsert_many(documents)
        except Exception as error:
            for candidate, _ in extracted:
                summary.record_error(candidate.relative_path, error)
            self._logger.error(
                f"Firestore upsert failed after {perf_counter() - firestore_started:.2f}s: "
                f"{error}"
            )
            return
        summary.processed += len(documents)
        self._logger.info(
            f"Firestore upsert finished in {perf_counter() - firestore_started:.2f}s "
            f"for {len(documents)} invoices."
        )

    def _is_current(self, status: IndexStatus | None, candidate: _Candidate) -> bool:
        return bool(
            status
            and status.processing_status == "processed"
            and status.content_hash == candidate.content_hash
            and status.embedding_model == self._embeddings.model_name
            and status.extraction_model == self._extractor.model_name
        )

    def _write_extraction(self, candidate: _Candidate, invoice: InvoiceData) -> None:
        destination = self._settings.output_dir / "extracted" / f"{candidate.document_id}.json"
        temporary = destination.with_suffix(".json.tmp")
        payload = {
            "image_identifier": candidate.path.name,
            "source_path": candidate.relative_path,
            "content_hash": candidate.content_hash,
            "extraction_model": self._extractor.model_name,
            "invoice": invoice.to_dict(),
            "search_text": invoice.search_text,
        }
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(destination)

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        distance_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Embed a natural-language query and retrieve nearest invoice records."""

        self._settings.validate(require_cloud=True, require_input=False)
        if not query.strip():
            raise ValueError("Search query cannot be empty.")
        if limit <= 0 or limit > FIRESTORE_MAX_RESULTS:
            raise ValueError(f"Search limit must be between 1 and {FIRESTORE_MAX_RESULTS}.")
        query_vector = self._embeddings.embed_query(query.strip())
        return self._vector_index.search(
            query_vector,
            limit=limit,
            distance_threshold=distance_threshold,
        )

    def list_invoices(self, *, limit: int) -> list[InvoiceRecord]:
        """Load compact structured records for one global invoice analysis."""

        self._settings.validate(require_cloud=True, require_input=False)
        if limit <= 0 or limit > FIRESTORE_MAX_RESULTS:
            raise ValueError(f"List limit must be between 1 and {FIRESTORE_MAX_RESULTS}.")
        firestore_started = perf_counter()
        records = self._vector_index.list_invoices(limit=limit)
        self._logger.info(
            f"Firestore structured invoice read finished in "
            f"{perf_counter() - firestore_started:.3f}s for {len(records)} records."
        )
        return records
