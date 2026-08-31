"""Tests for indexing and retrieval use cases with deterministic fakes.

Author: Miguel Medina Cantos
"""

import base64
from pathlib import Path
import tempfile
import unittest

from config.settings import Settings
from src.agents.extractor.agent import InvoiceData
from src.application.orchestrator import AgentLedgerOrchestrator
from src.embeddings.vector_index import IndexStatus, SearchResult, VectorDocument


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)


class _FakeExtractor:
    model_name = "fake-gemini"

    def __init__(self) -> None:
        self.calls = 0

    def extract(self, image_path: Path) -> InvoiceData:
        self.calls += 1
        return InvoiceData(
            supplier_name=f"Supplier {image_path.stem}",
            invoice_number=image_path.stem,
            total="10.00",
            currency="EUR",
        )


class _FakeEmbeddings:
    model_name = "fake-minilm"
    dimension = 3

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class _FakeVectorIndex:
    def __init__(self) -> None:
        self.documents: dict[str, VectorDocument] = {}
        self.search_results: list[SearchResult] = []

    def lookup_statuses(self, document_ids: list[str]) -> dict[str, IndexStatus]:
        return {
            document_id: IndexStatus(
                content_hash=self.documents[document_id].content_hash,
                embedding_model=self.documents[document_id].embedding_model,
                extraction_model=self.documents[document_id].extraction_model,
                processing_status="processed",
            )
            for document_id in document_ids
            if document_id in self.documents
        }

    def upsert_many(self, documents: list[VectorDocument]) -> None:
        for document in documents:
            self.documents[document.document_id] = document

    def search(
        self,
        query_vector: list[float],
        *,
        limit: int,
        distance_threshold: float | None = None,
    ) -> list[SearchResult]:
        return self.search_results[:limit]


class OrchestratorTests(unittest.TestCase):
    """Verify batching, duplicate prevention, failures, indexing, and empty search."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_dir = self.root / "input"
        self.output_dir = self.root / "output"
        self.input_dir.mkdir()
        self.settings = Settings(
            app_root=self.root,
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            embedding_dimension=3,
            embedding_batch_size=2,
            firestore_project_id="test-project",
        )
        self.extractor = _FakeExtractor()
        self.embeddings = _FakeEmbeddings()
        self.index = _FakeVectorIndex()
        self.orchestrator = AgentLedgerOrchestrator(
            settings=self.settings,
            extractor=self.extractor,
            embeddings=self.embeddings,
            vector_index=self.index,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _add_image(self, name: str) -> Path:
        path = self.input_dir / name
        path.write_bytes(PNG_BYTES)
        return path

    def test_indexing_persists_vectors_and_extracted_json(self) -> None:
        self._add_image("000001.png")
        self._add_image("000002.png")
        summary = self.orchestrator.index_images()
        self.assertEqual(summary.to_dict()["processed"], 2)
        self.assertEqual(len(self.index.documents), 2)
        self.assertEqual(len(list((self.output_dir / "extracted").glob("*.json"))), 2)
        self.assertEqual(len(self.embeddings.batches), 1)

    def test_second_run_skips_current_documents(self) -> None:
        self._add_image("000001.png")
        first = self.orchestrator.index_images()
        second = self.orchestrator.index_images()
        self.assertEqual(first.processed, 1)
        self.assertEqual(second.processed, 0)
        self.assertEqual(second.skipped, 1)
        self.assertEqual(self.extractor.calls, 1)

    def test_force_reprocesses_current_documents(self) -> None:
        self._add_image("000001.png")
        self.orchestrator.index_images()
        summary = self.orchestrator.index_images(force=True)
        self.assertEqual(summary.processed, 1)
        self.assertEqual(self.extractor.calls, 2)

    def test_invalid_image_is_reported_without_stopping_batch(self) -> None:
        self._add_image("valid.png")
        (self.input_dir / "broken.jpg").write_text("broken", encoding="utf-8")
        summary = self.orchestrator.index_images()
        self.assertEqual(summary.discovered, 2)
        self.assertEqual(summary.processed, 1)
        self.assertEqual(summary.failed, 1)

    def test_empty_index_search_returns_empty_results(self) -> None:
        self.assertEqual(self.orchestrator.search("March invoices"), [])

    def test_empty_query_is_rejected_before_embedding(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            self.orchestrator.search("  ")


if __name__ == "__main__":
    unittest.main()
