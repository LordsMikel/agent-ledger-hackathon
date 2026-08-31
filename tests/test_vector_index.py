"""Tests for deterministic vector identifiers and Firestore index configuration.

Author: Miguel Medina Cantos
"""

from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from config.settings import Settings
from src.embeddings.vector_index import (
    FirestoreVectorIndex,
    VectorDocument,
    build_vector_index_command,
    make_document_id,
)


class _DocumentReference:
    def __init__(self, document_id: str) -> None:
        self.id = document_id


class _Snapshot:
    def __init__(self, document_id: str, data: dict[str, object], exists: bool = True) -> None:
        self.id = document_id
        self._data = data
        self.exists = exists

    def to_dict(self) -> dict[str, object]:
        return self._data


class _Query:
    def __init__(self, snapshots: list[_Snapshot]) -> None:
        self._snapshots = snapshots

    def stream(self) -> list[_Snapshot]:
        return self._snapshots


class _Collection:
    def __init__(self) -> None:
        self.snapshots: list[_Snapshot] = []
        self.query_options: dict[str, object] | None = None

    def document(self, document_id: str) -> _DocumentReference:
        return _DocumentReference(document_id)

    def find_nearest(self, **options: object) -> _Query:
        self.query_options = options
        return _Query(self.snapshots)


class _Batch:
    def __init__(self, client: "_Client") -> None:
        self._client = client
        self.operations: list[tuple[_DocumentReference, dict[str, object], bool]] = []

    def set(
        self, reference: _DocumentReference, payload: dict[str, object], *, merge: bool
    ) -> None:
        self.operations.append((reference, payload, merge))

    def commit(self) -> None:
        self._client.committed.extend(self.operations)


class _Client:
    def __init__(self) -> None:
        self.collection_reference = _Collection()
        self.committed: list[tuple[_DocumentReference, dict[str, object], bool]] = []

    def collection(self, _: str) -> _Collection:
        return self.collection_reference

    def batch(self) -> _Batch:
        return _Batch(self)

    def get_all(self, _: list[_DocumentReference]) -> list[_Snapshot]:
        return self.collection_reference.snapshots


def _firestore_modules() -> dict[str, ModuleType]:
    google_module = ModuleType("google")
    cloud_module = ModuleType("google.cloud")
    firestore_module = ModuleType("google.cloud.firestore")
    firestore_v1_module = ModuleType("google.cloud.firestore_v1")
    vector_module = ModuleType("google.cloud.firestore_v1.vector")
    distance_module = ModuleType("google.cloud.firestore_v1.base_vector_query")
    firestore_module.SERVER_TIMESTAMP = object()
    vector_module.Vector = lambda values: tuple(values)

    class _DistanceMeasure:
        COSINE = "COSINE"
        DOT_PRODUCT = "DOT_PRODUCT"
        EUCLIDEAN = "EUCLIDEAN"

    distance_module.DistanceMeasure = _DistanceMeasure
    cloud_module.firestore = firestore_module
    google_module.cloud = cloud_module
    return {
        "google": google_module,
        "google.cloud": cloud_module,
        "google.cloud.firestore": firestore_module,
        "google.cloud.firestore_v1": firestore_v1_module,
        "google.cloud.firestore_v1.vector": vector_module,
        "google.cloud.firestore_v1.base_vector_query": distance_module,
    }


class VectorIndexTests(unittest.TestCase):
    """Verify stable IDs and documented Firestore vector-index parameters."""

    def test_document_identifier_is_stable_and_path_sensitive(self) -> None:
        first = make_document_id("input/000001.jpg")
        self.assertEqual(first, make_document_id("input/000001.jpg"))
        self.assertNotEqual(first, make_document_id("input/000002.jpg"))
        self.assertEqual(len(first), 64)

    def test_index_command_contains_collection_dimension_and_flat_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            settings = Settings(
                app_root=root,
                input_dir=input_dir,
                output_dir=root / "output",
                firestore_project_id="example-project",
            )
            command = build_vector_index_command(settings)
            self.assertIn("--collection-group=invoices", command)
            self.assertIn("dimension", command)
            self.assertIn("384", command)
            self.assertIn("flat", command)
            self.assertIn("--project=example-project", command)

    def test_firestore_adapter_upserts_vector_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            client = _Client()
            index = FirestoreVectorIndex(settings, client=client)
            document = VectorDocument(
                document_id="document-id",
                image_identifier="000001.jpg",
                source_path="input/000001.jpg",
                content_hash="content-hash",
                extraction_model="gemini-3.5-flash",
                embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
                embedding=[1.0, 0.0, 0.0],
                invoice={"total": "10.00"},
                search_text="Total: 10.00",
            )
            with patch.dict(sys.modules, _firestore_modules()):
                index.upsert_many([document])
            self.assertEqual(len(client.committed), 1)
            _, payload, merge = client.committed[0]
            self.assertTrue(merge)
            self.assertEqual(payload["embedding"], (1.0, 0.0, 0.0))
            self.assertEqual(payload["processing_status"], "processed")

    def test_firestore_adapter_executes_search_and_returns_distance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            client = _Client()
            client.collection_reference.snapshots = [
                _Snapshot(
                    "document-id",
                    {
                        "vector_distance": 0.91,
                        "image_identifier": "000001.jpg",
                        "source_path": "input/000001.jpg",
                        "invoice": {"supplier_name": "Acme"},
                    },
                )
            ]
            index = FirestoreVectorIndex(settings, client=client)
            with patch.dict(sys.modules, _firestore_modules()):
                results = index.search([1.0, 0.0, 0.0], limit=5)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].distance, 0.91)
            self.assertEqual(results[0].invoice["supplier_name"], "Acme")
            self.assertEqual(
                client.collection_reference.query_options["distance_measure"],
                "DOT_PRODUCT",
            )

    def test_firestore_adapter_handles_empty_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            index = FirestoreVectorIndex(settings, client=_Client())
            with patch.dict(sys.modules, _firestore_modules()):
                self.assertEqual(index.search([1.0, 0.0, 0.0], limit=5), [])

    @staticmethod
    def _settings(root: Path) -> Settings:
        input_dir = root / "input"
        input_dir.mkdir()
        return Settings(
            app_root=root,
            input_dir=input_dir,
            output_dir=root / "output",
            embedding_dimension=3,
            firestore_project_id="example-project",
        )


if __name__ == "__main__":
    unittest.main()
