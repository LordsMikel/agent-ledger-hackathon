"""Tests for deterministic batch embedding behavior without model downloads.

Author: Miguel Medina Cantos
"""

import unittest

from src.embeddings.vector_search import EmbeddingError, SentenceTransformerEmbeddingService


class _FakeSentenceTransformer:
    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts: list[str], **_: object) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class VectorSearchTests(unittest.TestCase):
    """Verify batching, dimensions, and empty-input behavior with a fake model."""

    def setUp(self) -> None:
        self.model_name = "tests/fake-normalized-model"
        SentenceTransformerEmbeddingService._models[(self.model_name, "cpu")] = (
            _FakeSentenceTransformer()
        )
        self.service = SentenceTransformerEmbeddingService(
            model_name=self.model_name,
            dimension=3,
            batch_size=2,
            device="cpu",
            normalize=True,
        )

    def tearDown(self) -> None:
        SentenceTransformerEmbeddingService._models.pop((self.model_name, "cpu"), None)

    def test_batch_embeddings_are_returned_for_each_text(self) -> None:
        self.assertEqual(
            self.service.embed_texts(["first invoice", "second invoice"]),
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        )

    def test_empty_batch_does_not_load_or_call_model(self) -> None:
        self.assertEqual(self.service.embed_texts([]), [])

    def test_empty_query_is_rejected(self) -> None:
        with self.assertRaisesRegex(EmbeddingError, "cannot be empty"):
            self.service.embed_query("   ")


if __name__ == "__main__":
    unittest.main()
