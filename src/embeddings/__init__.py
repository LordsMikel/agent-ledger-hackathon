"""Embedding generation and Firestore vector persistence.

Author: Miguel Medina Cantos
"""

from src.embeddings.vector_index import (
    FirestoreVectorIndex,
    InvoiceRecord,
    SearchResult,
    VectorDocument,
)
from src.embeddings.vector_search import SentenceTransformerEmbeddingService

__all__ = [
    "FirestoreVectorIndex",
    "InvoiceRecord",
    "SearchResult",
    "SentenceTransformerEmbeddingService",
    "VectorDocument",
]
