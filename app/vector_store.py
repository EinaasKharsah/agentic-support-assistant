"""Store the knowledge base vectors in Qdrant and search them."""

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.embeddings import Embedder
from app.knowledge_base import Entry


@dataclass(frozen=True)
class SearchResult:
    entry: Entry
    score: float  # cosine similarity: close to 1 = very similar meaning


def make_client(url: str | None) -> QdrantClient:
    """Connect to a Qdrant server, or run Qdrant in memory if no URL is set.

    In-memory mode is handy for tests and for running without Docker.
    """
    return QdrantClient(url=url) if url else QdrantClient(":memory:")


def _point_id(entry_id: str) -> str:
    # Qdrant needs a number or UUID as id. We derive a fixed UUID from "Q1",
    # so indexing again overwrites the same points instead of duplicating them.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, entry_id))


class VectorStore:
    def __init__(self, client: QdrantClient, embedder: Embedder, collection: str):
        self.client = client
        self.embedder = embedder
        self.collection = collection

    def index(self, entries: list[Entry]) -> None:
        """Embed all entries and save them. Safe to run on every start."""
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.embedder.dimension, distance=Distance.COSINE
                ),
            )

        vectors = self.embedder.embed_documents([e.to_text() for e in entries])
        points = [
            PointStruct(
                id=_point_id(entry.id),
                vector=vector,
                # The payload is stored next to the vector but is not searched.
                # We keep the full entry so a search result can cite its source.
                payload={
                    "id": entry.id,
                    "category": entry.category,
                    "question": entry.question,
                    "answer": entry.answer,
                    "out_of_scope": entry.out_of_scope,
                },
            )
            for entry, vector in zip(entries, vectors)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        """Return the top_k closest entries, best match first."""
        response = self.client.query_points(
            collection_name=self.collection,
            query=self.embedder.embed_query(question),
            limit=top_k,
            with_payload=True,
        )
        return [
            SearchResult(entry=Entry(**point.payload), score=point.score)
            for point in response.points
        ]
