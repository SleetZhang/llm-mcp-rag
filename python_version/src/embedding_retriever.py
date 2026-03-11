from __future__ import annotations

import os

import requests

from .utils import log_title
from .vector_store import VectorStore


class EmbeddingRetriever:
    def __init__(self, embedding_model: str) -> None:
        self.embedding_model = embedding_model
        self.vector_store = VectorStore()

    def embed_document(self, document: str) -> list[float]:
        log_title("EMBEDDING DOCUMENT")
        embedding = self._embed(document)
        self.vector_store.add_embedding(embedding, document)
        return embedding

    def embed_query(self, query: str) -> list[float]:
        log_title("EMBEDDING QUERY")
        return self._embed(query)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        query_embedding = self.embed_query(query)
        return self.vector_store.search(query_embedding, top_k)

    def _embed(self, text: str) -> list[float]:
        response = requests.post(
            f"{os.environ['EMBEDDING_BASE_URL']}/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['EMBEDDING_KEY']}",
            },
            json={
                "model": self.embedding_model,
                "input": text,
                "encoding_format": "float",
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]
