"""Shared Neo4j GraphRAG VectorRetriever construction."""

from __future__ import annotations

from pathlib import Path

from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.types import RetrieverResultItem

from app.retrieval.vector.local_embedder import LocalMiniLMEmbedder


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MODEL_CACHE_DIR = PROJECT_ROOT / "tools" / "model_cache"
INDEX_NAME = "metaqa_chunk_embedding"
NODE_LABEL = "KnowledgeChunk"
EMBEDDING_PROPERTY = "embedding"
RETURN_PROPERTIES = ["id", "text", "subject", "relation", "object"]


def create_embedder(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Embedder:
    """Create the local embedding model used for both indexing and querying.

    The official SentenceTransformers wrapper is tried first. On this machine
    `sentence-transformers` 6.2 plus `transformers` 5.22 cannot load the cached
    checkpoint offline (the loader requires an AutoProcessor that the repository
    does not contain), so the equivalent local pipeline is used as a fallback. Both
    paths produce mean-pooled, L2-normalised vectors for the same weights, which
    keeps them interchangeable with the index built earlier.
    """
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        return SentenceTransformerEmbeddings(
            model=model_name,
            cache_folder=str(MODEL_CACHE_DIR),
            device="cpu",
        )
    except Exception as error:
        print(
            "SentenceTransformers loader unavailable "
            f"({type(error).__name__}); using the local MiniLM pipeline instead.",
            flush=True,
        )
        return LocalMiniLMEmbedder(
            model_name=model_name,
            cache_folder=MODEL_CACHE_DIR,
            device="cpu",
        )


def _format_result(record) -> RetrieverResultItem:
    node = dict(record["node"])
    return RetrieverResultItem(
        content=node.get("text", ""),
        metadata={
            "id": node.get("id"),
            "subject": node.get("subject"),
            "relation": node.get("relation"),
            "object": node.get("object"),
            "score": float(record["score"]),
        },
    )


def build_vector_retriever(driver, embedder=None) -> VectorRetriever:
    """Return the official Neo4j GraphRAG retriever for the MetaQA index."""
    return VectorRetriever(
        driver=driver,
        index_name=INDEX_NAME,
        embedder=embedder or create_embedder(),
        return_properties=RETURN_PROPERTIES,
        result_formatter=_format_result,
        neo4j_database="neo4j",
    )
