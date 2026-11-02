"""Embed MetaQA triple documents and create a Neo4j vector index."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase, ManagedTransaction
from neo4j_graphrag.indexes import create_vector_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.retrieval.vector.neo4j_vector import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_PROPERTY,
    INDEX_NAME,
    NODE_LABEL,
    create_embedder,
)


DEFAULT_DOCUMENTS = (
    PROJECT_ROOT / "data" / "processed" / "documents" / "metaqa_documents.jsonl"
)
DEFAULT_ENV = PROJECT_ROOT / "neo4j" / ".env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Step 5 Neo4j GraphRAG vector index."
    )
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete only KnowledgeChunk nodes and rebuild their vector index.",
    )
    return parser.parse_args()


def load_documents(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/data/prepare_metaqa.py first."
        )
    documents = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            required = {"id", "text", "subject", "relation", "object"}
            missing = required - item.keys()
            if missing:
                raise ValueError(f"Line {line_number} is missing {sorted(missing)}")
            documents.append(item)
    return documents


def write_batch(tx: ManagedTransaction, rows: list[dict]) -> None:
    tx.run(
        f"""
        UNWIND $rows AS row
        MERGE (chunk:{NODE_LABEL} {{id: row.id}})
        SET chunk.text = row.text,
            chunk.subject = row.subject,
            chunk.relation = row.relation,
            chunk.object = row.object,
            chunk.embedding_model = row.embedding_model,
            chunk.{EMBEDDING_PROPERTY} = row.embedding
        """,
        rows=rows,
    ).consume()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be greater than zero")

    load_dotenv(args.env_file.resolve())
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError(f"NEO4J_PASSWORD is missing from {args.env_file}")

    documents = load_documents(args.documents.resolve())
    print(f"Loaded {len(documents):,} MetaQA triple documents.", flush=True)
    print(f"Embedding model: {args.model}", flush=True)

    driver = GraphDatabase.driver(uri, auth=(user, password))
    started = time.perf_counter()
    try:
        driver.verify_connectivity()
        if args.rebuild:
            driver.execute_query(
                f"MATCH (chunk:{NODE_LABEL}) DETACH DELETE chunk",
                database_="neo4j",
            )

        driver.execute_query(
            f"""
            CREATE CONSTRAINT knowledge_chunk_id_unique IF NOT EXISTS
            FOR (chunk:{NODE_LABEL}) REQUIRE chunk.id IS UNIQUE
            """,
            database_="neo4j",
        )

        records, _, _ = driver.execute_query(
            f"""
            MATCH (chunk:{NODE_LABEL})
            WHERE chunk.{EMBEDDING_PROPERTY} IS NOT NULL
              AND chunk.embedding_model = $model
            RETURN chunk.id AS id
            """,
            model=args.model,
            database_="neo4j",
        )
        completed_ids = {record["id"] for record in records}
        pending = [item for item in documents if item["id"] not in completed_ids]
        print(
            f"Already indexed: {len(completed_ids):,}; remaining: {len(pending):,}",
            flush=True,
        )

        embedder = create_embedder(args.model)
        dimension = int(embedder.model.get_embedding_dimension())
        print(f"Embedding dimension: {dimension}", flush=True)

        with driver.session(database="neo4j") as session:
            for offset in range(0, len(pending), args.batch_size):
                batch = pending[offset : offset + args.batch_size]
                texts = [item["text"] for item in batch]
                vectors = embedder.model.encode(
                    texts,
                    batch_size=args.batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                rows = []
                for item, vector in zip(batch, vectors, strict=True):
                    rows.append(
                        {
                            **item,
                            "embedding_model": args.model,
                            "embedding": vector.tolist(),
                        }
                    )
                session.execute_write(write_batch, rows)
                processed = min(offset + len(batch), len(pending))
                if processed % 5_120 < args.batch_size or processed == len(pending):
                    elapsed = time.perf_counter() - started
                    print(
                        f"Embedded {processed:,}/{len(pending):,} remaining "
                        f"documents ({elapsed:.1f}s).",
                        flush=True,
                    )

        create_vector_index(
            driver=driver,
            name=INDEX_NAME,
            label=NODE_LABEL,
            embedding_property=EMBEDDING_PROPERTY,
            dimensions=dimension,
            similarity_fn="cosine",
            fail_if_exists=False,
            neo4j_database="neo4j",
        )
        driver.execute_query(
            "CALL db.awaitIndex($index_name, 300)",
            index_name=INDEX_NAME,
            database_="neo4j",
        )

        count_records, _, _ = driver.execute_query(
            f"""
            MATCH (chunk:{NODE_LABEL})
            WHERE chunk.{EMBEDDING_PROPERTY} IS NOT NULL
            RETURN count(chunk) AS count
            """,
            database_="neo4j",
        )
        index_records, _, _ = driver.execute_query(
            """
            SHOW VECTOR INDEXES YIELD name, state, populationPercent
            WHERE name = $index_name
            RETURN name, state, populationPercent
            """,
            index_name=INDEX_NAME,
            database_="neo4j",
        )
        print(f"Indexed KnowledgeChunk nodes: {count_records[0]['count']:,}")
        print(f"Vector index: {dict(index_records[0])}")
        print(f"Finished in {time.perf_counter() - started:.1f}s.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
