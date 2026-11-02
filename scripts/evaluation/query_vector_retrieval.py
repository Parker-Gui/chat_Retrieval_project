"""Run one interactive query against the Step 5 vector retriever."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.retrieval.vector.neo4j_vector import build_vector_retriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the MetaQA vector index.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / "neo4j" / ".env")
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]),
    )
    try:
        result = build_vector_retriever(driver).search(
            query_text=args.question, top_k=args.top_k
        )
        print(f"Question: {args.question}\n")
        for rank, item in enumerate(result.items, 1):
            score = (item.metadata or {}).get("score", 0.0)
            print(f"{rank}. [{score:.4f}] {item.content}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
