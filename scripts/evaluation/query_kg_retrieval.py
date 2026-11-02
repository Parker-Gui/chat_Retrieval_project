"""Ask one question through DeepSeek + Neo4j GraphRAG Text2CypherRetriever."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.retrieval.kg import build_kg_retriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one KG Retrieval query.")
    parser.add_argument("question")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / "backend" / ".env")
    load_dotenv(PROJECT_ROOT / "neo4j" / ".env")
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]),
    )
    try:
        result = build_kg_retriever(driver).search(query_text=args.question)
        answers = [item.content for item in result.items if item.content]
        print(f"Question: {args.question}")
        print(f"Cypher: {result.metadata.get('cypher')}")
        print("Answers:")
        for answer in answers:
            print(f"- {answer}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
