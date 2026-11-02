"""Verify the imported MetaQA graph and multi-hop answer coverage."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV = PROJECT_ROOT / "neo4j" / ".env"
DEFAULT_BENCHMARK = (
    PROJECT_ROOT / "experiments" / "shared" / "benchmark" / "midterm_300.jsonl"
)
EXPECTED_ENTITIES = 43_234
EXPECTED_RELATIONSHIPS = 133_582
EXPECTED_RELATION_TYPES = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the MetaQA Neo4j graph.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--questions-per-hop", type=int, default=10)
    return parser.parse_args()


def load_benchmark(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def answer_is_reachable(driver, question: dict) -> bool:
    hop = int(question["hop"])
    if hop not in (1, 2, 3):
        raise ValueError(f"Unsupported hop value: {hop}")
    records, _, _ = driver.execute_query(
        f"""
        MATCH (topic:Entity {{name: $topic_entity}})
        MATCH (topic)-[*{hop}]-(answer:Entity)
        WHERE answer.name IN $answers
        RETURN answer.name AS answer
        LIMIT 1
        """,
        topic_entity=question["topic_entity"],
        answers=question["answers"],
        database_="neo4j",
    )
    return bool(records)


def main() -> None:
    args = parse_args()
    if args.questions_per_hop <= 0:
        raise ValueError("--questions-per-hop must be greater than zero")

    load_dotenv(args.env_file.resolve())
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError(f"NEO4J_PASSWORD is missing from {args.env_file}")
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        count_records, _, _ = driver.execute_query(
            """
            MATCH (entity:Entity)
            WITH count(entity) AS entities
            MATCH ()-[relationship]->()
            RETURN entities, count(relationship) AS relationships
            """,
            database_="neo4j",
        )
        type_records, _, _ = driver.execute_query(
            "MATCH ()-[relationship]->() RETURN count(DISTINCT type(relationship)) AS types",
            database_="neo4j",
        )
        entities = count_records[0]["entities"]
        relationships = count_records[0]["relationships"]
        relation_types = type_records[0]["types"]

        if entities != EXPECTED_ENTITIES:
            raise AssertionError(
                f"Expected {EXPECTED_ENTITIES:,} entities, found {entities:,}"
            )
        if relationships != EXPECTED_RELATIONSHIPS:
            raise AssertionError(
                f"Expected {EXPECTED_RELATIONSHIPS:,} relationships, "
                f"found {relationships:,}"
            )
        if relation_types != EXPECTED_RELATION_TYPES:
            raise AssertionError(
                f"Expected {EXPECTED_RELATION_TYPES} relationship types, "
                f"found {relation_types}"
            )

        benchmark = load_benchmark(args.benchmark.resolve())
        checked = Counter()
        reachable = Counter()
        for question in benchmark:
            hop = int(question["hop"])
            if checked[hop] >= args.questions_per_hop:
                continue
            checked[hop] += 1
            if answer_is_reachable(driver, question):
                reachable[hop] += 1

        print("Neo4j graph verification passed.")
        print(f"  Entities: {entities:,}")
        print(f"  Relationships: {relationships:,}")
        print(f"  Relationship types: {relation_types}")
        for hop in (1, 2, 3):
            print(
                f"  {hop}-hop answer coverage: "
                f"{reachable[hop]}/{checked[hop]}"
            )
            if reachable[hop] != checked[hop]:
                raise AssertionError(f"{hop}-hop answer coverage verification failed")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
