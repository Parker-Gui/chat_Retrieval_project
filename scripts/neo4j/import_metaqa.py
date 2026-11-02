"""Import the normalized MetaQA triples into Neo4j."""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase, ManagedTransaction


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "graph" / "metaqa_triples.csv"
DEFAULT_ENV = PROJECT_ROOT / "neo4j" / ".env"
RELATION_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import MetaQA triples into Neo4j.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing Neo4j graph before importing. Use with care.",
    )
    return parser.parse_args()


def import_batch(
    tx: ManagedTransaction, relation_type: str, rows: list[dict[str, str]]
) -> None:
    if not RELATION_PATTERN.fullmatch(relation_type):
        raise ValueError(f"Unsafe relationship type: {relation_type!r}")
    cypher = f"""
    UNWIND $rows AS row
    MERGE (subject:Entity {{name: row.subject}})
    MERGE (object:Entity {{name: row.object}})
    MERGE (subject)-[:`{relation_type.upper()}`]->(object)
    """
    tx.run(cypher, rows=rows).consume()


def graph_statistics(driver) -> dict:
    records, _, _ = driver.execute_query(
        """
        MATCH (entity:Entity)
        WITH count(entity) AS entity_count
        MATCH ()-[relationship]->()
        RETURN entity_count, count(relationship) AS relationship_count
        """,
        database_="neo4j",
    )
    relation_records, _, _ = driver.execute_query(
        """
        MATCH ()-[relationship]->()
        RETURN type(relationship) AS relationship, count(*) AS count
        ORDER BY relationship
        """,
        database_="neo4j",
    )
    summary = dict(records[0])
    summary["relations"] = {
        record["relationship"]: record["count"] for record in relation_records
    }
    return summary


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")

    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Missing processed triples: {input_path}. "
            "Run scripts/data/prepare_metaqa.py first."
        )

    load_dotenv(args.env_file.resolve())
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError(f"NEO4J_PASSWORD is missing from {args.env_file}")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        if args.reset:
            driver.execute_query(
                "MATCH (node) DETACH DELETE node", database_="neo4j"
            )

        driver.execute_query(
            """
            CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
            FOR (entity:Entity) REQUIRE entity.name IS UNIQUE
            """,
            database_="neo4j",
        )

        pending: dict[str, list[dict[str, str]]] = defaultdict(list)
        imported_rows = 0
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required_fields = {"subject", "relation", "object"}
            if set(reader.fieldnames or []) != required_fields:
                raise ValueError(
                    f"Expected CSV fields {sorted(required_fields)}, "
                    f"found {reader.fieldnames}"
                )

            with driver.session(database="neo4j") as session:
                for row in reader:
                    relation = row["relation"]
                    pending[relation].append(
                        {"subject": row["subject"], "object": row["object"]}
                    )
                    if len(pending[relation]) >= args.batch_size:
                        batch = pending[relation]
                        session.execute_write(import_batch, relation, batch)
                        imported_rows += len(batch)
                        pending[relation] = []
                        if imported_rows % 20_000 < args.batch_size:
                            print(f"Processed {imported_rows:,} triples...")

                for relation, batch in pending.items():
                    if batch:
                        session.execute_write(import_batch, relation, batch)
                        imported_rows += len(batch)

        print(f"Processed {imported_rows:,} source triples.")
        print("Graph statistics:")
        statistics = graph_statistics(driver)
        print(f"  Entities: {statistics['entity_count']:,}")
        print(f"  Relationships: {statistics['relationship_count']:,}")
        for relation, count in statistics["relations"].items():
            print(f"  {relation}: {count:,}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
