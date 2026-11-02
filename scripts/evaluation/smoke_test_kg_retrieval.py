"""Test 1/2/3-hop Text2Cypher execution without calling an external LLM."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.llm.types import LLMResponse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.retrieval.kg import build_kg_retriever  # noqa: E402


CASES = [
    {
        "name": "1-hop",
        "question": "what movies did Tom Hanks act in",
        "cypher": 'MATCH (answer:Entity)-[:STARRED_ACTORS]->(actor:Entity) WHERE toLower(actor.name) = toLower("Tom Hanks") RETURN DISTINCT answer.name AS answer',
        "expected": "Forrest Gump",
    },
    {
        "name": "2-hop",
        "question": "which person wrote the films directed by Yuriy Norshteyn",
        "cypher": 'MATCH (director:Entity)<-[:DIRECTED_BY]-(movie:Entity)-[:WRITTEN_BY]->(answer:Entity) WHERE toLower(director.name) = toLower("Yuriy Norshteyn") RETURN DISTINCT answer.name AS answer',
        "expected": "Sergei Kozlov",
    },
    {
        "name": "3-hop",
        "question": "the films that share actors with Dil Chahta Hai were released in which years",
        "cypher": 'MATCH (source:Entity)-[:STARRED_ACTORS]->(actor:Entity)<-[:STARRED_ACTORS]-(movie:Entity)-[:RELEASE_YEAR]->(answer:Entity) WHERE toLower(source.name) = toLower("Dil Chahta Hai") AND movie <> source RETURN DISTINCT answer.name AS answer',
        "expected": "1997",
    },
]


class FixedCypherLLM:
    def __init__(self, cypher: str) -> None:
        self.cypher = cypher

    def invoke(self, _prompt: str) -> LLMResponse:
        return LLMResponse(content=self.cypher)


def main() -> None:
    load_dotenv(PROJECT_ROOT / "neo4j" / ".env")
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]),
    )
    try:
        driver.verify_connectivity()
        for case in CASES:
            result = build_kg_retriever(
                driver, llm=FixedCypherLLM(case["cypher"])
            ).search(query_text=case["question"])
            answers = {
                answer
                for item in result.items
                for answer in (item.metadata or {}).get("answers", [])
            }
            if case["expected"] not in answers:
                raise AssertionError(
                    f"{case['name']} missing {case['expected']!r}: {sorted(answers)}"
                )
            print(
                f"PASS {case['name']}: {len(answers)} answers; "
                f"contains {case['expected']!r}"
            )

        unsafe_query = 'CREATE (:Entity {name: "SHOULD_NOT_EXIST"}) RETURN "x" AS answer'
        try:
            build_kg_retriever(
                driver, llm=FixedCypherLLM(unsafe_query)
            ).search(query_text="create a node")
        except Exception as error:
            if "Refusing to execute non-read-only Cypher" not in str(error):
                raise
        else:
            raise AssertionError("The retriever did not reject a write query")
        records, _, _ = driver.execute_query(
            'MATCH (node:Entity {name: "SHOULD_NOT_EXIST"}) RETURN count(node) AS count',
            database_="neo4j",
        )
        if records[0]["count"] != 0:
            raise AssertionError("Unsafe test query modified the graph")
        print("PASS safety: non-read-only Cypher was rejected before execution")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
