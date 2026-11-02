"""DeepSeek-backed Text2Cypher retrieval for the MetaQA knowledge graph."""

from __future__ import annotations

import os
from collections.abc import Iterable

from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.types import RetrieverResultItem


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"

METAQA_SCHEMA = """
Node type:
(:Entity {name: STRING})

All relationships point from a movie Entity to an attribute Entity:
(:Entity)-[:DIRECTED_BY]->(:Entity)       // movie -> director
(:Entity)-[:WRITTEN_BY]->(:Entity)        // movie -> screenwriter
(:Entity)-[:STARRED_ACTORS]->(:Entity)    // movie -> actor
(:Entity)-[:HAS_GENRE]->(:Entity)         // movie -> genre
(:Entity)-[:RELEASE_YEAR]->(:Entity)      // movie -> release year
(:Entity)-[:IN_LANGUAGE]->(:Entity)       // movie -> language
(:Entity)-[:HAS_TAGS]->(:Entity)          // movie -> tag/topic
(:Entity)-[:HAS_IMDB_RATING]->(:Entity)   // movie -> rating
(:Entity)-[:HAS_IMDB_VOTES]->(:Entity)    // movie -> vote count
""".strip()

METAQA_EXAMPLES = [
    """USER INPUT: what movies did Tom Hanks act in
QUERY: MATCH (answer:Entity)-[:STARRED_ACTORS]->(actor:Entity) WHERE toLower(actor.name) = toLower("Tom Hanks") RETURN DISTINCT answer.name AS answer""",
    """USER INPUT: who directed The Green Mile
QUERY: MATCH (movie:Entity)-[:DIRECTED_BY]->(answer:Entity) WHERE toLower(movie.name) = toLower("The Green Mile") RETURN DISTINCT answer.name AS answer""",
    """USER INPUT: what genres does Forrest Gump belong to
QUERY: MATCH (movie:Entity)-[:HAS_GENRE]->(answer:Entity) WHERE toLower(movie.name) = toLower("Forrest Gump") RETURN DISTINCT answer.name AS answer""",
    """USER INPUT: which person wrote the films directed by Yuriy Norshteyn
QUERY: MATCH (director:Entity)<-[:DIRECTED_BY]-(movie:Entity)-[:WRITTEN_BY]->(answer:Entity) WHERE toLower(director.name) = toLower("Yuriy Norshteyn") RETURN DISTINCT answer.name AS answer""",
    """USER INPUT: which movies have the same director as Just Cause
QUERY: MATCH (source:Entity)-[:DIRECTED_BY]->(director:Entity)<-[:DIRECTED_BY]-(answer:Entity) WHERE toLower(source.name) = toLower("Just Cause") AND answer <> source RETURN DISTINCT answer.name AS answer""",
    """USER INPUT: what genres are the films acted by Jane Lynch in
QUERY: MATCH (actor:Entity)<-[:STARRED_ACTORS]-(movie:Entity)-[:HAS_GENRE]->(answer:Entity) WHERE toLower(actor.name) = toLower("Jane Lynch") RETURN DISTINCT answer.name AS answer""",
    """USER INPUT: the films that share actors with Dil Chahta Hai were released in which years
QUERY: MATCH (source:Entity)-[:STARRED_ACTORS]->(actor:Entity)<-[:STARRED_ACTORS]-(movie:Entity)-[:RELEASE_YEAR]->(answer:Entity) WHERE toLower(source.name) = toLower("Dil Chahta Hai") AND movie <> source RETURN DISTINCT answer.name AS answer""",
    """USER INPUT: who are the directors of movies written by the writer of The Green Mile
QUERY: MATCH (source:Entity)-[:WRITTEN_BY]->(writer:Entity)<-[:WRITTEN_BY]-(movie:Entity)-[:DIRECTED_BY]->(answer:Entity) WHERE toLower(source.name) = toLower("The Green Mile") RETURN DISTINCT answer.name AS answer""",
    """USER INPUT: what languages are films that share directors with The Age of Innocence in
QUERY: MATCH (source:Entity)-[:DIRECTED_BY]->(director:Entity)<-[:DIRECTED_BY]-(movie:Entity)-[:IN_LANGUAGE]->(answer:Entity) WHERE toLower(source.name) = toLower("The Age of Innocence") AND movie <> source RETURN DISTINCT answer.name AS answer""",
    """USER INPUT: who acted together with Didi Conn
QUERY: MATCH (topic:Entity)<-[:STARRED_ACTORS]-(movie:Entity)-[:STARRED_ACTORS]->(answer:Entity) WHERE toLower(topic.name) = toLower("Didi Conn") AND answer <> topic RETURN DISTINCT answer.name AS answer""",
    """USER INPUT: who are the actors in the movies written by Amber Benson
QUERY: MATCH (topic:Entity)<-[:WRITTEN_BY]-(movie:Entity)-[:STARRED_ACTORS]->(answer:Entity) WHERE toLower(topic.name) = toLower("Amber Benson") AND answer <> topic RETURN DISTINCT answer.name AS answer""",
    """USER INPUT: what are the languages spoken in the movies written by Rage writers
QUERY: MATCH (source:Entity)-[:WRITTEN_BY]->(writer:Entity)<-[:WRITTEN_BY]-(movie:Entity)-[:IN_LANGUAGE]->(answer:Entity) WHERE toLower(source.name) = toLower("Rage") RETURN DISTINCT answer.name AS answer""",
    """USER INPUT: who acted in the movies directed by Jim Simpson
QUERY: MATCH (director:Entity)<-[:DIRECTED_BY]-(movie:Entity)-[:STARRED_ACTORS]->(answer:Entity) WHERE toLower(director.name) = toLower("Jim Simpson") AND answer <> director RETURN DISTINCT answer.name AS answer""",
    """USER INPUT: what types are the movies starred by Barbara Sukowa
QUERY: MATCH (actor:Entity)<-[:STARRED_ACTORS]-(movie:Entity)-[:HAS_GENRE]->(answer:Entity) WHERE toLower(actor.name) = toLower("Barbara Sukowa") RETURN DISTINCT answer.name AS answer""",
    """USER INPUT: who is listed as screenwriter of the films starred by Extreme Ops actors
QUERY: MATCH (topic:Entity)-[:STARRED_ACTORS]->(actor:Entity)<-[:STARRED_ACTORS]-(movie:Entity)-[:WRITTEN_BY]->(answer:Entity) WHERE toLower(topic.name) = toLower("Extreme Ops") RETURN DISTINCT answer.name AS answer""",
]

TEXT2CYPHER_PROMPT = """
Task: Generate exactly one read-only Cypher query that answers the user question
using the MetaQA movie graph.

Schema:
{schema}

Examples:
{examples}

Rules:
1. Use only MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, DISTINCT, ORDER BY,
   aggregation functions, and read-only subqueries. Never write to the graph.
2. Use only the Entity label, name property, and relationships in the schema.
3. Relationship direction matters: every stored relationship starts at a movie.
4. Translate the semantic steps in the question into an explicit relationship
   chain. Do not use an unrestricted variable-length relationship pattern.
5. Match the named entity case-insensitively with toLower(...).
6. Return every answer and do not add LIMIT.
7. Return the final value using exactly `answer.name AS answer`.
8. Exclude the source movie when the question asks for other movies sharing an
   actor, director, or writer.
9. Output only the Cypher statement, without Markdown or explanation.
10. The topic entity is the entity the question names, and it must never appear in
    the answer set:
    - person topic: always add `AND answer <> <the variable holding it>`, such as
      `AND answer <> topic`, `AND answer <> actor`, or `AND answer <> director`;
    - movie topic whose answer can also be a movie: add `AND answer <> source`.
    Intermediate entities are never excluded: in "the movies directed by the director
    of The Hanging Garden" the topic is the movie "The Hanging Garden", and the
    director is an intermediate, so no `answer <> director` filter is added even
    though that director may also appear among the answers.
11. The topic can be either a movie title or a person name, and the direction of
    the first hop depends on which it is:
    - person topic: the first hop enters the person, e.g.
      `(topic:Entity)<-[:STARRED_ACTORS]-(movie:Entity)`;
    - movie topic: the first hop leaves the movie, e.g.
      `(source:Entity)-[:DIRECTED_BY]->(director:Entity)`.
    Phrases like "acted together with X" and "wrote movies together with X" name a
    person. Phrases like "the X director", "the X writers", "the films starred by X
    actors", or "the movies directed by the X director" name a movie title X.
12. Use exactly one relationship hop per semantic step in the question. A question
    that mentions two intermediate entities needs three hops, not two.
13. The words "type" and "types" mean genre, so they map to `HAS_GENRE`, never to
    `HAS_TAGS`. Use `HAS_TAGS` only when the question says "tags".
14. Distinguish these two forms carefully:
    - "the films starred by X actors", with the trailing word "actors", means X is a
      movie title: start at movie X, expand to its actors, then to the other films
      those actors starred in, then apply the final relationship;
    - "the movies starred by X", without that trailing "actors", means X is a person:
      the query starts at `(topic:Entity)<-[:STARRED_ACTORS]-(movie:Entity)`.

User question:
{query_text}

Cypher query:
""".strip()


def _configured_api_key() -> str:
    api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not api_key or api_key == "YOUR_DEEPSEEK_API_KEY":
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not configured. Fill it in backend/.env first."
        )
    return api_key


def build_deepseek_llm(max_tokens: int = 600) -> OpenAILLM:
    """Create the official GraphRAG OpenAI-compatible client for DeepSeek."""
    return OpenAILLM(
        model_name=os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
        model_params={
            "temperature": 0,
            "max_tokens": max_tokens,
            "extra_body": {"thinking": {"type": "disabled"}},
        },
        base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
        api_key=_configured_api_key(),
    )


def _answer_values(value) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def format_answer_record(record) -> RetrieverResultItem:
    """Normalize the required `answer` alias into a retriever result item."""
    if "answer" not in record.keys():
        raise ValueError(
            f"Generated Cypher must return `answer`; received {list(record.keys())}"
        )
    values = list(_answer_values(record["answer"]))
    return RetrieverResultItem(
        content=" | ".join(values),
        metadata={"answers": values},
    )


def build_kg_retriever(driver, llm=None) -> Text2CypherRetriever:
    """Create the official, read-only Neo4j GraphRAG Text2CypherRetriever."""
    return Text2CypherRetriever(
        driver=driver,
        llm=llm or build_deepseek_llm(),
        neo4j_schema=METAQA_SCHEMA,
        examples=METAQA_EXAMPLES,
        custom_prompt=TEXT2CYPHER_PROMPT,
        result_formatter=format_answer_record,
        neo4j_database="neo4j",
    )
