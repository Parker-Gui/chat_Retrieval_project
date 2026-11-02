"""Process-wide lazy wiring for the answering pipelines.

The embedding model and the Neo4j retrievers are expensive to build, so each
system is created once per process and reused by every request. Nothing is
constructed at import time, which keeps the FastAPI app importable (and testable)
when no API key or database is available.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from app.rag.workflow import AnswerPipeline, SUPPORTED_SYSTEMS
from app.retrieval.kg.text2cypher import build_deepseek_llm, build_kg_retriever
from app.retrieval.vector.neo4j_vector import build_vector_retriever, create_embedder


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ENV = PROJECT_ROOT / "backend" / ".env"
NEO4J_ENV = PROJECT_ROOT / "neo4j" / ".env"
DEFAULT_SYSTEM = os.getenv("ANSWER_SYSTEM", "kg_retrieval")

_lock = threading.Lock()
_state: dict = {"pipelines": {}, "driver": None, "llm": None, "embedder": None}


class PipelineConfigurationError(RuntimeError):
    """Raised when a pipeline cannot be built from the current environment."""


def _load_environment() -> None:
    load_dotenv(BACKEND_ENV)
    load_dotenv(NEO4J_ENV)


def configured_system() -> str:
    """System selected by ANSWER_SYSTEM, validated against the supported set."""
    system = (os.getenv("ANSWER_SYSTEM") or DEFAULT_SYSTEM or "kg_retrieval").strip()
    if system not in SUPPORTED_SYSTEMS:
        raise PipelineConfigurationError(
            f"ANSWER_SYSTEM={system!r} is not supported; choose from {SUPPORTED_SYSTEMS}."
        )
    return system


def _require_api_key() -> str:
    api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not api_key or api_key == "YOUR_DEEPSEEK_API_KEY":
        raise PipelineConfigurationError(
            "DEEPSEEK_API_KEY is not configured. Fill it in backend/.env first."
        )
    return api_key


def _driver():
    """Create the shared Neo4j driver on first use."""
    if _state["driver"] is not None:
        return _state["driver"]
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise PipelineConfigurationError(
            "NEO4J_PASSWORD is not configured. Fill it in neo4j/.env first."
        )
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), password),
    )
    driver.verify_connectivity()
    _state["driver"] = driver
    return driver


def _generation_llm():
    if _state["llm"] is None:
        _state["llm"] = build_deepseek_llm(max_tokens=400)
    return _state["llm"]


def _embedder():
    if _state["embedder"] is None:
        _state["embedder"] = create_embedder()
    return _state["embedder"]


def build_pipeline(system: str) -> AnswerPipeline:
    """Build one answering system with shared, lazily-created resources."""
    _load_environment()
    _require_api_key()
    if system not in SUPPORTED_SYSTEMS:
        raise PipelineConfigurationError(
            f"Unknown system {system!r}; choose from {SUPPORTED_SYSTEMS}."
        )
    # The closed-book baseline must remain independent of Neo4j. Retrieval
    # systems create the shared driver lazily when they actually need it.
    driver = _driver() if system != "pure_llm" else None
    kg_retriever = (
        build_kg_retriever(driver, llm=build_deepseek_llm())
        if system in ("kg_retrieval", "hybrid")
        else None
    )
    vector_retriever = (
        build_vector_retriever(driver, embedder=_embedder())
        if system in ("vector_rag", "hybrid")
        else None
    )
    return AnswerPipeline(
        llm=_generation_llm(),
        system=system,
        kg_retriever=kg_retriever,
        vector_retriever=vector_retriever,
    )


def get_pipeline(system: str | None = None) -> AnswerPipeline:
    """Return the process-wide pipeline for a system, building it on first use."""
    target = system or configured_system()
    pipeline = _state["pipelines"].get(target)
    if pipeline is not None:
        return pipeline
    with _lock:
        if target not in _state["pipelines"]:
            _state["pipelines"][target] = build_pipeline(target)
    return _state["pipelines"][target]


def shutdown() -> None:
    """Close the shared driver; used by the FastAPI shutdown hook and tests."""
    with _lock:
        driver = _state.get("driver")
        if driver is not None:
            driver.close()
        _state["driver"] = None
        _state["pipelines"] = {}
        _state["llm"] = None
        _state["embedder"] = None
