"""Evaluate the Step 5 Neo4j GraphRAG VectorRetriever on MetaQA."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.retrieval.vector.neo4j_vector import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    INDEX_NAME,
    build_vector_retriever,
    create_embedder,
)


DEFAULT_BENCHMARK = (
    PROJECT_ROOT / "experiments" / "shared" / "benchmark" / "midterm_300.jsonl"
)
DEFAULT_ENV = PROJECT_ROOT / "neo4j" / ".env"
DEFAULT_METRICS = (
    PROJECT_ROOT / "experiments" / "midterm" / "metrics" / "vector_rag_metrics.json"
)
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "experiments"
    / "midterm"
    / "predictions"
    / "vector_rag_predictions.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Vector RAG retrieval.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    return parser.parse_args()


def normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_example(example: dict, retrieved: list[dict], latency_ms: float) -> dict:
    gold = {normalize(answer) for answer in example["answers"]}
    answer_ranks: dict[str, int] = {}
    for rank, item in enumerate(retrieved, 1):
        candidates = {
            normalize(str(item.get("subject") or "")),
            normalize(str(item.get("object") or "")),
        }
        for answer in gold & candidates:
            answer_ranks.setdefault(answer, rank)

    first_rank = min(answer_ranks.values()) if answer_ranks else None
    return {
        **example,
        "retrieved": retrieved,
        "hit_at_1": first_rank is not None and first_rank <= 1,
        "hit_at_3": first_rank is not None and first_rank <= 3,
        "hit_at_5": first_rank is not None and first_rank <= 5,
        "answer_recall_at_5": len(answer_ranks) / len(gold) if gold else 0.0,
        "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
        "latency_ms": latency_ms,
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "hits_at_1": sum(row["hit_at_1"] for row in rows) / len(rows),
        "hits_at_3": sum(row["hit_at_3"] for row in rows) / len(rows),
        "hits_at_5": sum(row["hit_at_5"] for row in rows) / len(rows),
        "answer_recall_at_5": statistics.fmean(
            row["answer_recall_at_5"] for row in rows
        ),
        "mrr": statistics.fmean(row["reciprocal_rank"] for row in rows),
        "mean_latency_ms": statistics.fmean(row["latency_ms"] for row in rows),
        "median_latency_ms": statistics.median(row["latency_ms"] for row in rows),
    }


def main() -> None:
    args = parse_args()
    if args.top_k < 5:
        raise ValueError("--top-k must be at least 5 to compute Hits@5")

    benchmark = load_jsonl(args.benchmark.resolve())
    load_dotenv(args.env_file.resolve())
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError(f"NEO4J_PASSWORD is missing from {args.env_file}")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    predictions = []
    try:
        driver.verify_connectivity()
        embedder = create_embedder(args.model)
        retriever = build_vector_retriever(driver, embedder=embedder)
        print(
            f"Evaluating {len(benchmark)} questions with {INDEX_NAME}, "
            f"top_k={args.top_k}."
        )
        for number, example in enumerate(benchmark, 1):
            started = time.perf_counter()
            result = retriever.search(query_text=example["question"], top_k=args.top_k)
            latency_ms = (time.perf_counter() - started) * 1_000
            retrieved = [item.metadata or {} for item in result.items]
            predictions.append(score_example(example, retrieved, latency_ms))
            if number % 25 == 0 or number == len(benchmark):
                print(f"Evaluated {number}/{len(benchmark)} questions.")
    finally:
        driver.close()

    by_hop: dict[int, list[dict]] = defaultdict(list)
    for row in predictions:
        by_hop[int(row["hop"])].append(row)
    metrics = {
        "experiment": "step5_vector_rag_retrieval",
        "framework": "neo4j-graphrag VectorRetriever",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": args.benchmark.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "embedding_model": args.model,
        "vector_index": INDEX_NAME,
        "top_k": args.top_k,
        "overall": aggregate(predictions),
        "by_hop": {str(hop): aggregate(rows) for hop, rows in sorted(by_hop.items())},
        "metric_note": (
            "A retrieval hit means a gold answer appears as the subject or object "
            "of a retrieved MetaQA triple. This evaluates retrieval, not LLM generation."
        ),
    }

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Predictions: {args.predictions}")
    print(f"Metrics: {args.metrics}")


if __name__ == "__main__":
    main()
