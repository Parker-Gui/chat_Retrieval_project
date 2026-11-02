"""Evaluate DeepSeek Text2Cypher KG Retrieval on the fixed MetaQA benchmark."""

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
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.retrieval.kg import build_kg_retriever  # noqa: E402


DEFAULT_BENCHMARK = (
    PROJECT_ROOT / "experiments" / "shared" / "benchmark" / "midterm_300.jsonl"
)
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "experiments"
    / "midterm"
    / "predictions"
    / "kg_retrieval_predictions.jsonl"
)
DEFAULT_METRICS = (
    PROJECT_ROOT / "experiments" / "midterm" / "metrics" / "kg_retrieval_metrics.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Text2Cypher KG Retrieval.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help=(
            "Metrics output path. Defaults to the main metrics file, or to a "
            "subset-specific file when --hop/--limit are used."
        ),
    )
    parser.add_argument(
        "--hop", type=int, choices=[1, 2, 3], action="append", dest="hops"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Evaluate only the first N selected rows."
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry examples that already have an error result.",
    )
    parser.add_argument(
        "--retry-wrong",
        action="store_true",
        help=(
            "Retry examples that errored or whose recorded answer set was not an "
            "exact match. Use after changing the schema, prompt, or few-shot examples."
        ),
    )
    return parser.parse_args()


def normalize(value: str) -> str:
    return " ".join(str(value).casefold().split())


def resolve_metrics_path(args: argparse.Namespace) -> Path:
    """Keep subset runs from overwriting the metrics of the full benchmark."""
    if args.metrics is not None:
        return args.metrics.resolve()
    suffixes = []
    if args.hops:
        suffixes.append("hop" + "-".join(str(hop) for hop in sorted(set(args.hops))))
    if args.limit > 0:
        suffixes.append(f"limit{args.limit}")
    if not suffixes:
        return DEFAULT_METRICS
    return DEFAULT_METRICS.with_name(
        f"{DEFAULT_METRICS.stem}_{'_'.join(suffixes)}{DEFAULT_METRICS.suffix}"
    )


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_predictions(path: Path, rows_by_id: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows_by_id.values(),
        key=lambda row: (int(row.get("hop", 0)), int(row.get("source_line", 0))),
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def extract_answers(result) -> list[str]:
    values = []
    for item in result.items:
        values.extend((item.metadata or {}).get("answers", []))
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))


def score_success(example: dict, predicted: list[str], cypher: str, latency_ms: float) -> dict:
    gold = {normalize(value) for value in example["answers"]}
    prediction = {normalize(value) for value in predicted}
    true_positive = len(gold & prediction)
    precision = true_positive / len(prediction) if prediction else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        **example,
        "status": "success",
        "predicted_answers": predicted,
        "cypher": cypher,
        "exact_match": prediction == gold,
        "answer_precision": precision,
        "answer_recall": recall,
        "answer_f1": f1,
        "latency_ms": latency_ms,
    }


def aggregate(rows: list[dict]) -> dict:
    """Score every selected question, counting failed queries as zero accuracy."""
    total = len(rows)
    successes = [row for row in rows if row.get("status") == "success"]
    if not total:
        return {"count": 0, "successful_queries": 0}

    def mean(key: str, selected: list[dict]) -> float:
        return statistics.fmean(row.get(key, 0) or 0 for row in selected)

    summary = {
        "count": total,
        "successful_queries": len(successes),
        "cypher_execution_success_rate": len(successes) / total,
        "exact_match": mean("exact_match", rows),
        "answer_precision": mean("answer_precision", rows),
        "answer_recall": mean("answer_recall", rows),
        "answer_f1": mean("answer_f1", rows),
    }
    if successes:
        summary["mean_latency_ms"] = mean("latency_ms", successes)
        summary["median_latency_ms"] = statistics.median(
            row["latency_ms"] for row in successes
        )
        summary["success_only"] = {
            "count": len(successes),
            "exact_match": mean("exact_match", successes),
            "answer_precision": mean("answer_precision", successes),
            "answer_recall": mean("answer_recall", successes),
            "answer_f1": mean("answer_f1", successes),
        }
    else:
        summary["mean_latency_ms"] = None
        summary["median_latency_ms"] = None
    return summary


def main() -> None:
    args = parse_args()
    benchmark = load_jsonl(args.benchmark.resolve())
    if args.hops:
        allowed = set(args.hops)
        benchmark = [row for row in benchmark if int(row["hop"]) in allowed]
    if args.limit > 0:
        benchmark = benchmark[: args.limit]
    selected_ids = {row["id"] for row in benchmark}

    existing = {
        row["id"]: row
        for row in load_jsonl(args.predictions.resolve())
        if row.get("id")
    }
    def needs_rerun(example_id: str) -> bool:
        if example_id not in existing:
            return True
        recorded = existing[example_id]
        if args.retry_errors and recorded.get("status") != "success":
            return True
        if args.retry_wrong and (
            recorded.get("status") != "success" or not recorded.get("exact_match")
        ):
            return True
        return False

    selected_existing_count = sum(example_id in existing for example_id in selected_ids)
    pending = [row for row in benchmark if needs_rerun(row["id"])]

    load_dotenv(PROJECT_ROOT / "backend" / ".env")
    load_dotenv(PROJECT_ROOT / "neo4j" / ".env")
    api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not api_key or api_key == "YOUR_DEEPSEEK_API_KEY":
        raise RuntimeError(
            "DEEPSEEK_API_KEY is still a placeholder. Fill backend/.env, then rerun."
        )

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]),
    )
    try:
        driver.verify_connectivity()
        retriever = build_kg_retriever(driver)
        print(
            f"Selected {len(benchmark)} questions; already recorded {len(existing)}; "
            f"selected records {selected_existing_count}; pending {len(pending)}.",
            flush=True,
        )
        for number, example in enumerate(pending, 1):
            started = time.perf_counter()
            try:
                result = retriever.search(query_text=example["question"])
                latency_ms = (time.perf_counter() - started) * 1_000
                row = score_success(
                    example,
                    extract_answers(result),
                    str((result.metadata or {}).get("cypher", "")),
                    latency_ms,
                )
            except Exception as error:
                row = {
                    **example,
                    "status": "error",
                    "predicted_answers": [],
                    "cypher": "",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "latency_ms": (time.perf_counter() - started) * 1_000,
                }
            existing[example["id"]] = row
            save_predictions(args.predictions.resolve(), existing)
            print(
                f"[{number}/{len(pending)}] hop={example['hop']} "
                f"status={row['status']} id={example['id']}",
                flush=True,
            )
    finally:
        driver.close()

    evaluated = [existing[row["id"]] for row in benchmark if row["id"] in existing]
    by_hop: dict[int, list[dict]] = defaultdict(list)
    for row in evaluated:
        by_hop[int(row["hop"])].append(row)
    metrics = {
        "experiment": "step6_kg_retrieval_text2cypher",
        "framework": "neo4j-graphrag Text2CypherRetriever",
        "llm_provider": "DeepSeek",
        "llm_model": os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": args.benchmark.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "selection": {
            "hops": sorted(set(args.hops)) if args.hops else [1, 2, 3],
            "limit": args.limit or None,
        },
        "selected_count": len(benchmark),
        "overall": aggregate(evaluated),
        "by_hop": {str(hop): aggregate(rows) for hop, rows in sorted(by_hop.items())},
        "metric_note": (
            "Metrics compare the complete set of Neo4j answers returned by the "
            "LLM-generated read-only Cypher with the MetaQA gold answer set. "
            "Top-level exact_match/precision/recall/F1 average over every selected "
            "question and count a failed query as zero; `success_only` repeats the "
            "same metrics for queries whose Cypher executed."
        ),
    }
    metrics_path = resolve_metrics_path(args)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Predictions: {args.predictions}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
