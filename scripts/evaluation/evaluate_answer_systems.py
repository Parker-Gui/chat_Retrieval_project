"""Evaluate one answering system on the fixed MetaQA benchmark.

Systems share the same generation prompt and answer-set scoring, so differences
are attributable to the retrieval stage:

  pure_llm, vector_rag, kg_retrieval   -> midterm comparison
  hybrid                               -> final-stage comparison

Answer-set metrics reuse the Step 6 scoring helpers; retrieval-stage diagnostics
(Hits@K are measured separately by evaluate_vector_retrieval.py, Cypher success by
evaluate_kg_retrieval.py) are reported alongside so they are never confused with
final-answer accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "evaluation"))

from evaluate_kg_retrieval import (  # noqa: E402
    aggregate,
    load_jsonl,
    normalize,
    save_predictions,
    score_success,
)

from app.rag.service import get_pipeline, shutdown  # noqa: E402
from app.rag.workflow import SUPPORTED_SYSTEMS  # noqa: E402


DEFAULT_BENCHMARK = (
    PROJECT_ROOT / "experiments" / "shared" / "benchmark" / "midterm_300.jsonl"
)
PREDICTION_DIR = PROJECT_ROOT / "experiments" / "midterm" / "predictions"
METRICS_DIR = PROJECT_ROOT / "experiments" / "midterm" / "metrics"
# The `answer_` prefix keeps these files apart from the Step 5 retrieval-only and
# Step 6 Text2Cypher artefacts, which use the bare system names.
FILE_PREFIX = "answer_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one answering system.")
    parser.add_argument("--system", required=True, choices=list(SUPPORTED_SYSTEMS))
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Metrics path; defaults to a system- and subset-specific name.",
    )
    parser.add_argument("--hop", type=int, choices=[1, 2, 3], action="append", dest="hops")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--retry-wrong", action="store_true")
    return parser.parse_args()


def prediction_path(args: argparse.Namespace) -> Path:
    if args.predictions is not None:
        return args.predictions.resolve()
    return PREDICTION_DIR / f"{FILE_PREFIX}{args.system}_predictions.jsonl"


def metrics_path(args: argparse.Namespace) -> Path:
    if args.metrics is not None:
        return args.metrics.resolve()
    suffixes = []
    if args.hops:
        suffixes.append("hop" + "-".join(str(hop) for hop in sorted(set(args.hops))))
    if args.limit > 0:
        suffixes.append(f"limit{args.limit}")
    stem = f"{FILE_PREFIX}{args.system}_metrics"
    if suffixes:
        stem += "_" + "_".join(suffixes)
    return METRICS_DIR / f"{stem}.json"


def text_covers_gold(final_answer: str, gold: list[str]) -> bool:
    """True when every gold answer appears in the generated sentence."""
    text = normalize(final_answer)
    return bool(gold) and all(normalize(value) in text for value in gold)


def fatal_pipeline_errors(system: str, errors: list[str]) -> list[str]:
    """Return failures that make a system's answer invalid for evaluation.

    KG generation may fail without invalidating its authoritative structured answer
    set. Pure LLM and Vector RAG, however, obtain their scored answer sets from the
    generator, so a generation failure must never be silently counted as a valid
    zero-score prediction.
    """
    fatal_stages = {
        "pure_llm": ("generation",),
        "vector_rag": ("vector_retrieval", "merge", "generation"),
        "kg_retrieval": ("kg_retrieval", "merge"),
        "hybrid": ("kg_retrieval", "vector_retrieval", "merge"),
    }[system]
    return [
        error
        for error in errors
        if any(error.startswith(stage + ":") for stage in fatal_stages)
    ]


def system_aggregate(rows: list[dict], system: str) -> dict:
    base = aggregate(rows)
    if not rows:
        return base
    # ``aggregate`` labels the executed-query rate after the KG retriever; for the
    # answer systems it is simply the share of questions that produced an answer.
    if system not in ("kg_retrieval", "hybrid"):
        base.pop("cypher_execution_success_rate", None)
    base["kg_retrieval_success_rate"] = (
        sum(1 for row in rows if row.get("used_kg") and not row.get("kg_error")) / len(rows)
    )
    base["vector_retrieval_success_rate"] = (
        sum(1 for row in rows if row.get("used_vector") and not row.get("vector_error"))
        / len(rows)
    )
    base["answer_text_covers_gold_rate"] = (
        sum(1 for row in rows if row.get("text_covers_gold")) / len(rows)
    )
    return base


def main() -> None:
    args = parse_args()
    predictions_file = prediction_path(args)
    benchmark = load_jsonl(args.benchmark.resolve())
    if args.hops:
        allowed = set(args.hops)
        benchmark = [row for row in benchmark if int(row["hop"]) in allowed]
    if args.limit > 0:
        benchmark = benchmark[: args.limit]

    existing = {
        row["id"]: row for row in load_jsonl(predictions_file) if row.get("id")
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

    pending = [row for row in benchmark if needs_rerun(row["id"])]
    print(
        f"System {args.system}: selected {len(benchmark)} questions; "
        f"already recorded {len(existing)}; pending {len(pending)}.",
        flush=True,
    )

    pipeline = get_pipeline(args.system) if pending else None
    try:
        for number, example in enumerate(pending, 1):
            started = time.perf_counter()
            try:
                state = pipeline.run(example["question"])
                latency_ms = (time.perf_counter() - started) * 1_000
                fatal_errors = fatal_pipeline_errors(
                    args.system, state.retrieval_errors
                )
                if fatal_errors:
                    raise RuntimeError("; ".join(fatal_errors))
                row = score_success(
                    example, state.answer_set, state.kg_cypher, latency_ms
                )
                row.update(
                    {
                        "system": args.system,
                        "final_answer": state.final_answer,
                        "vector_context": state.vector_context,
                        "retrieval_errors": state.retrieval_errors,
                        "timings_ms": state.timings_ms,
                        "used_kg": state.used_kg,
                        "used_vector": state.used_vector,
                        "kg_error": any(
                            e.startswith("kg_retrieval") for e in state.retrieval_errors
                        ),
                        "vector_error": any(
                            e.startswith("vector_retrieval") for e in state.retrieval_errors
                        ),
                        "text_covers_gold": text_covers_gold(
                            state.final_answer, example["answers"]
                        ),
                    }
                )
            except Exception as error:
                row = {
                    **example,
                    "system": args.system,
                    "status": "error",
                    "predicted_answers": [],
                    "cypher": "",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "latency_ms": (time.perf_counter() - started) * 1_000,
                    "text_covers_gold": False,
                }
            existing[example["id"]] = row
            save_predictions(predictions_file, existing)
            print(
                f"[{number}/{len(pending)}] hop={example['hop']} status={row['status']} "
                f"em={row.get('exact_match')} id={example['id']}",
                flush=True,
            )
    finally:
        if pending:
            shutdown()

    evaluated = [existing[row["id"]] for row in benchmark if row["id"] in existing]
    by_hop: dict[int, list[dict]] = defaultdict(list)
    for row in evaluated:
        by_hop[int(row["hop"])].append(row)
    metrics = {
        "experiment": f"midterm_answer_system_{args.system}",
        "system": args.system,
        "framework": "neo4j-graphrag retrievers + deterministic pipeline",
        "llm_provider": "DeepSeek",
        "benchmark": args.benchmark.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "selection": {
            "hops": sorted(set(args.hops)) if args.hops else [1, 2, 3],
            "limit": args.limit or None,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_count": len(benchmark),
        "overall": system_aggregate(evaluated, args.system),
        "by_hop": {
            str(hop): system_aggregate(rows, args.system)
            for hop, rows in sorted(by_hop.items())
        },
        "metric_note": (
            "answer_set metrics use the same scoring code as Step 6 and count a failed "
            "question as zero. For kg_retrieval and hybrid the structured KG answer set "
            "is authoritative; for pure_llm and vector_rag the answer list is the one the "
            "generator returned. `answer_text_covers_gold_rate` checks the generated "
            "sentence instead of the answer set and is a stricter text-level check."
        ),
    }
    out = metrics_path(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Predictions: {predictions_file}")
    print(f"Metrics: {out}")


if __name__ == "__main__":
    main()
