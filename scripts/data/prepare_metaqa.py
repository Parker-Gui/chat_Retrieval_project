"""Validate and normalize the MetaQA knowledge base and question files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "metaqa"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
HOPS = (1, 2, 3)
VALID_SPLITS = ("train", "dev", "test")
EXPECTED_COUNTS = {
    1: {"train": 96_106, "dev": 9_992, "test": 9_947},
    2: {"train": 118_980, "dev": 14_872, "test": 14_872},
    3: {"train": 114_196, "dev": 14_274, "test": 14_274},
}
TOPIC_ENTITY_PATTERN = re.compile(r"\[([^\[\]]+)]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate MetaQA and create normalized graph, document, and QA files."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=VALID_SPLITS,
        default=["test"],
        help="Question splits to normalize. The default is test only.",
    )
    return parser.parse_args()


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_triples(path: Path) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            parts = line.split("|")
            if len(parts) != 3 or any(not part.strip() for part in parts):
                raise ValueError(f"Invalid knowledge triple at {path}:{line_number}")
            triples.append(tuple(part.strip() for part in parts))
    return triples


def triple_to_document(subject: str, relation: str, obj: str) -> str:
    readable_relation = relation.replace("_", " ")
    return f"{subject} {readable_relation} {obj}."


def export_knowledge_base(
    triples: list[tuple[str, str, str]], output_dir: Path
) -> dict:
    graph_dir = output_dir / "graph"
    document_dir = output_dir / "documents"
    graph_dir.mkdir(parents=True, exist_ok=True)
    document_dir.mkdir(parents=True, exist_ok=True)

    csv_path = graph_dir / "metaqa_triples.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject", "relation", "object"])
        writer.writerows(triples)

    write_jsonl(
        graph_dir / "metaqa_triples.jsonl",
        (
            {"id": index, "subject": subject, "relation": relation, "object": obj}
            for index, (subject, relation, obj) in enumerate(triples)
        ),
    )
    write_jsonl(
        document_dir / "metaqa_documents.jsonl",
        (
            {
                "id": f"triple-{index:06d}",
                "text": triple_to_document(subject, relation, obj),
                "subject": subject,
                "relation": relation,
                "object": obj,
            }
            for index, (subject, relation, obj) in enumerate(triples)
        ),
    )

    entities = {entity for subject, _, obj in triples for entity in (subject, obj)}
    relation_counts = Counter(relation for _, relation, _ in triples)
    return {
        "triple_count": len(triples),
        "entity_count": len(entities),
        "relation_count": len(relation_counts),
        "relations": dict(sorted(relation_counts.items())),
    }


def parse_question_line(
    raw_line: str, *, hop: int, split: str, line_number: int
) -> dict:
    line = raw_line.rstrip("\r\n")
    try:
        question_raw, answer_text = line.split("\t", maxsplit=1)
    except ValueError as error:
        raise ValueError(
            f"Invalid QA record in {hop}-hop {split} at line {line_number}"
        ) from error

    topic_match = TOPIC_ENTITY_PATTERN.search(question_raw)
    if topic_match is None:
        raise ValueError(
            f"Missing bracketed topic entity in {hop}-hop {split} at line {line_number}"
        )

    answers = [answer.strip() for answer in answer_text.split("|") if answer.strip()]
    if not answers:
        raise ValueError(f"Missing answer in {hop}-hop {split} at line {line_number}")

    return {
        "id": f"metaqa-{hop}hop-{split}-{line_number:06d}",
        "question": question_raw.replace("[", "").replace("]", ""),
        "question_raw": question_raw,
        "topic_entity": topic_match.group(1),
        "answers": answers,
        "hop": hop,
        "split": split,
        "source_line": line_number,
    }


def normalize_questions(
    raw_dir: Path, output_dir: Path, splits: list[str]
) -> dict[str, int]:
    benchmark_dir = output_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for hop in HOPS:
        for split in splits:
            source_path = raw_dir / f"{hop}-hop" / "vanilla" / f"qa_{split}.txt"
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing MetaQA question file: {source_path}")

            with source_path.open("r", encoding="utf-8") as handle:
                records = [
                    parse_question_line(
                        line, hop=hop, split=split, line_number=line_number
                    )
                    for line_number, line in enumerate(handle, start=1)
                ]

            expected = EXPECTED_COUNTS[hop][split]
            if len(records) != expected:
                raise ValueError(
                    f"Unexpected count for {hop}-hop {split}: "
                    f"expected {expected}, found {len(records)}"
                )

            output_path = benchmark_dir / f"metaqa_{hop}hop_{split}.jsonl"
            write_jsonl(output_path, records)
            counts[f"{hop}-hop/{split}"] = len(records)

    return counts


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    kb_path = raw_dir / "kb" / "kb.txt"
    if not kb_path.is_file():
        raise FileNotFoundError(f"Missing MetaQA knowledge base: {kb_path}")

    triples = read_triples(kb_path)
    statistics = {
        "source": "MetaQA",
        "knowledge_base": export_knowledge_base(triples, output_dir),
        "questions": normalize_questions(raw_dir, output_dir, args.splits),
    }
    statistics_path = output_dir / "metaqa_statistics.json"
    statistics_path.write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    print(f"\nPrepared MetaQA data in: {output_dir}")


if __name__ == "__main__":
    main()
