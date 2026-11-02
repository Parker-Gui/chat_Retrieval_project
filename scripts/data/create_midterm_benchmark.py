"""Create a deterministic 300-question benchmark from MetaQA test data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "shared" / "benchmark"
HOPS = (1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a fixed number of MetaQA test questions for each hop."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-hop", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.per_hop <= 0:
        raise ValueError("--per-hop must be greater than zero")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_records: list[dict] = []
    counts: dict[str, int] = {}
    for hop in HOPS:
        source_path = input_dir / f"metaqa_{hop}hop_test.jsonl"
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Missing normalized test data: {source_path}. "
                "Run scripts/data/prepare_metaqa.py first."
            )

        records = read_jsonl(source_path)
        if len(records) < args.per_hop:
            raise ValueError(
                f"Requested {args.per_hop} questions from {hop}-hop, "
                f"but only {len(records)} are available"
            )

        rng = random.Random(args.seed + hop)
        selected_indexes = sorted(rng.sample(range(len(records)), args.per_hop))
        selected = [records[index] for index in selected_indexes]
        write_jsonl(output_dir / f"midterm_{hop}hop.jsonl", selected)
        combined_records.extend(selected)
        counts[f"{hop}-hop"] = len(selected)

    benchmark_path = output_dir / "midterm_300.jsonl"
    write_jsonl(benchmark_path, combined_records)
    manifest = {
        "dataset": "MetaQA vanilla test split",
        "seed": args.seed,
        "questions_per_hop": args.per_hop,
        "total_questions": len(combined_records),
        "counts": counts,
        "benchmark_file": benchmark_path.name,
    }
    (output_dir / "midterm_300_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nCreated benchmark: {benchmark_path}")


if __name__ == "__main__":
    main()
