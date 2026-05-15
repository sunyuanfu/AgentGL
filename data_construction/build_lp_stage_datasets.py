#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set


def parse_difficulties(value: str) -> Set[str]:
    difficulties = {item.strip().lower() for item in value.split(",") if item.strip()}
    if not difficulties:
        raise ValueError("Difficulty list cannot be empty")
    return difficulties


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build LP stage JSONLs by filtering raw train pairs by difficulty."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="ogbn-arxiv,ogbn-products",
        help="Comma-separated dataset names.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/link_prediction"),
        help="Root directory containing {dataset}/train.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/link_prediction_stage"),
        help="Directory to store stage JSONL files.",
    )
    parser.add_argument(
        "--stage1-difficulties",
        type=str,
        default="easy,medium",
        help="Comma-separated difficulties included in stage1.",
    )
    parser.add_argument(
        "--stage2-difficulties",
        type=str,
        default="medium,hard",
        help="Comma-separated difficulties included in stage2.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def filter_records(path: Path, difficulties: Set[str]) -> List[dict]:
    records: List[dict] = []
    for rec in iter_jsonl(path):
        diff = str(rec.get("difficulty", "medium")).lower()
        if diff in difficulties:
            records.append(rec)
    return records


def write_jsonl(path: Path, records: List[dict], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists. Use --force to overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def describe(records: List[dict]) -> str:
    counts: Dict[str, int] = Counter(str(rec.get("difficulty", "unknown")).lower() for rec in records)
    parts = ", ".join(f"{key}={counts.get(key, 0)}" for key in ("easy", "medium", "hard"))
    return f"total={len(records)} ({parts})"


def main() -> None:
    args = parse_args()
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    if not datasets:
        raise ValueError("No datasets provided.")

    stage_specs = {
        "stage1": parse_difficulties(args.stage1_difficulties),
        "stage2": parse_difficulties(args.stage2_difficulties),
    }

    for dataset in datasets:
        src_path = args.input_root / dataset / "train.jsonl"
        if not src_path.is_file():
            raise FileNotFoundError(f"{src_path} not found")

        for stage_name, difficulties in stage_specs.items():
            records = filter_records(src_path, difficulties)
            out_path = args.output_dir / f"{dataset}_{stage_name}.jsonl"
            write_jsonl(out_path, records, force=args.force)
            print(f"{dataset} {stage_name}: {describe(records)} -> {out_path}")


if __name__ == "__main__":
    main()
