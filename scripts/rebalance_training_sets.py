#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise or rebalance stage1/2 training JSONLs.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="ogbn-arxiv,ogbn-products",
        help="Comma separated dataset names (default: %(default)s)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/training_set"),
        help="Directory containing original stage jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training_set_3b"),
        help="Directory to store rebalanced jsonl files.",
    )
    parser.add_argument(
        "--plan-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON file describing desired sample counts per dataset/stage/difficulty. "
            "If omitted, the script only prints current statistics."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling records into new splits.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting existing output jsonl files.",
    )
    return parser.parse_args()


def find_stage_file(input_dir: Path, dataset: str, stage_name: str) -> Path:
    pattern = f"{dataset}_{stage_name}_*.jsonl"
    matches = sorted(input_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} under {input_dir}")
    if len(matches) > 1:
        print(f"[warn] Multiple matches for {dataset}/{stage_name}, using {matches[-1]}")
    return matches[-1]


def load_records(path: Path) -> List[dict]:
    data: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def count_difficulties(records: List[dict]) -> Counter:
    counter = Counter()
    for rec in records:
        counter[rec.get("difficulty", "unknown")] += 1
    return counter


def print_summary(dataset: str, stage: str, counter: Counter) -> None:
    total = sum(counter.values())
    parts = ", ".join(f"{diff}={counter.get(diff, 0)}" for diff in ("easy", "medium", "hard"))
    print(f"{dataset} {stage}: total={total} ({parts})")


def load_plan(plan_path: Path) -> Dict[str, Dict[str, Dict[str, int]]]:
    with plan_path.open("r", encoding="utf-8") as f:
        plan = json.load(f)
    return plan


def rebalance_dataset(
    dataset: str,
    pooled: Dict[str, List[dict]],
    stage_plan: Dict[str, Dict[str, int]],
    rng: random.Random,
) -> Dict[str, List[dict]]:
    pools = {diff: list(records) for diff, records in pooled.items()}
    for records in pools.values():
        rng.shuffle(records)

    stage_outputs: Dict[str, List[dict]] = {}
    for stage_name in ("stage1", "stage2"):
        targets = stage_plan.get(stage_name, {})
        selected: List[dict] = []
        for diff, records in pools.items():
            need = int(targets.get(diff, 0))
            if need > len(records):
                raise ValueError(
                    f"{dataset}/{stage_name}: need {need} samples for '{diff}' but only {len(records)} available"
                )
            selected.extend(records[:need])
            pools[diff] = records[need:]
        stage_outputs[stage_name] = selected
    return stage_outputs


def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if not datasets:
        raise ValueError("No datasets provided.")

    input_dir = args.input_dir
    output_dir = args.output_dir
    plan = load_plan(args.plan_file) if args.plan_file else None
    rng = random.Random(args.seed)

    staged_data: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))

    for dataset in datasets:
        stage1_path = find_stage_file(input_dir, dataset, "stage1")
        stage2_path = find_stage_file(input_dir, dataset, "stage2")

        stage1_records = load_records(stage1_path)
        stage2_records = load_records(stage2_path)

        print_summary(dataset, "stage1", count_difficulties(stage1_records))
        print_summary(dataset, "stage2", count_difficulties(stage2_records))

        combined: Dict[str, List[dict]] = defaultdict(list)
        for rec in stage1_records + stage2_records:
            combined[rec.get("difficulty", "unknown")].append(rec)
        staged_data[dataset] = combined

    if not plan:
        print("\nNo plan file provided; summary only.")
        return

    for dataset in datasets:
        if dataset not in plan:
            raise ValueError(f"Plan missing dataset '{dataset}'")
        stage_plan = plan[dataset]
        outputs = rebalance_dataset(dataset, staged_data[dataset], stage_plan, rng)

        for stage_name, records in outputs.items():
            out_path = output_dir / f"{dataset}_{stage_name}_rebalanced.jsonl"
            if out_path.exists() and not args.force:
                raise FileExistsError(f"{out_path} already exists. Use --force to overwrite.")
            write_jsonl(out_path, records)
            summary = count_difficulties(records)
            print_summary(dataset, f"{stage_name} (new)", summary)
            print(f"  -> wrote {len(records)} samples to {out_path}")


if __name__ == "__main__":
    main()
