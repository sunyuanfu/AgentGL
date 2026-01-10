#!/usr/bin/env python3
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stage1/stage2 LP training JSONLs following a difficulty plan.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="ogbn-arxiv,ogbn-products",
        help="Comma separated dataset names (default: %(default)s)",
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
        help="Directory to store generated stage jsonl files.",
    )
    parser.add_argument(
        "--plan-file",
        type=Path,
        required=True,
        help="JSON plan describing desired counts per dataset/stage/difficulty.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[dict]:
    data: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_plan(path: Path) -> Dict[str, Dict[str, Dict[str, int]]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rebalance_lp_dataset(
    dataset: str,
    records: List[dict],
    stage_plan: Dict[str, Dict[str, int]],
    rng: random.Random,
) -> Dict[str, List[dict]]:
    pools: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        diff = str(rec.get("difficulty", "unknown")).lower()
        pools[diff].append(rec)
    for pool in pools.values():
        rng.shuffle(pool)

    outputs: Dict[str, List[dict]] = {}
    for stage_name, counts in stage_plan.items():
        bucket: List[dict] = []
        for diff, need in counts.items():
            available = pools[diff]
            if need > len(available):
                raise ValueError(
                    f"{dataset}/{stage_name}: need {need} '{diff}' samples but only {len(available)} available"
                )
            bucket.extend(available[:need])
            pools[diff] = available[need:]
        outputs[stage_name] = bucket
    return outputs


def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def describe_counts(records: List[dict]) -> str:
    counts: Dict[str, int] = defaultdict(int)
    for rec in records:
        counts[str(rec.get("difficulty", "unknown")).lower()] += 1
    parts = [f"{k}={counts.get(k,0)}" for k in ("easy", "medium", "hard")]
    return f"total={len(records)} ({', '.join(parts)})"


def main() -> None:
    args = parse_args()
    plan = load_plan(args.plan_file)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    rng = random.Random(args.seed)

    for dataset in datasets:
        if dataset not in plan:
            raise ValueError(f"Plan missing dataset '{dataset}'")
        src_path = args.input_root / dataset / "train.jsonl"
        if not src_path.is_file():
            raise FileNotFoundError(f"{src_path} not found")
        records = load_records(src_path)
        outputs = rebalance_lp_dataset(dataset, records, plan[dataset], rng)
        for stage_name, bucket in outputs.items():
            out_path = args.output_dir / f"{dataset}_{stage_name}.jsonl"
            if out_path.exists() and not args.force:
                raise FileExistsError(f"{out_path} exists. Use --force to overwrite.")
            write_jsonl(out_path, bucket)
            print(f"{dataset} {stage_name}: {describe_counts(bucket)} -> {out_path}")


if __name__ == "__main__":
    main()
