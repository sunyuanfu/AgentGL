import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_node_texts(path: Path) -> List[dict]:
    data = load_json(path)
    out: List[dict] = []
    for item in data:
        if isinstance(item, dict):
            out.append(item)
        else:
            out.append({"summary": str(item)})
    return out


def load_categories(path: Path) -> List[str]:
    data = load_json(path)
    if data and isinstance(data[0], list):
        data = [item[0] for item in data]
    return [str(item) for item in data]


def load_first_hop_indices(path: Path) -> List[Sequence[int]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("first_hop_indices.json should contain a list of neighbour indices")
    return data


def load_splits(path: Path) -> Dict[str, List[int]]:
    data = load_json(path)
    splits: Dict[str, List[int]] = {}
    for key, value in data.items():
        splits[key] = [int(v) for v in value]
    return splits


def wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    if trials <= 0:
        return 0.0
    phat = successes / trials
    denominator = 1.0 + (z ** 2) / trials
    centre = phat + (z ** 2) / (2 * trials)
    margin = z * math.sqrt((phat * (1 - phat) + (z ** 2) / (4 * trials)) / trials)
    value = (centre - margin) / denominator
    return max(0.0, min(1.0, value))


def compute_difficulty_labels(
    valid_indices: List[int],
    neighbours: List[Sequence[int]],
    labels: List[str],
) -> Dict[int, str]:
    if not neighbours:
        return {idx: "medium" for idx in valid_indices}

    scores: Dict[int, float] = {}
    max_degree = 0
    for idx in valid_indices:
        if idx < len(neighbours):
            max_degree = max(max_degree, len(neighbours[idx]))

    for idx in valid_indices:
        neigh = neighbours[idx] if idx < len(neighbours) else []
        degree = len(neigh)
        same_label = 0
        for nb in neigh:
            if isinstance(nb, int) and 0 <= nb < len(labels) and labels[nb] == labels[idx]:
                same_label += 1
        score = wilson_lower_bound(same_label, degree)
        if degree > 0 and max_degree > 0:
            score += 0.05 * (math.log1p(degree) / math.log1p(max_degree))
        scores[idx] = score

    if not scores:
        return {idx: "medium" for idx in valid_indices}

    sorted_indices = sorted(valid_indices, key=lambda i: scores[i])
    n = len(sorted_indices)
    if n < 3:
        return {idx: "medium" for idx in valid_indices}

    split_a = n // 3
    split_b = (2 * n) // 3
    buckets = {
        "hard": set(sorted_indices[:split_a]),
        "medium": set(sorted_indices[split_a:split_b]),
        "easy": set(sorted_indices[split_b:]),
    }
    return {idx: next(label for label, members in buckets.items() if idx in members) for idx in valid_indices}


def compute_offsets_and_sizes(node_data_root: Path, datasets: List[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    offsets: Dict[str, int] = {}
    sizes: Dict[str, int] = {}
    running = 0
    for ds in datasets:
        texts_path = node_data_root / ds / "node_texts.json"
        size = len(load_json(texts_path))
        offsets[ds] = running
        sizes[ds] = size
        running += size
    return offsets, sizes


def ensure_enough_candidates(
    candidates: Dict[str, List[Tuple[int, str]]],
    requested: Dict[str, int],
    dataset: str,
) -> None:
    for diff, count in requested.items():
        if count <= 0:
            continue
        available = len(candidates.get(diff, []))
        if count > available:
            raise ValueError(
                f"Dataset {dataset}: not enough {diff} samples (need {count}, have {available})"
            )


def sample_with_counts(
    candidates: Dict[str, List[Tuple[int, str]]],
    requested: Dict[str, int],
    rng: random.Random,
) -> List[Tuple[int, str, str]]:
    selected: List[Tuple[int, str, str]] = []
    for diff, count in requested.items():
        if count <= 0:
            continue
        pool = list(candidates.get(diff, []))
        rng.shuffle(pool)
        chosen = pool[:count]
        selected.extend((idx, summary, diff) for idx, summary in chosen)
    return selected


def load_stage_exclusions(paths: List[Path]) -> Dict[str, Set[int]]:
    exclusions: Dict[str, Set[int]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                dataset = record.get("dataset")
                if not dataset:
                    continue
                exclusions.setdefault(dataset, set()).add(int(record["node_id"]))
    return exclusions


def convert_to_local(ids: Set[int], offset: int, size: int) -> Set[int]:
    locals_: Set[int] = set()
    for nid in ids:
        if offset <= nid < offset + size:
            locals_.add(nid - offset)
        elif 0 <= nid < size:
            locals_.add(nid)
    return locals_


def stage1_for_dataset(
    node_data_root: Path,
    dataset: str,
    offset: int,
    counts: Dict[str, int],
    rng: random.Random,
) -> Tuple[List[dict], Dict[str, int]]:
    ddir = node_data_root / dataset
    records = load_node_texts(ddir / "node_texts.json")
    categories = load_categories(ddir / "category.json")
    neighbours = load_first_hop_indices(ddir / "first_hop_indices.json")
    splits = load_splits(ddir / "splits.json")

    train_ids = set(int(i) for i in splits.get("train", []))
    summaries: List[Tuple[int, str]] = []
    for idx, rec in enumerate(records):
        summary = rec.get("summary") or rec.get("summary_en") or rec.get("original") or ""
        summary = str(summary).strip()
        if summary:
            summaries.append((idx, summary))

    valid_indices = [idx for idx, _ in summaries]
    difficulty_labels = compute_difficulty_labels(valid_indices, neighbours, categories)

    grouped: Dict[str, List[Tuple[int, str]]] = {"easy": [], "medium": [], "hard": []}
    for idx, summary in summaries:
        if idx not in train_ids:
            continue
        diff = difficulty_labels.get(idx, "medium")
        grouped.setdefault(diff, []).append((idx, summary))

    ensure_enough_candidates(grouped, counts, dataset)
    picked = sample_with_counts(grouped, counts, rng)

    jsonl_records: List[dict] = []
    for local_idx, summary, diff in picked:
        jsonl_records.append(
            {
                "node_id": offset + local_idx,
                "summary_en": summary,
                "dataset": dataset,
                "difficulty": diff,
            }
        )

    stats = {f"{diff}_count": counts.get(diff, 0) for diff in counts}
    stats["count"] = len(jsonl_records)
    return jsonl_records, stats


def stage2_for_dataset(
    node_data_root: Path,
    dataset: str,
    offset: int,
    dataset_size: int,
    counts: Dict[str, int],
    stage1_global: Set[int],
    eval_local: Set[int],
    rng: random.Random,
) -> Tuple[List[dict], Dict[str, int]]:
    ddir = node_data_root / dataset
    records = load_node_texts(ddir / "node_texts.json")
    categories = load_categories(ddir / "category.json")
    neighbours = load_first_hop_indices(ddir / "first_hop_indices.json")
    splits = load_splits(ddir / "splits.json")

    train_ids = set(int(i) for i in splits.get("train", []))
    stage1_local = convert_to_local(stage1_global, offset, dataset_size)
    blocked = set(stage1_local)
    blocked.update(eval_local)

    summaries: List[Tuple[int, str]] = []
    for idx, rec in enumerate(records):
        summary = rec.get("summary") or rec.get("summary_en") or rec.get("original") or ""
        summary = str(summary).strip()
        if summary:
            summaries.append((idx, summary))

    valid_indices = [idx for idx, _ in summaries]
    difficulty_labels = compute_difficulty_labels(valid_indices, neighbours, categories)

    grouped: Dict[str, List[Tuple[int, str]]] = {"medium": [], "hard": []}
    for idx, summary in summaries:
        if idx not in train_ids or idx in blocked:
            continue
        diff = difficulty_labels.get(idx, "medium")
        if diff not in grouped:
            continue
        grouped[diff].append((idx, summary))

    ensure_enough_candidates(grouped, counts, dataset)
    picked = sample_with_counts(grouped, counts, rng)

    jsonl_records: List[dict] = []
    for local_idx, summary, diff in picked:
        jsonl_records.append(
            {
                "node_id": offset + local_idx,
                "summary_en": summary,
                "dataset": dataset,
                "difficulty": diff,
            }
        )

    stats = {f"{diff}_count": counts.get(diff, 0) for diff in counts}
    stats["count"] = len(jsonl_records)
    stats["excluded"] = len(blocked)
    return jsonl_records, stats


def eval_for_dataset(
    node_data_root: Path,
    dataset: str,
    allowed_split: str,
    sample_size: int,
    rng: random.Random,
) -> Tuple[List[dict], Dict[str, int]]:
    ddir = node_data_root / dataset
    records = load_node_texts(ddir / "node_texts.json")
    splits = load_splits(ddir / "splits.json")

    allowed_ids = set(int(i) for i in splits.get(allowed_split, []))
    if not allowed_ids:
        raise ValueError(f"Dataset {dataset}: split '{allowed_split}' is empty")

    available: List[int] = []
    for idx in allowed_ids:
        if 0 <= idx < len(records):
            rec = records[idx]
            summary = rec.get("summary") or rec.get("summary_en") or rec.get("original") or ""
            if str(summary).strip():
                available.append(idx)

    if not available:
        raise ValueError(f"Dataset {dataset}: no summaries available in split '{allowed_split}'")

    pick = min(sample_size, len(available))
    chosen = rng.sample(available, pick)

    jsonl_records: List[dict] = []
    for idx in chosen:
        rec = records[idx]
        summary = rec.get("summary") or rec.get("summary_en") or rec.get("original") or ""
        jsonl_records.append(
            {
                "node_id": idx,
                "summary_en": str(summary).strip(),
                "dataset": dataset,
            }
        )

    stats = {"count": len(jsonl_records)}
    return jsonl_records, stats


def resolve_output_path(base_dir: Path, dataset: str, mode: str, total: int) -> Path:
    if mode == "stage1":
        filename = f"{dataset}_stage1_{total}.jsonl"
    elif mode == "stage2":
        filename = f"{dataset}_stage2_{total}.jsonl"
    else:
        filename = f"{dataset}.jsonl"
    return base_dir / filename


def read_eval_exclusions(paths: List[Path], dataset: str) -> Set[int]:
    excluded: Set[int] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("dataset") not in {None, dataset}:
                    continue
                excluded.add(int(record["node_id"]))
    return excluded


def collect_eval_paths(eval_dir: Path, eval_files: List[Path], datasets: List[str]) -> List[Path]:
    paths: List[Path] = []
    for path in eval_files:
        paths.append(path)
    if eval_dir:
        for dataset in datasets:
            paths.append(eval_dir / f"{dataset}.jsonl")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate graph datasets using splits.json constraints")
    parser.add_argument("--node-data-root", type=str, default="/PATH/TO/WORKSPACE/node_data")
    parser.add_argument("--datasets", type=str, required=True, help="Comma-separated dataset names")
    parser.add_argument("--mode", choices=["stage1", "stage2", "eval"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--stage1-easy", type=int, default=1000)
    parser.add_argument("--stage1-medium", type=int, default=500)
    parser.add_argument("--stage1-hard", type=int, default=500)
    parser.add_argument("--stage2-medium", type=int, default=500)
    parser.add_argument("--stage2-hard", type=int, default=500)
    parser.add_argument("--stage1-files", type=str, default="", help="Comma-separated stage1 JSONL paths to exclude")
    parser.add_argument("--eval-dir", type=str, default="", help="Directory containing eval JSONLs to exclude when building stage2")
    parser.add_argument("--eval-files", type=str, default="", help="Comma-separated evaluation JSONL paths to exclude when building stage2")
    parser.add_argument("--eval-split", type=str, default="test", help="Split name to use when generating eval sets")
    parser.add_argument("--eval-size", type=int, default=1000)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if not datasets:
        raise ValueError("No datasets specified")

    node_data_root = Path(args.node_data_root)
    if not node_data_root.is_dir():
        raise FileNotFoundError(f"node_data_root not found: {node_data_root}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if args.mode == "eval":
            output_dir = Path("data/eval_set")
        else:
            output_dir = Path("data/training_set")
    output_dir.mkdir(parents=True, exist_ok=True)

    offsets: Dict[str, int] = {}
    sizes: Dict[str, int] = {}
    if args.mode in {"stage1", "stage2"}:
        offsets, sizes = compute_offsets_and_sizes(node_data_root, datasets)

    rng = random.Random(args.seed)

    if args.mode == "stage1":
        counts = {"easy": args.stage1_easy, "medium": args.stage1_medium, "hard": args.stage1_hard}
        total = sum(counts.values())
        for dataset in datasets:
            records, stats = stage1_for_dataset(node_data_root, dataset, offsets[dataset], counts, rng)
            out_path = resolve_output_path(output_dir, dataset, args.mode, total)
            with out_path.open("w", encoding="utf-8") as f:
                for rec in records:
                    json.dump(rec, f, ensure_ascii=False)
                    f.write("\n")
            print(
                f"Stage1 {dataset}: wrote {len(records)} samples to {out_path} "
                f"(easy={counts['easy']}, medium={counts['medium']}, hard={counts['hard']})"
            )
        return

    if args.mode == "stage2":
        stage1_paths = [Path(p.strip()) for p in args.stage1_files.split(",") if p.strip()]
        stage1_map = load_stage_exclusions(stage1_paths)

        eval_dir = Path(args.eval_dir) if args.eval_dir else None
        if eval_dir and not eval_dir.is_dir():
            raise FileNotFoundError(f"eval_dir not found: {eval_dir}")
        eval_file_paths = [Path(p.strip()) for p in args.eval_files.split(",") if p.strip()]
        eval_paths = collect_eval_paths(eval_dir, eval_file_paths, datasets)

        counts = {"medium": args.stage2_medium, "hard": args.stage2_hard}
        total = sum(counts.values())
        for dataset in datasets:
            stage1_global = stage1_map.get(dataset, set())
            eval_local = read_eval_exclusions(eval_paths, dataset)
            records, stats = stage2_for_dataset(
                node_data_root,
                dataset,
                offsets[dataset],
                sizes[dataset],
                counts,
                stage1_global,
                eval_local,
                rng,
            )
            out_path = resolve_output_path(output_dir, dataset, args.mode, total)
            with out_path.open("w", encoding="utf-8") as f:
                for rec in records:
                    json.dump(rec, f, ensure_ascii=False)
                    f.write("\n")
            print(
                f"Stage2 {dataset}: wrote {len(records)} samples to {out_path} "
                f"(medium={counts['medium']}, hard={counts['hard']}, excluded={stats['excluded']})"
            )
        return

    # eval mode
    for dataset in datasets:
        records, stats = eval_for_dataset(
            node_data_root,
            dataset,
            args.eval_split,
            args.eval_size,
            rng,
        )
        out_path = resolve_output_path(output_dir, dataset, args.mode, stats["count"])
        with out_path.open("w", encoding="utf-8") as f:
            for rec in records:
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")
        print(
            f"Eval {dataset}: wrote {len(records)} samples to {out_path} "
            f"(split={args.eval_split})"
        )


if __name__ == "__main__":
    main()
