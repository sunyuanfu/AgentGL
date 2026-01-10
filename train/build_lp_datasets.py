import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


@dataclass
class PairSample:
    node_u: int
    node_v: int
    label: int
    split: str
    similarity: float = 0.0
    difficulty: Optional[str] = None
    pair_id: Optional[int] = None
    vector: Optional[np.ndarray] = None
    similar_pairs: List[Dict[str, object]] = field(default_factory=list)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_node_texts(path: Path) -> List[str]:
    raw = load_json(path)
    texts: List[str] = []
    if raw and isinstance(raw[0], dict):
        for item in raw:
            if not isinstance(item, dict):
                texts.append(str(item))
                continue
            summary = (
                item.get("summary_en")
                or item.get("summary")
                or item.get("text")
                or item.get("original")
                or ""
            )
            texts.append(str(summary).strip())
    else:
        texts = [str(x) for x in raw]
    return texts


def load_neighbor_list(path: Path) -> List[List[int]]:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a list")
    cleaned: List[List[int]] = []
    for entry in raw:
        if isinstance(entry, list):
            cleaned.append([int(x) for x in entry if isinstance(x, (int, float))])
        else:
            cleaned.append([])
    return cleaned


def resolve_edge_index(dataset: str, edge_dir: Path) -> Path:
    candidates = [
        edge_dir / f"{dataset}_edge_index.pt",
        edge_dir / f"{dataset}_edge_index.npy",
    ]
    alt = dataset.replace("-", "_")
    if alt != dataset:
        candidates.extend(
            [edge_dir / f"{alt}_edge_index.pt", edge_dir / f"{alt}_edge_index.npy"]
        )
    alt2 = dataset.replace("_", "-")
    if alt2 != dataset:
        candidates.extend(
            [edge_dir / f"{alt2}_edge_index.pt", edge_dir / f"{alt2}_edge_index.npy"]
        )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"edge_index file for dataset '{dataset}' not found in {edge_dir}")


def load_edge_index(path: Path) -> np.ndarray:
    if path.suffix == ".pt":
        tensor = torch.load(path, map_location="cpu")
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.load(path)
    if arr.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {arr.shape}")
    return arr


def canonical_edge_list(edge_index: np.ndarray) -> List[Tuple[int, int]]:
    edges: Set[Tuple[int, int]] = set()
    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)
    for u, v in zip(src, dst):
        if u == v:
            continue
        a, b = (int(u), int(v))
        if a > b:
            a, b = b, a
        edges.add((a, b))
    return sorted(edges)


def sample_positive_edges(
    edges: List[Tuple[int, int]],
    count: int,
    used: Set[Tuple[int, int]],
    rng: random.Random,
) -> List[Tuple[int, int]]:
    available = [edge for edge in edges if edge not in used]
    if len(available) < count:
        raise ValueError(f"Not enough positive edges to sample {count} pairs")
    rng.shuffle(available)
    selected = available[:count]
    used.update(selected)
    return selected


def sample_negative_pairs(
    num_nodes: int,
    count: int,
    forbidden: Set[Tuple[int, int]],
    used_negatives: Set[Tuple[int, int]],
    rng: random.Random,
) -> List[Tuple[int, int]]:
    selected: Set[Tuple[int, int]] = set()
    max_attempts = count * 50
    attempts = 0
    while len(selected) < count:
        if attempts > max_attempts:
            raise RuntimeError("Failed to sample enough negative pairs; consider fewer samples.")
        attempts += 1
        u, v = rng.sample(range(num_nodes), 2)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        edge = (a, b)
        if edge in forbidden or edge in used_negatives or edge in selected:
            continue
        selected.add(edge)
    used_negatives.update(selected)
    return list(selected)


def load_precomputed_embeddings(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{path} should have shape (N, D), got {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def encode_nodes(
    model: SentenceTransformer,
    node_texts: Sequence[str],
    node_ids: Set[int],
    batch_size: int,
) -> Dict[int, np.ndarray]:
    idx_list = sorted(node_ids)
    texts = [node_texts[i] if 0 <= i < len(node_texts) else "" for i in idx_list]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms
    return {idx: emb for idx, emb in zip(idx_list, embeddings)}


def assign_similarities(pairs: List[PairSample], embeddings: Dict[int, np.ndarray]) -> None:
    for pair in pairs:
        u_vec = embeddings.get(pair.node_u)
        v_vec = embeddings.get(pair.node_v)
        if u_vec is None or v_vec is None:
            raise ValueError(f"Missing embedding for nodes {pair.node_u} or {pair.node_v}")
        pair.similarity = float(np.dot(u_vec, v_vec))


def assign_difficulties(pairs: List[PairSample], is_positive: bool) -> None:
    if not pairs:
        return
    bucket = len(pairs) // 3 or len(pairs)
    sorted_pairs = sorted(pairs, key=lambda p: p.similarity, reverse=is_positive)
    cut1 = bucket
    cut2 = bucket * 2
    for idx, pair in enumerate(sorted_pairs):
        if idx < cut1:
            pair.difficulty = "easy"
        elif idx < cut2:
            pair.difficulty = "medium"
        else:
            pair.difficulty = "hard"


def shuffle_by_difficulty(pairs: List[PairSample], rng: random.Random) -> List[PairSample]:
    buckets: Dict[str, List[PairSample]] = {"easy": [], "medium": [], "hard": []}
    others: List[PairSample] = []
    for pair in pairs:
        key = pair.difficulty or "unknown"
        if key in buckets:
            buckets[key].append(pair)
        else:
            others.append(pair)
    ordered: List[PairSample] = []
    for key in ("easy", "medium", "hard"):
        rng.shuffle(buckets[key])
        ordered.extend(buckets[key])
    if others:
        rng.shuffle(others)
        ordered.extend(others)
    return ordered


def _unique_sorted(seq: Sequence[int]) -> List[int]:
    return sorted({int(x) for x in seq if isinstance(x, (int, float))})


def build_neighbor_payload(
    pair: PairSample,
    first_hop: Sequence[Sequence[int]],
    second_hop: Sequence[Sequence[int]],
) -> Dict[str, object]:
    u = pair.node_u
    v = pair.node_v
    u_1hop = _unique_sorted(first_hop[u]) if 0 <= u < len(first_hop) else []
    v_1hop = _unique_sorted(first_hop[v]) if 0 <= v < len(first_hop) else []
    u_2hop = _unique_sorted(second_hop[u]) if 0 <= u < len(second_hop) else []
    v_2hop = _unique_sorted(second_hop[v]) if 0 <= v < len(second_hop) else []
    common_1 = sorted(set(u_1hop) & set(v_1hop))
    common_2 = sorted(set(u_2hop) & set(v_2hop))
    return {
        "common_1hop": common_1,
        "common_2hop": common_2,
        "node_u_1hop": u_1hop,
        "node_u_2hop": u_2hop,
        "node_v_1hop": v_1hop,
        "node_v_2hop": v_2hop,
    }


def write_split_jsonl(
    dataset: str,
    split: str,
    output_dir: Path,
    pairs: Sequence[PairSample],
    first_hop: Sequence[Sequence[int]],
    second_hop: Sequence[Sequence[int]],
) -> None:
    if not pairs:
        return
    path = output_dir / dataset / f"{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            payload = build_neighbor_payload(pair, first_hop, second_hop)
            record = {
                "pair_id": int(pair.pair_id if pair.pair_id is not None else -1),
                "dataset": dataset,
                "split": split,
                "node_u": int(pair.node_u),
                "node_v": int(pair.node_v),
                "label": int(pair.label),
                "similarity": round(pair.similarity, 6),
                "difficulty": pair.difficulty,
                "similar_pairs": pair.similar_pairs,
            }
            record.update(payload)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def pagerank_top_pairs(
    dataset_dir: Path,
    dataset: str,
    edges: Sequence[Tuple[int, int]],
    pagerank_path: Path,
    exclude_edges: Set[Tuple[int, int]],
) -> None:
    if not pagerank_path.is_file():
        return
    scores = np.load(pagerank_path)
    scored_edges: List[Tuple[float, Tuple[int, int]]] = []
    for edge in edges:
        if edge in exclude_edges:
            continue
        u, v = edge
        if u >= len(scores) or v >= len(scores):
            continue
        avg_score = float((scores[u] + scores[v]) / 2.0)
        scored_edges.append((avg_score, edge))
    if not scored_edges:
        return
    scored_edges.sort(key=lambda x: x[0], reverse=True)
    top_edges = scored_edges[:10]
    out_path = dataset_dir / "pagerank_top_pairs.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for score, (u, v) in top_edges:
            record = {
                "dataset": dataset,
                "node_u": int(u),
                "node_v": int(v),
                "avg_pagerank": score,
                "node_u_pagerank": float(scores[u]),
                "node_v_pagerank": float(scores[v]),
                "is_edge": True,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_pair_samples(edges: Sequence[Tuple[int, int]], label: int, split: str) -> List[PairSample]:
    return [PairSample(node_u=u, node_v=v, label=label, split=split) for u, v in edges]


def compute_pair_vectors(pairs: List[PairSample], embeddings: Dict[int, np.ndarray]) -> np.ndarray:
    if not pairs:
        return np.zeros((0, 1), dtype=np.float32)
    dim = next(iter(embeddings.values())).shape[0]
    matrix = np.zeros((len(pairs), dim), dtype=np.float32)
    for idx, pair in enumerate(pairs):
        u_vec = embeddings[pair.node_u]
        v_vec = embeddings[pair.node_v]
        vec = (u_vec + v_vec) / 2.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        matrix[idx] = vec
        pair.vector = vec
    return matrix


def top_k_indices(row: np.ndarray, k: int) -> List[int]:
    if row.size == 0 or k <= 0:
        return []
    if row.size <= k:
        return list(np.argsort(-row))
    idx = np.argpartition(-row, k)[:k]
    idx = idx[np.argsort(-row[idx])]
    return idx.tolist()


def attach_similar_pairs(
    reference_pairs: List[PairSample],
    target_pairs: List[PairSample],
    top_k: int,
    desc: str,
) -> None:
    if not reference_pairs or not target_pairs:
        for pair in target_pairs:
            pair.similar_pairs = []
        return
    ref_matrix = np.stack([pair.vector for pair in reference_pairs])
    edge_to_index = {}
    for idx, pair in enumerate(reference_pairs):
        key = (min(pair.node_u, pair.node_v), max(pair.node_u, pair.node_v))
        edge_to_index[key] = idx
    tgt_matrix = np.stack([pair.vector for pair in target_pairs])
    sims = tgt_matrix @ ref_matrix.T
    for idx, pair in enumerate(tqdm(target_pairs, desc=desc, leave=False)):
        row = sims[idx]
        key = (min(pair.node_u, pair.node_v), max(pair.node_u, pair.node_v))
        if key in edge_to_index:
            row[edge_to_index[key]] = -np.inf
        idxs = top_k_indices(row, top_k)
        pair.similar_pairs = [
            {
                "pair_id": int(reference_pairs[j].pair_id if reference_pairs[j].pair_id is not None else -1),
                "split": reference_pairs[j].split,
                "node_u": int(reference_pairs[j].node_u),
                "node_v": int(reference_pairs[j].node_v),
                "label": int(reference_pairs[j].label),
                "similarity": float(row[j]),
                "is_edge": True,
            }
            for j in idxs
        ]


def discover_datasets(root: Path) -> List[str]:
    datasets: List[str] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "node_texts.json").is_file():
            datasets.append(entry.name)
    return datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build link prediction datasets with node-level similarity features.")
    parser.add_argument(
        "--node-data-root",
        type=str,
        default="/PATH/TO/WORKSPACE/node_data",
        help="Root directory containing dataset artifacts",
    )
    parser.add_argument(
        "--edge-dir",
        type=str,
        default="/PATH/TO/WORKSPACE/edge_indices",
        help="Directory containing original edge_index files (.pt or .npy)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/link_prediction",
        help="Directory to write dataset-specific JSONL files",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/PATH/TO/WORKSPACE/all-roberta-large-v1",
        help="SentenceTransformer model path",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Encoding device, e.g., cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=128, help="Encoding batch size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-datasets", type=str, default="", help="Comma-separated dataset names for training splits")
    parser.add_argument("--test-datasets", type=str, default="", help="Comma-separated dataset names for test splits")
    parser.add_argument("--train-num-positive", type=int, default=1500)
    parser.add_argument("--train-num-negative", type=int, default=1500)
    parser.add_argument("--test-num-positive", type=int, default=500)
    parser.add_argument("--test-num-negative", type=int, default=500)
    parser.add_argument("--similar-top-k", type=int, default=10, help="Top-K similar pairs to record per sample")
    return parser.parse_args()


def build_reference_pool(
    positive_pool: Sequence[Tuple[int, int]],
    test_positive_edges: Set[Tuple[int, int]],
    embeddings: Dict[int, np.ndarray],
) -> List[PairSample]:
    pairs: List[PairSample] = []
    for edge in positive_pool:
        if edge in test_positive_edges:
            continue
        sample = PairSample(node_u=edge[0], node_v=edge[1], label=1, split="reference")
        u_vec = embeddings.get(sample.node_u)
        v_vec = embeddings.get(sample.node_v)
        if u_vec is None or v_vec is None:
            continue
        vec = (u_vec + v_vec) / 2.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        sample.vector = vec
        pairs.append(sample)
    return pairs


def assign_pair_vectors(pairs: List[PairSample], embeddings: Dict[int, np.ndarray]) -> None:
    for pair in pairs:
        if pair.vector is not None:
            continue
        u_vec = embeddings.get(pair.node_u)
        v_vec = embeddings.get(pair.node_v)
        if u_vec is None or v_vec is None:
            continue
        vec = (u_vec + v_vec) / 2.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        pair.vector = vec


def process_dataset(
    dataset: str,
    node_root: Path,
    edge_dir: Path,
    output_dir: Path,
    model: SentenceTransformer,
    batch_size: int,
    rng: random.Random,
    build_train: bool,
    build_test: bool,
    cfg: argparse.Namespace,
) -> None:
    if not build_train and not build_test:
        return

    dataset_dir = node_root / dataset
    if not dataset_dir.is_dir():
        print(f"[WARN] Dataset directory missing: {dataset_dir}")
        return

    node_texts = load_node_texts(dataset_dir / "node_texts.json")
    first_hop = load_neighbor_list(dataset_dir / "first_hop_indices.json")
    second_hop = load_neighbor_list(dataset_dir / "second_hop_indices.json")
    pagerank_path = dataset_dir / "pagerank.npy"
    precomputed_emb = load_precomputed_embeddings(dataset_dir / "node_emb.npy")
    num_nodes = len(node_texts)

    edge_index_path = resolve_edge_index(dataset, edge_dir)
    edge_index = load_edge_index(edge_index_path)
    positive_pool = canonical_edge_list(edge_index)
    positive_set = set(positive_pool)

    used_positive_edges: Set[Tuple[int, int]] = set()
    used_negative_edges: Set[Tuple[int, int]] = set()

    train_pos_edges: List[Tuple[int, int]] = []
    train_neg_edges: List[Tuple[int, int]] = []
    test_pos_edges: List[Tuple[int, int]] = []
    test_neg_edges: List[Tuple[int, int]] = []

    if build_train:
        train_pos_edges = sample_positive_edges(
            positive_pool,
            cfg.train_num_positive,
            used_positive_edges,
            rng,
        )
        train_neg_edges = sample_negative_pairs(
            num_nodes,
            cfg.train_num_negative,
            positive_set,
            used_negative_edges,
            rng,
        )

    if build_test:
        test_pos_edges = sample_positive_edges(
            positive_pool,
            cfg.test_num_positive,
            used_positive_edges,
            rng,
        )
        test_neg_edges = sample_negative_pairs(
            num_nodes,
            cfg.test_num_negative,
            positive_set,
            used_negative_edges,
            rng,
        )

    train_pairs: List[PairSample] = []
    test_pairs: List[PairSample] = []

    if build_train:
        train_pos = build_pair_samples(train_pos_edges, label=1, split="train")
        train_neg = build_pair_samples(train_neg_edges, label=0, split="train")
        train_pairs = train_pos + train_neg

    if build_test:
        test_pos = build_pair_samples(test_pos_edges, label=1, split="test")
        test_neg = build_pair_samples(test_neg_edges, label=0, split="test")
        test_pairs = test_pos + test_neg

    if not train_pairs and not test_pairs:
        return

    needed_nodes: Set[int] = set()
    for pair in train_pairs + test_pairs:
        needed_nodes.add(pair.node_u)
        needed_nodes.add(pair.node_v)
    for edge in positive_pool:
        needed_nodes.add(edge[0])
        needed_nodes.add(edge[1])

    if precomputed_emb is not None:
        missing = [idx for idx in needed_nodes if idx >= precomputed_emb.shape[0]]
        if missing:
            raise ValueError(f"node_emb.npy for {dataset} missing embeddings for nodes {missing[:5]}...")
        embeddings = {idx: precomputed_emb[idx] for idx in needed_nodes}
    else:
        embeddings = encode_nodes(model, node_texts, needed_nodes, batch_size=batch_size)

    assign_similarities(train_pairs + test_pairs, embeddings)

    if train_pairs:
        assign_difficulties([p for p in train_pairs if p.label == 1], is_positive=True)
        assign_difficulties([p for p in train_pairs if p.label == 0], is_positive=False)
        train_pairs = shuffle_by_difficulty(train_pairs, rng)
        for idx, pair in enumerate(train_pairs):
            pair.pair_id = idx

    if test_pairs:
        rng.shuffle(test_pairs)
        for idx, pair in enumerate(test_pairs):
            pair.pair_id = idx

    assign_pair_vectors(train_pairs + test_pairs, embeddings)

    test_pos_set = set(test_pos_edges)
    reference_pool = build_reference_pool(positive_pool, test_pos_set, embeddings)

    attach_similar_pairs(reference_pool, train_pairs, cfg.similar_top_k, desc=f"{dataset} train similarity")
    attach_similar_pairs(reference_pool, test_pairs, cfg.similar_top_k, desc=f"{dataset} test similarity")

    write_split_jsonl(dataset, "train", output_dir, train_pairs, first_hop, second_hop)
    write_split_jsonl(dataset, "test", output_dir, test_pairs, first_hop, second_hop)

    test_edge_set = set(test_pos_edges)
    pagerank_top_pairs(output_dir / dataset, dataset, positive_pool, pagerank_path, test_edge_set)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    node_root = Path(args.node_data_root)
    edge_dir = Path(args.edge_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_datasets = discover_datasets(node_root)

    def parse_set(value: str) -> Set[str]:
        if not value:
            return set()
        return {name.strip() for name in value.split(",") if name.strip()}

    train_set = parse_set(args.train_datasets)
    test_set = parse_set(args.test_datasets)

    if not train_set and not test_set:
        train_set = set(all_datasets)

    model = SentenceTransformer(args.model_path, device=args.device)

    for dataset in all_datasets:
        build_train = dataset in train_set
        build_test = dataset in test_set
        process_dataset(
            dataset=dataset,
            node_root=node_root,
            edge_dir=edge_dir,
            output_dir=output_dir,
            model=model,
            batch_size=args.batch_size,
            rng=rng,
            build_train=build_train,
            build_test=build_test,
            cfg=args,
        )


if __name__ == "__main__":
    main()
