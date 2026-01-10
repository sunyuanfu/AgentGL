import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class LinkPredictionSource:
    dataset: str
    split: str
    path: str
    base_offset: int
    pair_count: int
    dataset_dir: str

    @property
    def key(self) -> Tuple[str, str]:
        return (self.dataset, self.split)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _scan_jsonl(path: Path) -> Tuple[str, str, int]:
    dataset = ""
    split = ""
    count = 0
    for count, record in enumerate(_iter_jsonl(path), start=1):
        dataset = str(record.get("dataset") or dataset)
        split = str(record.get("split") or split or "train")
    if count == 0:
        raise ValueError(f"{path} is empty; expected link prediction JSONL content")
    return dataset, split, count


def build_lp_sources(graph_data_dir: Optional[str]) -> List[LinkPredictionSource]:
    if not graph_data_dir:
        return []
    paths: List[Path] = []
    for raw in graph_data_dir.split(","):
        raw = raw.strip()
        if not raw:
            continue
        path = Path(raw)
        if path.is_dir():
            raise ValueError(f"{path} is a directory. Pass explicit JSONL files for link prediction.")
        if not path.is_file():
            raise FileNotFoundError(f"Link prediction file not found: {path}")
        paths.append(path)

    sources: List[LinkPredictionSource] = []
    seen: Dict[Tuple[str, str], Path] = {}
    base = 0
    for path in paths:
        dataset, split, count = _scan_jsonl(path)
        if not dataset:
            dataset = path.parent.name
        key = (dataset, split)
        if key in seen:
            raise ValueError(
                f"Duplicate link prediction source for dataset={dataset}, split={split}: {seen[key]} vs {path}"
            )
        sources.append(
            LinkPredictionSource(
                dataset=dataset,
                split=split,
                path=str(path),
                base_offset=base,
                pair_count=count,
                dataset_dir=str(path.parent),
            )
        )
        base += count
        seen[key] = path
    return sources


def ensure_lp_sources(args) -> List[LinkPredictionSource]:
    cached = getattr(args, "_lp_sources", None)
    if cached is not None:
        return cached
    sources = build_lp_sources(getattr(args, "graph_data_dir", None))
    setattr(args, "_lp_sources", sources)
    logger.info(
        "Registered %d link prediction sources with %d total samples",
        len(sources),
        sum(src.pair_count for src in sources),
    )
    return sources


_NODE_TEXT_CACHE: Dict[str, List[str]] = {}


def _load_node_texts(dataset: str, node_data_root: Optional[str]) -> List[str]:
    if dataset in _NODE_TEXT_CACHE:
        return _NODE_TEXT_CACHE[dataset]
    if not node_data_root:
        raise ValueError("lp_node_data_root must be provided for link prediction mode")
    base = Path(node_data_root)
    candidates = [base / dataset, base / dataset.replace("-", "_")]
    node_path = None
    for candidate in candidates:
        candidate_file = candidate / "node_texts.json"
        if candidate_file.is_file():
            node_path = candidate_file
            break
    if node_path is None:
        raise FileNotFoundError(f"node_texts.json not found for dataset '{dataset}' under {node_data_root}")
    with node_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
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
    _NODE_TEXT_CACHE[dataset] = texts
    return texts


def get_node_summary(dataset: str, node_id: int, node_data_root: Optional[str]) -> str:
    if node_id is None or node_id < 0:
        return "No node id provided."
    try:
        texts = _load_node_texts(dataset, node_data_root)
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning("Falling back to ID string for %s node %s (%s)", dataset, node_id, exc)
        return f"Node {node_id}"
    if node_id >= len(texts):
        return f"Node {node_id}"
    text = texts[node_id]
    return text if text else f"Node {node_id}"


def global_pair_id(record: dict, sources: List[LinkPredictionSource]) -> int:
    dataset = str(record.get("dataset"))
    split = str(record.get("split"))
    if not dataset:
        raise ValueError("Record missing 'dataset' for link prediction data")
    pair_id = record.get("pair_id")
    if pair_id is None:
        raise ValueError("Record missing 'pair_id'; rerun build_lp_datasets.py to populate it")
    for src in sources:
        if src.dataset == dataset and src.split == split:
            return src.base_offset + int(pair_id)
    raise KeyError(f"No source registered for dataset={dataset}, split={split}")
