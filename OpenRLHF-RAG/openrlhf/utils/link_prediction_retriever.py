from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openrlhf.utils.logging_utils import init_logger
from .link_prediction_utils import ensure_lp_sources, get_node_summary

logger = init_logger(__name__)


def _detect_search_type(query: str) -> Tuple[str, str]:
    if query is None:
        return "similar", ""
    query = query.strip()
    if not query:
        return "similar", ""

    import re

    match = re.match(r"^([a-zA-Z0-9\- ]+?):(.*)$", query)
    if not match:
        return "similar", query.strip()

    prefix = match.group(1).strip().lower()
    remainder = match.group(2).strip()

    if prefix in {"1-hop", "1 hop", "1hop", "one-hop", "one hop"}:
        return "one_hop", remainder
    if prefix in {"2-hop", "2 hop", "2hop", "two-hop", "two hop"}:
        return "two_hop", remainder
    if prefix in {"pagerank", "page rank", "pr"}:
        return "pagerank", remainder
    if prefix in {"similar", "sim", "similarity"}:
        return "similar", remainder
    return "similar", remainder if remainder else query


def _preview(text: Optional[str], limit: int = 480) -> str:
    if not text:
        return ""
    t = str(text).strip()
    return t if len(t) <= limit else t[: limit - 3] + "..."


def _unique_ids(values: List[int]) -> List[int]:
    seen = set()
    ordered: List[int] = []
    for val in values:
        if not isinstance(val, int):
            continue
        if val in seen:
            continue
        seen.add(val)
        ordered.append(int(val))
    return ordered


def _load_pagerank_pairs(dataset_dir: str) -> List[dict]:
    path = Path(dataset_dir) / "pagerank_top_pairs.jsonl"
    if not path.is_file():
        return []
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class LinkPredictionRetriever:
    """Serve graph-search documents for link prediction pairs."""

    def __init__(self, args) -> None:
        sources = ensure_lp_sources(args)
        if not sources:
            raise ValueError("--graph_data_dir must list link-prediction JSONL files when graph_task=link")
        self.sources = sources
        self.node_data_root = getattr(args, "lp_node_data_root", None)
        self.default_topk = getattr(args, "graph_topk", 5)
        self.topk_similar = getattr(args, "graph_topk_similar", None)
        self.topk_one_hop = getattr(args, "graph_topk_one_hop", None)
        self.topk_two_hop = getattr(args, "graph_topk_two_hop", None)
        self.topk_pagerank = getattr(args, "graph_topk_pagerank", None)
        self.max_searches = getattr(args, "graph_max_searches", 5)
        self.seed = getattr(args, "lp_neighbor_seed", 20240101)

        self.records: Dict[int, dict] = {}
        self.pagerank_cache: Dict[str, List[dict]] = {}
        for src in self.sources:
            dataset_records = []
            with Path(src.path).open("r", encoding="utf-8") as fp:
                for idx, line in enumerate(fp):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    pair_id = int(record.get("pair_id", idx))
                    global_id = src.base_offset + pair_id
                    dataset_records.append(record)
                    self.records[global_id] = record
            if not dataset_records:
                raise ValueError(f"No records loaded from {src.path}")
            if src.dataset not in self.pagerank_cache:
                self.pagerank_cache[src.dataset] = _load_pagerank_pairs(src.dataset_dir)

        self.num_nodes = len(self.records)
        logger.info("LinkPredictionRetriever loaded %d samples from %d files", self.num_nodes, len(self.sources))

    def _resolve_topk(self, strategy: str, override: Optional[int]) -> int:
        if override is not None:
            try:
                return max(int(override), 0)
            except (TypeError, ValueError):
                pass
        mapping = {
            "similar": self.topk_similar,
            "one_hop": self.topk_one_hop,
            "two_hop": self.topk_two_hop,
            "pagerank": self.topk_pagerank,
        }
        val = mapping.get(strategy)
        if val is not None:
            try:
                return max(int(val), 0)
            except (TypeError, ValueError):
                return self.default_topk
        return max(int(self.default_topk), 0)

    def _build_neighbor_docs(
        self,
        record: dict,
        level: str,
        top_k: int,
    ) -> List[Dict[str, str]]:
        if top_k <= 0:
            return []
        if level == "one_hop":
            common = _unique_ids(record.get("common_1hop", []))
            u_pool = _unique_ids(record.get("node_u_1hop", []))
            v_pool = _unique_ids(record.get("node_v_1hop", []))
            label = "1-hop"
        else:
            common = _unique_ids(record.get("common_2hop", []))
            u_pool = _unique_ids(record.get("node_u_2hop", []))
            v_pool = _unique_ids(record.get("node_v_2hop", []))
            label = "2-hop"
        u_id = int(record.get("node_u", -1))
        v_id = int(record.get("node_v", -1))
        dataset = str(record.get("dataset"))
        rng = random.Random(self.seed + int(record.get("pair_id", 0)) + (17 if level == "two_hop" else 0))

        docs: List[Tuple[str, int]] = []
        for node in common[:top_k]:
            docs.append(("common", node))
        remaining = top_k - len(docs)
        if remaining <= 0:
            return [self._neighbor_doc(dataset, node, role, level, u_id, v_id, idx) for idx, (role, node) in enumerate(docs, start=1)]

        u_candidates = [n for n in u_pool if n not in common and n != v_id]
        v_candidates = [n for n in v_pool if n not in common and n != u_id]
        rng.shuffle(u_candidates)
        rng.shuffle(v_candidates)
        quota_u = (remaining + 1) // 2
        quota_v = remaining // 2

        selected_u = u_candidates[:quota_u]
        selected_v = v_candidates[:quota_v]

        shortfall = quota_u - len(selected_u)
        if shortfall > 0:
            selected_v.extend(v_candidates[quota_v : quota_v + shortfall])
        shortfall = quota_v - len(selected_v)
        if shortfall > 0:
            selected_u.extend(u_candidates[quota_u : quota_u + shortfall])

        for node in selected_u:
            docs.append(("u", node))
            if len(docs) >= top_k:
                break
        for node in selected_v:
            if len(docs) >= top_k:
                break
            docs.append(("v", node))

        return [
            self._neighbor_doc(dataset, node, role, level, u_id, v_id, idx)
            for idx, (role, node) in enumerate(docs, start=1)
        ]

    def _neighbor_doc(
        self,
        dataset: str,
        node_id: int,
        role: str,
        level: str,
        node_u: int,
        node_v: int,
        rank: int,
    ) -> Dict[str, str]:
        summary = (
            _preview(get_node_summary(dataset, node_id, self.node_data_root)) or "No additional description provided."
        )
        if role == "common":
            prefix = f"[common {level} neighbour]"
            relation = "This neighbour links both Node U and Node V."
        elif role == "u":
            prefix = f"[Node U {level} neighbour]"
            relation = "This neighbour only connects to Node U."
        else:
            prefix = f"[Node V {level} neighbour]"
            relation = "This neighbour only connects to Node V."
        text = f"{prefix} {summary} {relation}"
        return {"doc_id": rank, "text": text.strip()}

    def _similar_docs(self, record: dict, top_k: int) -> List[Dict[str, str]]:
        sims = record.get("similar_pairs") or []
        dataset = str(record.get("dataset"))
        docs: List[Dict[str, str]] = []
        for rank, pair in enumerate(sims[:top_k], start=1):
            a = int(pair.get("node_u", -1))
            b = int(pair.get("node_v", -1))
            edge = pair.get("is_edge", False)
            summary_a = _preview(get_node_summary(dataset, a, self.node_data_root)) or "No description available."
            summary_b = _preview(get_node_summary(dataset, b, self.node_data_root)) or "No description available."
            relation = "existing edge" if edge else "reference pair"
            text = (
                f"[similar pair] {relation}. "
                f"First node: {summary_a}. Second node: {summary_b}."
            )
            docs.append({"doc_id": rank, "text": text})
        return docs

    def _pagerank_docs(self, record: dict, top_k: int) -> List[Dict[str, str]]:
        dataset = str(record.get("dataset"))
        pagerank_rows = self.pagerank_cache.get(dataset, [])
        docs: List[Dict[str, str]] = []
        skip_nodes = {int(record.get("node_u", -1)), int(record.get("node_v", -1))}
        for row in pagerank_rows:
            u = int(row.get("node_u", -1))
            v = int(row.get("node_v", -1))
            if u in skip_nodes and v in skip_nodes:
                continue
            relation = "high PageRank reference edge" if row.get("is_edge") else "high-scoring PageRank candidate"
            summary_u = _preview(get_node_summary(dataset, u, self.node_data_root)) or "No description available."
            summary_v = _preview(get_node_summary(dataset, v, self.node_data_root)) or "No description available."
            docs.append(
                {
                    "doc_id": len(docs) + 1,
                    "text": (
                        f"[pagerank] {relation}. "
                        f"First node: {summary_u}. Second node: {summary_v}."
                    ),
                }
            )
            if len(docs) >= top_k:
                break
        return docs

    def _target_info(self, record: dict) -> Dict[str, object]:
        return {
            "pair_id": int(record.get("pair_id", -1)),
            "dataset": record.get("dataset"),
            "split": record.get("split"),
            "node_u": int(record.get("node_u", -1)),
            "node_v": int(record.get("node_v", -1)),
            "label": int(record.get("label", -1)),
        }

    def batch_query(
        self,
        node_ids: List[int],
        queries: List[str],
        max_results: Optional[int] = None,
    ) -> Dict[str, object]:
        if len(node_ids) != len(queries):
            raise ValueError("node_ids and queries must have the same length")
        strategies: List[str] = []
        score_names: List[str] = []
        targets: List[Dict[str, object]] = []
        results_batch: List[List[Dict[str, str]]] = []
        for node_id, query in zip(node_ids, queries):
            strategy, score_name, target, docs = self._search_single(node_id, query, max_results)
            strategies.append(strategy)
            score_names.append(score_name)
            targets.append(target)
            results_batch.append(docs)
        return {
            "success": True,
            "strategies": strategies,
            "score_names": score_names,
            "target_nodes": targets,
            "results_batch": results_batch,
        }

    def _search_single(
        self,
        global_id: int,
        raw_query: str,
        max_results: Optional[int],
    ) -> Tuple[str, str, Dict[str, object], List[Dict[str, str]]]:
        if global_id not in self.records:
            raise ValueError(f"pair id {global_id} not found in link-prediction retriever")
        record = self.records[global_id]
        strategy, _ = _detect_search_type(raw_query)
        score_name = "coverage" if strategy in {"one_hop", "two_hop"} else "reference"
        top_k = self._resolve_topk(strategy, max_results)
        if strategy == "one_hop":
            docs = self._build_neighbor_docs(record, "one_hop", top_k)
        elif strategy == "two_hop":
            docs = self._build_neighbor_docs(record, "two_hop", top_k)
        elif strategy == "pagerank":
            docs = self._pagerank_docs(record, top_k)
        else:
            strategy = "similar"
            docs = self._similar_docs(record, top_k)
        target = self._target_info(record)
        return strategy, score_name, target, docs


__all__ = ["LinkPredictionRetriever"]
