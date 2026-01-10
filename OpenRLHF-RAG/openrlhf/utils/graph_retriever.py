"""Graph-based retrieval utilities used during RL rollouts.

This module provides a lightweight in-process alternative to the original
HTTP graph retrieval server. It loads the preprocessed graph artifacts
(`node_texts.json`, `category.json`, neighbour indices, optional PageRank
scores, and cached sentence embeddings) and exposes a `GraphRetriever` that
can be queried directly from the PPO experience maker. The implementation
mirrors the behaviour of the user's prior Flask server so that existing
prompting and reward logic keep working unchanged.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import requests

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - dependency validated at runtime
    SentenceTransformer = None  # type: ignore[assignment]


def _preview_text(text: Optional[str], limit: int) -> str:
    if not text:
        return ""
    text = str(text)
    return text[:limit] + "..." if len(text) > limit else text


def _l2_normalize(mat: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return mat / norm


def _safe_candidate_set(
    indices: Iterable[int],
    num_nodes: int,
    exclude: Optional[int] = None,
    allowed: Optional[Set[int]] = None,
) -> List[int]:
    candidates = []
    seen = set()
    for idx in indices:
        if not isinstance(idx, int):
            continue
        if idx < 0 or idx >= num_nodes:
            continue
        if exclude is not None and idx == exclude:
            continue
        if allowed is not None and idx not in allowed:
            continue
        if idx in seen:
            continue
        seen.add(idx)
        candidates.append(idx)
    return candidates


def _topk_by_similarity(
    query_emb: np.ndarray,
    candidate_ids: List[int],
    node_emb: np.ndarray,
    k: int,
) -> List[Tuple[int, float]]:
    if not candidate_ids or k <= 0:
        return []
    cand_mat = node_emb[candidate_ids]
    sims = (cand_mat @ query_emb.reshape(-1, 1)).reshape(-1)
    if sims.size == 0:
        return []
    order = np.argsort(-sims)[:k]
    return [(candidate_ids[i], float(sims[i])) for i in order.tolist()]


def _detect_search_type(query: str) -> Tuple[str, str]:
    """Parse the search type prefix from the raw query text."""

    if query is None:
        return "similar", ""
    query = query.strip()
    if not query:
        return "similar", ""

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

    # default fallback when the prefix is unrecognised
    return "similar", remainder if remainder else query


@dataclass
class GraphRetrieverConfig:
    data_dir: str
    encoder_path: Optional[str] = None
    encode_batch_size: int = 128
    fusion_alpha: float = 0.5
    default_max_results: int = 5
    topk_similar: Optional[int] = None
    topk_one_hop: Optional[int] = None
    topk_two_hop: Optional[int] = None
    topk_pagerank: Optional[int] = None
    preview_len: int = 500
    target_preview_len: int = 120
    encoder_device: str = "cpu"
    encoder_remote_url: Optional[str] = None
    encoder_timeout: float = 60.0


class GraphRetriever:
    """In-memory graph retrieval facade."""

    def __init__(self, config: GraphRetrieverConfig) -> None:
        self.config = config
        self.remote_encoder_url = config.encoder_remote_url
        if self.remote_encoder_url and not self.remote_encoder_url.endswith("/encode"):
            self.remote_encoder_url = self.remote_encoder_url.rstrip("/") + "/encode"
        self.encoder_timeout = config.encoder_timeout
        self.encoder: Optional[SentenceTransformer] = None
        self._http_session: Optional[requests.Session] = None
        self._load_graph()
        self._load_embeddings()

    def _init_local_encoder(self) -> None:
        if self.remote_encoder_url is not None:
            return
        if self.encoder is not None:
            return
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required to encode queries locally. "
                "Install it or provide --graph_encoder_remote_url."
            )
        if not self.config.encoder_path:
            raise ValueError(
                "encoder_path must be provided when no remote encoder is configured."
            )
        self.encoder = SentenceTransformer(
            self.config.encoder_path,
            device=self.config.encoder_device,
        )
        try:
            self.encoder = self.encoder.to(self.config.encoder_device)
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    def batch_query(
        self,
        node_ids: List[int],
        queries: List[str],
        max_results: Optional[int] = None,
    ) -> Dict[str, object]:
        if len(node_ids) != len(queries):
            raise ValueError("node_ids and queries must have the same length")

        results_batch: List[List[Dict[str, object]]] = []
        strategies: List[str] = []
        score_names: List[str] = []
        target_nodes: List[Dict[str, object]] = []

        for node_id, query in zip(node_ids, queries):
            strategy, score_name, target, docs = self._search_single(
                node_id,
                query,
                max_results,
            )
            strategies.append(strategy)
            score_names.append(score_name)
            target_nodes.append(target)
            results_batch.append(docs)

        return {
            "success": True,
            "strategies": strategies,
            "score_names": score_names,
            "target_nodes": target_nodes,
            "results_batch": results_batch,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_graph(self) -> None:
        data_dir = self.config.data_dir
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Graph data directory not found: {data_dir}")

        def _load_json_file(name: str) -> List[object]:
            path = os.path.join(data_dir, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing graph artifact: {path}")
            with open(path, "r", encoding="utf-8") as fp:
                return json.load(fp)

        # Load node texts and coerce to summary-only strings
        raw_node_texts = _load_json_file("node_texts.json")
        node_texts: List[str] = []
        if raw_node_texts and isinstance(raw_node_texts[0], dict):
            for item in raw_node_texts:
                if not isinstance(item, dict):
                    node_texts.append(str(item))
                    continue
                s = item.get("summary")
                node_texts.append(str(s))
        else:
            node_texts = [str(x) for x in raw_node_texts]
        self.node_texts: List[str] = node_texts

        raw_labels = _load_json_file("category.json")
        if raw_labels and isinstance(raw_labels[0], list):
            raw_labels = [lab[0] for lab in raw_labels]
        self.node_labels: List[str] = [str(x) for x in raw_labels]

        self.first_hop = _load_json_file("first_hop_indices.json")
        self.second_hop = _load_json_file("second_hop_indices.json")

        pagerank_path = os.path.join(data_dir, "pagerank.npy")
        if os.path.exists(pagerank_path):
            self.pagerank = np.load(pagerank_path)
        else:
            self.pagerank = None

        self.num_nodes = len(self.node_texts)
        if not (
            len(self.node_labels) == self.num_nodes
            and len(self.first_hop) == self.num_nodes
            and len(self.second_hop) == self.num_nodes
        ):
            raise ValueError("Graph artifacts have inconsistent lengths")

        splits_path = os.path.join(data_dir, "splits.json")
        train_ids: Set[int]
        if os.path.exists(splits_path):
            try:
                with open(splits_path, "r", encoding="utf-8") as fp:
                    splits = json.load(fp)
            except (json.JSONDecodeError, OSError) as exc:
                raise ValueError(f"Failed to parse splits.json under {data_dir}") from exc
            train_ids = set()
            for raw_idx in splits.get("train", []):
                try:
                    idx = int(raw_idx)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < self.num_nodes:
                    train_ids.add(idx)
        else:
            train_ids = set(range(self.num_nodes))

        if not train_ids:
            train_ids = set(range(self.num_nodes))
        self._train_id_set: Set[int] = train_ids
        self._train_id_list: List[int] = sorted(train_ids)

    def _load_embeddings(self) -> None:
        data_dir = self.config.data_dir
        emb_path = os.path.join(data_dir, "node_emb.npy")
        if os.path.exists(emb_path):
            emb = np.load(emb_path)
            if emb.shape[0] != self.num_nodes:
                raise ValueError(
                    f"node_emb.npy has shape {emb.shape} but expected ({self.num_nodes}, d)"
                )
            self.node_emb = _l2_normalize(emb.astype(np.float32), axis=1)
            self.embed_dim = int(self.node_emb.shape[1])
            self._init_local_encoder()
            return

        self._init_local_encoder()

        batches = []
        for i in range(0, self.num_nodes, self.config.encode_batch_size):
            batch = self.node_texts[i : i + self.config.encode_batch_size]
            batches.append(self._encode_batch(batch))
        emb = np.concatenate(batches, axis=0).astype(np.float32)
        if not os.path.exists(emb_path):
            try:
                np.save(emb_path, emb)
            except Exception:
                pass
        self.node_emb = _l2_normalize(emb, axis=1)
        self.embed_dim = int(self.node_emb.shape[1])

    # ------------------------------------------------------------------
    # Search strategies
    # ------------------------------------------------------------------
    def _target_node_info(self, node_id: int) -> Dict[str, object]:
        return {
            "node_id": node_id,
            "text_preview": _preview_text(
                self.node_texts[node_id], self.config.target_preview_len
            ),
            "label": self.node_labels[node_id],
        }

    def _search_single(
        self,
        node_id: int,
        raw_query: str,
        max_results: Optional[int],
    ) -> Tuple[str, str, Dict[str, object], List[Dict[str, object]]]:
        if node_id < 0 or node_id >= self.num_nodes:
            raise ValueError(f"node_id {node_id} out of range (0-{self.num_nodes - 1})")

        strategy, query_text = _detect_search_type(raw_query)
        score_name = "similarity"
        limit = self._resolve_strategy_topk(strategy, max_results)

        if strategy == "one_hop":
            fused = self._fusion_embedding(node_id, query_text)
            candidates = _safe_candidate_set(
                self.first_hop[node_id],
                self.num_nodes,
                node_id,
            )
            pairs = _topk_by_similarity(fused, candidates, self.node_emb, limit)
        elif strategy == "two_hop":
            fused = self._fusion_embedding(node_id, query_text)
            candidates = _safe_candidate_set(
                self.second_hop[node_id],
                self.num_nodes,
                node_id,
            )
            pairs = _topk_by_similarity(fused, candidates, self.node_emb, limit)
        elif strategy == "pagerank":
            score_name = "pagerank"
            pairs = self._pagerank_topk(limit, exclude=node_id)
        else:  # similar/global
            strategy = "similar"
            base_vec = self.node_emb[node_id]
            candidates = _safe_candidate_set(
                range(self.num_nodes),
                self.num_nodes,
                node_id,
            )
            pairs = _topk_by_similarity(base_vec, candidates, self.node_emb, min(limit + 1, len(candidates)))
            pairs = [p for p in pairs if p[0] != node_id][:limit]

        docs = [
            {
                "doc_id": rank,
                "text": _preview_text(
                    self.node_texts[cand], self.config.preview_len
                ),
            }
            for rank, (cand, _score) in enumerate(pairs, start=1)
        ]

        return strategy, score_name, self._target_node_info(node_id), docs

    def _resolve_strategy_topk(self, strategy: str, override: Optional[int]) -> int:
        if override is not None:
            try:
                return max(int(override), 0)
            except (TypeError, ValueError):
                return max(self.config.default_max_results, 0)
        mapping = {
            "similar": self.config.topk_similar,
            "one_hop": self.config.topk_one_hop,
            "two_hop": self.config.topk_two_hop,
            "pagerank": self.config.topk_pagerank,
        }
        val = mapping.get(strategy)
        if val is not None:
            try:
                return max(int(val), 0)
            except (TypeError, ValueError):
                pass
        return max(self.config.default_max_results, 0)

    def _fusion_embedding(self, node_id: int, query_text: str) -> np.ndarray:
        node_vec = self.node_emb[node_id]
        if not query_text:
            return node_vec
        query_vec = self._encode_batch([query_text])[0].astype(np.float32)
        query_vec = _l2_normalize(query_vec.reshape(1, -1), axis=1)[0]
        fused = self.config.fusion_alpha * query_vec + (1.0 - self.config.fusion_alpha) * node_vec
        return fused / (np.linalg.norm(fused) + 1e-12)

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            dim = getattr(self, "embed_dim", 0)
            return np.zeros((0, dim), dtype=np.float32)

        if self.remote_encoder_url:
            if self._http_session is None:
                self._http_session = requests.Session()
            try:
                response = self._http_session.post(
                    self.remote_encoder_url,
                    json={"texts": texts},
                    timeout=self.encoder_timeout,
                )
            except requests.RequestException as exc:
                raise RuntimeError("Remote encoder request failed") from exc
            response.raise_for_status()
            payload = response.json()
            embeddings = payload.get("embeddings")
            if embeddings is None:
                raise ValueError("Remote encoder response missing 'embeddings'")
            arr = np.asarray(embeddings, dtype=np.float32)
        else:
            if self.encoder is None:
                raise RuntimeError("Local encoder is not initialised")
            arr = self.encoder.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            ).astype(np.float32)

        if arr.ndim != 2:
            raise ValueError("Encoder output must be 2D")
        return arr

    def _pagerank_topk(self, k: int, exclude: Optional[int] = None) -> List[Tuple[int, float]]:
        if k <= 0:
            return []
        if self.pagerank is not None:
            scores = np.array(self.pagerank, dtype=np.float32)
        else:
            # fall back to degree centrality
            scores = np.array([len(neigh) for neigh in self.first_hop], dtype=np.float32)
        if exclude is not None and 0 <= exclude < self.num_nodes:
            scores = scores.copy()
            scores[exclude] = -np.inf
        order = np.argsort(-scores)[:k]
        return [(int(idx), float(scores[idx])) for idx in order.tolist()]


class MultiGraphRetriever:
    """Wrapper that routes queries across multiple graph datasets.

    It maintains a list of GraphRetriever partitions, each loaded from a
    different `data_dir`. Global node IDs are mapped to a (partition, local_id)
    pair using fixed offsets determined by the dataset sizes and the order of
    `data_dir`s passed in.
    """

    def __init__(self, data_dirs: List[str], base_config: GraphRetrieverConfig) -> None:
        if not data_dirs:
            raise ValueError("MultiGraphRetriever requires at least one data_dir")

        self.partitions: List[GraphRetriever] = []
        self.offsets: List[int] = []
        self.total_nodes = 0

        # Load each partition with its own retriever instance
        for d in [p.strip() for p in data_dirs if p and p.strip()]:
            cfg = GraphRetrieverConfig(
                data_dir=d,
                encoder_path=base_config.encoder_path,
                encode_batch_size=base_config.encode_batch_size,
                fusion_alpha=base_config.fusion_alpha,
                default_max_results=base_config.default_max_results,
                topk_similar=base_config.topk_similar,
                topk_one_hop=base_config.topk_one_hop,
                topk_two_hop=base_config.topk_two_hop,
                topk_pagerank=base_config.topk_pagerank,
                preview_len=base_config.preview_len,
                target_preview_len=base_config.target_preview_len,
                encoder_device=base_config.encoder_device,
                encoder_remote_url=base_config.encoder_remote_url,
                encoder_timeout=base_config.encoder_timeout,
            )
            retr = GraphRetriever(cfg)
            self.partitions.append(retr)
            self.offsets.append(self.total_nodes)
            self.total_nodes += retr.num_nodes

        if not self.partitions:
            raise RuntimeError("Failed to load any graph partitions")

    @property
    def num_nodes(self) -> int:
        return self.total_nodes

    def _locate(self, global_id: int) -> Tuple[int, int]:
        if global_id < 0 or global_id >= self.total_nodes:
            raise ValueError(f"node_id {global_id} out of range (0-{self.total_nodes - 1})")
        # Linear scan over a small number of partitions is fast and robust.
        for idx, offset in enumerate(self.offsets):
            part = self.partitions[idx]
            if global_id < offset + part.num_nodes:
                return idx, global_id - offset
        # Should never happen due to range check
        raise RuntimeError("Failed to locate partition for node id")

    def batch_query(
        self,
        node_ids: List[int],
        queries: List[str],
        max_results: Optional[int] = None,
    ) -> Dict[str, object]:
        if len(node_ids) != len(queries):
            raise ValueError("node_ids and queries must have the same length")

        override = max_results

        # Group by partition to leverage underlying batch implementation
        groups: Dict[int, List[int]] = {}
        for i, gid in enumerate(node_ids):
            pidx, _ = self._locate(int(gid))
            groups.setdefault(pidx, []).append(i)

        strategies = [""] * len(node_ids)
        score_names = [""] * len(node_ids)
        target_nodes = [None] * len(node_ids)
        results_batch: List[Optional[List[Dict[str, object]]]] = [None] * len(node_ids)

        for pidx, indices in groups.items():
            part = self.partitions[pidx]
            offset = self.offsets[pidx]
            local_node_ids = []
            part_queries = []
            for i in indices:
                gid = int(node_ids[i])
                _, lid = self._locate(gid)
                local_node_ids.append(lid)
                part_queries.append(queries[i])

            out = part.batch_query(local_node_ids, part_queries, max_results=override)
            # Map results back to global order and adjust target node ids
            for j, i in enumerate(indices):
                strategies[i] = out["strategies"][j]
                score_names[i] = out["score_names"][j]
                tn = out["target_nodes"][j]
                # adjust node_id to global
                tn = dict(tn)
                tn["node_id"] = offset + int(tn.get("node_id", 0))
                target_nodes[i] = tn
                results_batch[i] = out["results_batch"][j]

        return {
            "success": True,
            "strategies": strategies,
            "score_names": score_names,
            "target_nodes": target_nodes,  # type: ignore[list-item]
            "results_batch": results_batch,  # type: ignore[list-item]
        }


__all__ = ["GraphRetriever", "GraphRetrieverConfig", "MultiGraphRetriever"]
