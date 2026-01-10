from __future__ import annotations

from typing import Dict, List, Optional, Set

from torch.utils.data import Dataset
from tqdm import tqdm

from openrlhf.utils.link_prediction_utils import (
    ensure_lp_sources,
    get_node_summary,
    global_pair_id,
)


def _parse_difficulties(spec: Optional[str]) -> Optional[Set[str]]:
    if not spec:
        return None
    allowed = {item.strip().lower() for item in spec.split(",") if item.strip()}
    return allowed or None


def _pool_limits(args) -> Dict[str, Optional[int]]:
    return {
        "default": getattr(args, "graph_topk", 5),
        "similar": getattr(args, "graph_topk_similar", None),
        "one_hop": getattr(args, "graph_topk_one_hop", None),
        "two_hop": getattr(args, "graph_topk_two_hop", None),
        "pagerank": getattr(args, "graph_topk_pagerank", None),
    }


def _dataset_relation_desc(dataset: str) -> str:
    ds = dataset.lower()
    if "arxiv" in ds or "pubmed" in ds:
        return (
            "Nodes are research papers. An edge represents a citation linkage between the two papers."
        )
    if "amazon" in ds or "products" in ds:
        return (
            "Nodes are Amazon products. An edge indicates strong co-purchase relationships between the items."
        )
    if "reddit" in ds:
        return (
            "Nodes are Reddit posts. An edge indicates strong co-post relationships between the posts."
        )
    return "Nodes come from the same graph dataset; an edge captures the canonical relation defined for that dataset."


def _fmt(dataset: str, max_search_limit: int, pool_topk: Dict[str, Optional[int]]) -> str:
    def _resolve(name: str) -> int:
        base = pool_topk.get("default") or 5
        val = pool_topk.get(name)
        return int(val if val is not None else base)

    limits = (
        f"per-pool limits → 1-hop {_resolve('one_hop')}, 2-hop {_resolve('two_hop')}, "
        f"pagerank {_resolve('pagerank')}, similar {_resolve('similar')}"
    )

    return (
        "<|im_start|>system\n"
        "You are a reasoning assistant with access to graph searches.\n"
        f"Determine whether two nodes from the **{dataset}** dataset should be connected.\n"
        f"{_dataset_relation_desc(dataset)}\n"
        "Treat the task as binary classification and return 'yes' if the edge should exist and 'no' otherwise.\n"
        "\n"
        "THINK/ANSWER FORMAT:\n"
        "- Perform all reasoning and search planning inside <think>...</think>.\n"
        "- Output ONLY the final judgment as <answer>yes</answer> or <answer>no</answer>.\n"
        "- Never leak your chain-of-thought outside of <think>...</think>.\n"
        "\n"
        "GRAPH SEARCH POOLS:\n"
        "- 1-hop: prioritize common direct neighbours—nodes that are directly connected to BOTH endpoints (U and V). If common neighbours are insufficient, fill the remaining slots with non-common 1-hop neighbours from U and/or V (balanced when possible). \n"
        "- 2-hop: prioritize common 2-hop neighbours—nodes that can reach BOTH endpoints within two hops (neighbors-of-neighbors to U and V). If common 2-hop neighbours are insufficient, fill the remaining slots with non-common 2-hop neighbours from U and/or V (balanced when possible). \n"
        "- pagerank: list globally influential reference edges selected offline using PageRank, highlighting edges that are structurally important in the overall graph as complementary examples. \n"
        "- similar: retrieve the Top-K node pairs that are most similar to the current pair in the graph, and include their connectivity/edge status as reference (i.e., whether those similar pairs are connected). \n"
        "\n"
        f"SEARCH FORMAT ({limits}):\n"
        "- Every search must happen inside <think>...</think>.\n"
        "- To launch a search, emit exactly one of the following tags with an optional free-form hint:\n"
        "  * <|begin_of_query|>1-hop:Query<|end_of_query|>\n"
        "  * <|begin_of_query|>2-hop:Query<|end_of_query|>\n"
        "  * <|begin_of_query|>pagerank:Query<|end_of_query|>\n"
        "  * <|begin_of_query|>similar:Query<|end_of_query|>\n"
        "- Retrieved blocks arrive as <|begin_of_documents|> ... <|end_of_documents|> and already explain whether a row is a common neighbour or belongs to node U or V.\n"
        "- Use at most one search per round and no more than "
        f"{max_search_limit} total searches before answering.\n"
        "\n"
        "Guidelines:\n"
        "- Start by analysing the given node descriptions.\n"
        "- Prefer covering multiple pools (shared neighbours, unique structure, global priors).\n"
        "- Explain your reasoning inside <think>...</think> but keep the final judgement concise.\n"
        "-"
        "- Final answer MUST be <answer>yes</answer> or <answer>no</answer>.\n"
        "<|im_end|>\n"
    )


def format_link_prediction_prompt(
    record: dict,
    summary_u: str,
    summary_v: str,
    max_search_limit: int,
    pool_topk: Dict[str, Optional[int]],
) -> str:
    dataset = record.get("dataset", "graph")
    header = _fmt(dataset, max_search_limit, pool_topk)
    node_u = int(record.get("node_u", -1))
    node_v = int(record.get("node_v", -1))
    label_hint = (
        "- Label 1 corresponds to <answer>yes</answer> (an edge exists under the dataset's relationship definition).\n"
        "- Label 0 corresponds to <answer>no</answer> (nodes should remain disconnected).\n"
    )
    user_block = (
        "<|im_start|>user\n"
        "We are investigating whether the following two nodes should be linked:\n"
        f"- Node U (id={node_u}): {summary_u or 'No summary available'}\n"
        f"- Node V (id={node_v}): {summary_v or 'No summary available'}\n"
        "\n"
        f"Use up to {max_search_limit} searches to gather evidence from the graph retriever. "
        "Pay attention to shared neighbours, structural motifs, and graph priors before making a decision.\n"
        f"{label_hint}"
        "Remember to answer strictly with yes/no enclosed by <answer> tags.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return header + user_block


class LinkPredictionPromptDataset(Dataset):
    def __init__(self, dataset, tokenizer, strategy, input_template=None) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        args = strategy.args
        self.sources = ensure_lp_sources(args)
        self.node_data_root = getattr(args, "lp_node_data_root", None)
        self.allowed_difficulties = _parse_difficulties(getattr(args, "lp_allowed_difficulties", None))

        max_search_limit = getattr(args, "graph_max_searches", 5)
        pool_topk = _pool_limits(args)

        difficulty_buckets: Dict[int, List[str]] = {0: [], 1: [], 2: []}
        difficulty_rank = {"easy": 0, "medium": 1, "hard": 2}

        for record in tqdm(dataset, desc="Preprocessing LP data", disable=not strategy.is_rank_0()):
            diff = str(record.get("difficulty", "medium")).lower()
            if self.allowed_difficulties and diff not in self.allowed_difficulties:
                continue
            sample_id = global_pair_id(record, self.sources)
            dataset_name = str(record.get("dataset", "graph"))
            summary_u = get_node_summary(dataset_name, int(record.get("node_u", -1)), self.node_data_root)
            summary_v = get_node_summary(dataset_name, int(record.get("node_v", -1)), self.node_data_root)
            prompt = format_link_prediction_prompt(record, summary_u, summary_v, max_search_limit, pool_topk)
            bucket = difficulty_rank.get(diff, 1)
            difficulty_buckets[bucket].append(f"{sample_id}<|idx_prompt_split|>{prompt}")

        self.prompts: List[str] = []
        for bucket in (0, 1, 2):
            self.prompts.extend(difficulty_buckets[bucket])

        if not self.prompts:
            raise ValueError(
                "No link-prediction prompts were created. "
                "Check --lp_allowed_difficulties and the supplied JSONL files."
            )

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> str:
        return self.prompts[idx]
