#!/usr/bin/env python
"""Graph-search evaluation script aligned with the two-stage training pipeline."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.distributed as dist
from vllm import LLM, SamplingParams

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXTRA_PATHS = (_REPO_ROOT, _REPO_ROOT / "OpenRLHF-RAG")
for extra_path in _EXTRA_PATHS:
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from openrlhf.datasets import format_link_prediction_prompt
from openrlhf.datasets.prompts_dataset import _format_graph_prompt
from openrlhf.utils.graph_retriever import GraphRetriever, GraphRetrieverConfig
from openrlhf.utils.link_prediction_retriever import LinkPredictionRetriever
from openrlhf.utils.link_prediction_utils import build_lp_sources, get_node_summary

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclass
class EvalSample:
    index: int
    node_id: int
    summary: str
    prompt: str
    answer: str
    metadata: Optional[dict] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate graph-search policy with vLLM")
    parser.add_argument("--model", default="/PATH/TO/WORKDIR/rag_rl/results/ckpts/qwen_graph_rl_stage1_whole",
                        help="Path to the model weights for vLLM")
    parser.add_argument("--tokenizer", help="Optional tokenizer path; defaults to model path")
    parser.add_argument(
        "--eval-file",
        default="/PATH/TO/WORKSPACE/AgentGL/data/eval_set/ogbn_arxiv.jsonl",
        help="JSONL file containing evaluation samples (node_id & summary_en)",
    )
    parser.add_argument(
        "--output-file",
        default="/PATH/TO/WORKSPACE/AgentGL/data/eval_results/ogbn_arxiv_stage1.jsonl",
        help="Destination JSONL file under data/eval_results",
    )
    parser.add_argument(
        "--graph-data-dir",
        default="/PATH/TO/WORKSPACE/AgentGL-main/node_data",
        help="Directory with graph artifacts (node_texts.json, category.json, etc.)",
    )
    parser.add_argument(
        "--graph-encoder-path",
        default="/PATH/TO/WORKSPACE/all-roberta-large-v1",
        help="SentenceTransformer encoder path when node_emb.npy is missing",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of prompts evaluated together per vLLM call",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--stop-sequences",
        type=str,
        default="<|end_of_query|>,</answer>,<|im_end|>",
        help="Comma separated stop strings passed to vLLM",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=12,
        help="Max rollouts per sample (includes search and final answer rounds)",
    )
    parser.add_argument(
        "--graph-max-searches",
        type=int,
        default=5,
        help="Upper bound on search turns per sample (matches training)",
    )
    parser.add_argument(
        "--graph-topk",
        type=int,
        default=5,
        help="Top-k neighbors returned by the graph retriever",
    )
    parser.add_argument(
        "--graph-topk-similar",
        type=int,
        default=None,
        help="Override top-k for the 'similar' pool during evaluation (defaults to --graph-topk)",
    )
    parser.add_argument(
        "--graph-topk-one-hop",
        type=int,
        default=None,
        help="Override top-k for the '1-hop' pool during evaluation (defaults to --graph-topk)",
    )
    parser.add_argument(
        "--graph-topk-two-hop",
        type=int,
        default=None,
        help="Override top-k for the '2-hop' pool during evaluation (defaults to --graph-topk)",
    )
    parser.add_argument(
        "--graph-topk-pagerank",
        type=int,
        default=None,
        help="Override top-k for the 'pagerank' pool during evaluation (defaults to --graph-topk)",
    )
    parser.add_argument(
        "--graph-task",
        type=str,
        choices=["node", "link"],
        default="node",
        help="Graph reasoning task. Use 'link' for link prediction datasets.",
    )
    parser.add_argument(
        "--lp-node-data-root",
        type=str,
        default=None,
        help="Node text root for link prediction evaluation.",
    )
    parser.add_argument(
        "--lp-neighbor-seed",
        type=int,
        default=20240101,
        help="Random seed for neighbour sampling in link prediction retrieval.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallelism degree for vLLM",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="GPU memory ratio hint for vLLM",
    )
    parser.add_argument(
        "--run-metrics",
        action="store_true",
        help="Run metric_calc_rule on the merged result file (rank 0 only)",
    )
    parser.add_argument(
        "--overwrite",
        default=True,
        help="Overwrite existing output file",
    )
    # New: only take the first N samples
    parser.add_argument(
        "--head",
        type=int,
        default=1000,
        help="Only evaluate the first N samples from the JSONL (<=0 means no limit).",
    )

    args = parser.parse_args()
    args.stop_sequences = [s.strip() for s in args.stop_sequences.split(",") if s.strip()]
    return args


def init_distributed() -> tuple[int, int, int]:
    """Bind device first, then init process group (if really distributed)."""
    env_world_size = int(os.environ.get("WORLD_SIZE", 1))
    have_dist = "RANK" in os.environ and env_world_size > 1

    if have_dist:
        world_size = env_world_size
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)          # bind device first
        dist.init_process_group(backend="nccl")        # then initialize PG
    else:
        rank = 0
        world_size = 1
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return rank, world_size, local_rank


def load_graph(
    graph_dir: str,
    encoder_path: Optional[str],
    topk_default: int,
    pool_topk: Optional[Dict[str, Optional[int]]] = None,
    graph_task: str = "node",
    lp_node_data_root: Optional[str] = None,
    lp_neighbor_seed: int = 20260101,
    graph_max_searches: int = 5,
):
    pool_topk = pool_topk or {}
    if graph_task == "link":
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            graph_data_dir=graph_dir,
            lp_node_data_root=lp_node_data_root,
            graph_topk=topk_default,
            graph_topk_similar=pool_topk.get("similar"),
            graph_topk_one_hop=pool_topk.get("one_hop"),
            graph_topk_two_hop=pool_topk.get("two_hop"),
            graph_topk_pagerank=pool_topk.get("pagerank"),
            graph_max_searches=graph_max_searches,
            lp_neighbor_seed=lp_neighbor_seed,
        )
        return LinkPredictionRetriever(cfg)

    config = GraphRetrieverConfig(
        data_dir=graph_dir,
        encoder_path=encoder_path,
        default_max_results=topk_default,
        topk_similar=pool_topk.get("similar"),
        topk_one_hop=pool_topk.get("one_hop"),
        topk_two_hop=pool_topk.get("two_hop"),
        topk_pagerank=pool_topk.get("pagerank"),
    )
    return GraphRetriever(config)


def _coerce_summary(record: dict) -> str:
    summary = (
        record.get("summary_en")
        or record.get("summary")
        or record.get("question")
        or record.get("text")
    )
    if summary is None:
        raise ValueError("Evaluation sample missing summary-like field")
    return str(summary).strip()


def load_samples(
    eval_path: str,
    graph_retriever,
    max_searches: int,
    graph_data_dir: Optional[str] = None,
    head: int = -1,
    pool_topk: Optional[Dict[str, Optional[int]]] = None,
    graph_task: str = "node",
    lp_node_data_root: Optional[str] = None,
) -> List[EvalSample]:
    samples: List[EvalSample] = []
    if not os.path.isfile(eval_path):
        raise FileNotFoundError(f"Eval file not found: {eval_path}")
    pool_topk = pool_topk or {}
    lp_sources = None
    if graph_task == "link":
        lp_sources = { (src.dataset, src.split): src for src in build_lp_sources(graph_data_dir) }

    with open(eval_path, "r", encoding="utf-8-sig") as f:
        parsed = 0
        for idx, raw in enumerate(f):
            if head is not None and head > 0 and parsed >= head:
                break
            line = raw.strip()
            if not line:
                continue
            record = json.loads(line)
            if graph_task == "link":
                dataset_name = str(record.get("dataset"))
                split_name = str(record.get("split"))
                if (dataset_name, split_name) not in lp_sources:
                    raise KeyError(f"No link-prediction source for dataset={dataset_name}, split={split_name}")
                source = lp_sources[(dataset_name, split_name)]
                pair_id = int(record.get("pair_id", parsed))
                node_id = source.base_offset + pair_id
                summary_u = get_node_summary(dataset_name, int(record.get("node_u", -1)), lp_node_data_root)
                summary_v = get_node_summary(dataset_name, int(record.get("node_v", -1)), lp_node_data_root)
                prompt = format_link_prediction_prompt(
                    record,
                    summary_u,
                    summary_v,
                    max_searches,
                    pool_topk,
                )
                answer = "yes" if int(record.get("label", 0)) == 1 else "no"
                summary = f"U: {summary_u or 'N/A'} | V: {summary_v or 'N/A'}"
                metadata = {
                    "dataset": dataset_name,
                    "split": split_name,
                    "node_u": int(record.get("node_u", -1)),
                    "node_v": int(record.get("node_v", -1)),
                }
            else:
                node_id = int(record["node_id"])
                summary = _coerce_summary(record)
                dataset_name = record.get("dataset")
                prompt = _format_graph_prompt(
                    node_id,
                    summary,
                    max_searches,
                    dataset_name,
                    graph_data_dir,
                    pool_topk,
                )
                answer = graph_retriever.node_labels[node_id]
                metadata = {"dataset": dataset_name}
            samples.append(EvalSample(parsed, node_id, summary, prompt, answer, metadata))
            parsed += 1
    return samples


def shard_samples(samples: Sequence[EvalSample], world_size: int, rank: int) -> List[EvalSample]:
    if world_size <= 1:
        return list(samples)
    per_rank = int(math.ceil(len(samples) / world_size))
    start = rank * per_rank
    end = min(len(samples), start + per_rank)
    return list(samples[start:end])


def chunks(items: Sequence[EvalSample], batch_size: int) -> Iterable[List[EvalSample]]:
    for i in range(0, len(items), batch_size):
        yield list(items[i : i + batch_size])


def extract_answer(text: str) -> str:
    match = ANSWER_PATTERN.search(text)
    if not match:
        return ""
    return match.group(1).strip()


def extract_query(text: str) -> Optional[str]:
    if "<|begin_of_query|>" not in text or "<|end_of_query|>" not in text:
        return None
    tail = text.rsplit("<|begin_of_query|>", 1)[-1]
    query = tail.split("<|end_of_query|>", 1)[0]
    query = query.replace('"', "").strip()
    query = " ".join(query.split())
    return query if query else None


def format_neighbors(neighbors: Sequence[dict]) -> str:
    if not neighbors:
        return "None\n"
    lines: List[str] = []
    for idx, doc in enumerate(neighbors, 1):
        text = doc.get("text") or doc.get("text_preview") or ""
        text = re.sub(r"^\s*\d+\s+", "", str(text)).strip()
        rank_val = doc.get("doc_id")
        header = rank_val if isinstance(rank_val, int) else idx
        lines.append(f"({header}) {text}")
    return "\n".join(lines) + "\n"


def graph_search(
    graph_retriever: GraphRetriever,
    node_id: int,
    query: str,
) -> str:
    result = graph_retriever.batch_query([node_id], [query])
    neighbors = []
    if result:
        neighbors = result.get("results_batch", [[]])[0] or []
    return format_neighbors(neighbors)


def evaluate_batch(
    llm: LLM,
    samples: List[EvalSample],
    sampling_params: SamplingParams,
    graph_retriever: GraphRetriever,
    max_rounds: int,
    max_searches: int,
) -> List[dict]:
    pending = [
        {
            "sample": sample,
            "prompt": sample.prompt,
            "transcript": "",
            "search_count": 0,
            "retrievals": [],
        }
        for sample in samples
    ]
    finished: List[dict] = []

    for round_idx in range(max_rounds):
        if not pending:
            break
        prompts = [item["prompt"] for item in pending]
        outputs = llm.generate(prompts, sampling_params)
        next_pending: List[dict] = []
        for item, output in zip(pending, outputs):
            generation = output.outputs[0]
            new_text = generation.text
            item["transcript"] += new_text
            item["prompt"] += new_text

            query = extract_query(new_text)
            if query and item["search_count"] < max_searches:
                doc_content = graph_search(
                    graph_retriever,
                    item["sample"].node_id,
                    query,
                )
                item["retrievals"].append({"query": query, "documents": doc_content})
                item["search_count"] += 1
                tagged_block = (
                    "\n\n<|begin_of_documents|>\n"
                    + doc_content
                    + "<|end_of_documents|>\n\n"
                )
                item["transcript"] += tagged_block
                item["prompt"] += tagged_block
                next_pending.append(item)
                continue

            if "</answer>" in item["transcript"]:
                finished.append(item)
            else:
                next_pending.append(item)
        pending = next_pending

    finished.extend(pending)

    results: List[dict] = []
    for item in finished:
        sample = item["sample"]
        generation = item["transcript"]
        pred_ans = extract_answer(generation)
        results.append(
            {
                "node_id": sample.node_id,
                "summary": sample.summary,
                "prompt": sample.prompt,
                "generation": generation,
                "pred_ans": pred_ans if pred_ans else "",
                "answer": sample.answer,
                "retrievals": item["retrievals"],
                "sample_index": sample.index,
                "metadata": sample.metadata or {},
            }
        )
    return results


def ensure_output_path(path: str, overwrite: bool) -> None:
    dst_dir = os.path.dirname(path)
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}")


def save_jsonl(path: str, records: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def merge_rank_outputs(base_path: str, world_size: int) -> List[dict]:
    merged: List[dict] = []
    for rank in range(world_size):
        shard_path = f"{base_path}.rank{rank}"
        if not os.path.exists(shard_path):
            continue
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                merged.append(json.loads(line))
        os.remove(shard_path)
    merged.sort(key=lambda x: x["sample_index"])
    for record in merged:
        record.pop("sample_index", None)
    return merged


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed()

    if rank == 0:
        print(f"Using world size {world_size}; tensor-parallel {args.tensor_parallel_size}")
        ensure_output_path(args.output_file, args.overwrite)
    if world_size > 1:
        dist.barrier(device_ids=[local_rank])  # explicit device for NCCL barrier

    pool_topk = {
        "default": args.graph_topk,
        "similar": args.graph_topk_similar,
        "one_hop": args.graph_topk_one_hop,
        "two_hop": args.graph_topk_two_hop,
        "pagerank": args.graph_topk_pagerank,
    }

    graph_retriever = load_graph(
        args.graph_data_dir,
        args.graph_encoder_path,
        args.graph_topk,
        pool_topk,
        graph_task=args.graph_task,
        lp_node_data_root=args.lp_node_data_root,
        lp_neighbor_seed=args.lp_neighbor_seed,
        graph_max_searches=args.graph_max_searches,
    )
    samples = load_samples(
        args.eval_file,
        graph_retriever,
        args.graph_max_searches,
        graph_data_dir=args.graph_data_dir,
        head=args.head,
        pool_topk=pool_topk,
        graph_task=args.graph_task,
        lp_node_data_root=args.lp_node_data_root,
    )
    local_samples = shard_samples(samples, world_size, rank)

    if not local_samples:
        shard_path = f"{args.output_file}.rank{rank}"
        save_jsonl(shard_path, [])
        if world_size > 1:
            dist.barrier(device_ids=[local_rank])
        if rank == 0:
            merged = merge_rank_outputs(args.output_file, world_size)
            save_jsonl(args.output_file, merged)
            if args.run_metrics:
                from evaluation.metric_calc_rule import eval as eval_metrics
                print(eval_metrics(args.output_file))
        return

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stop=args.stop_sequences,
        include_stop_str_in_output=True,
    )

    shard_results: List[dict] = []
    for batch in chunks(local_samples, args.batch_size):
        shard_results.extend(
            evaluate_batch(
                llm,
                batch,
                sampling_params,
                graph_retriever,
                args.max_rounds,
                args.graph_max_searches,
            )
        )

    shard_path = f"{args.output_file}.rank{rank}"
    save_jsonl(shard_path, shard_results)

    if world_size > 1:
        dist.barrier(device_ids=[local_rank])

    if rank == 0:
        merged = merge_rank_outputs(args.output_file, world_size)
        save_jsonl(args.output_file, merged)
        if args.run_metrics:
            from evaluation.metric_calc_rule import eval as eval_metrics
            print(eval_metrics(args.output_file))


if __name__ == "__main__":
    main()
