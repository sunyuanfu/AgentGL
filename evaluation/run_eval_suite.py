import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run eval_gs.py across multiple datasets sequentially")
    parser.add_argument("--datasets", type=str, required=True, help="Comma-separated dataset names")
    parser.add_argument("--eval_dir", type=str, required=True, help="Directory containing {dataset}.jsonl eval files")
    parser.add_argument("--graph_data_root", type=str, required=True, help="Root directory with graph data subfolders")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store eval results JSONL files")
    parser.add_argument("--output_suffix", type=str, default="_stage2.jsonl", help="Suffix appended to each output filename")
    parser.add_argument("--model", type=str, required=True, help="Model path forwarded to eval_gs.py")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path forwarded to eval_gs.py (defaults to model)")
    parser.add_argument("--graph_encoder_path", type=str, required=True, help="Graph encoder path forwarded to eval_gs.py")
    parser.add_argument("--head", type=int, default=1000, help="Number of samples per dataset to evaluate")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--stop-sequences", type=str, default="<|end_of_query|>,</answer>,<|im_end|>")
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--graph-max-searches", type=int, default=5)
    parser.add_argument("--graph-topk", type=int, default=5)
    parser.add_argument("--graph-topk-similar", type=int, default=None)
    parser.add_argument("--graph-topk-one-hop", type=int, default=None)
    parser.add_argument("--graph-topk-two-hop", type=int, default=None)
    parser.add_argument("--graph-topk-pagerank", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--run-metrics", action="store_true")
    parser.add_argument("--graph-task", type=str, choices=["node", "link"], default="node")
    parser.add_argument("--lp-node-data-root", type=str, default=None)
    parser.add_argument("--extra-args", type=str, default=None, help="Additional arguments passed verbatim to eval_gs.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    datasets: List[str] = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if not datasets:
        raise ValueError("No datasets specified")

    eval_dir = Path(args.eval_dir)
    graph_root = Path(args.graph_data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        sys.executable,
        str(Path(__file__).resolve().with_name("eval_gs.py")),
        "--model",
        args.model,
        "--graph-encoder-path",
        args.graph_encoder_path,
        "--batch-size",
        str(args.batch_size),
        "--temperature",
        str(args.temperature),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--stop-sequences",
        args.stop_sequences,
        "--max-rounds",
        str(args.max_rounds),
        "--graph-max-searches",
        str(args.graph_max_searches),
        "--graph-topk",
        str(args.graph_topk),
        "--graph-task",
        args.graph_task,
    ]

    if args.graph_topk_similar is not None:
        base_cmd.extend(["--graph-topk-similar", str(args.graph_topk_similar)])
    if args.graph_topk_one_hop is not None:
        base_cmd.extend(["--graph-topk-one-hop", str(args.graph_topk_one_hop)])
    if args.graph_topk_two_hop is not None:
        base_cmd.extend(["--graph-topk-two-hop", str(args.graph_topk_two_hop)])
    if args.graph_topk_pagerank is not None:
        base_cmd.extend(["--graph-topk-pagerank", str(args.graph_topk_pagerank)])

    base_cmd.extend([
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--head",
        str(args.head),
    ])

    if args.tokenizer:
        base_cmd.extend(["--tokenizer", args.tokenizer])
    if args.lp_node_data_root:
        base_cmd.extend(["--lp-node-data-root", args.lp_node_data_root])
    if args.run_metrics:
        base_cmd.append("--run-metrics")
    if args.extra_args:
        base_cmd.extend(args.extra_args.split())

    for dataset in datasets:
        eval_file = eval_dir / f"{dataset}.jsonl"
        if not eval_file.exists():
            raise FileNotFoundError(f"Eval file not found: {eval_file}")
        graph_dir = graph_root / dataset
        if not graph_dir.exists():
            raise FileNotFoundError(f"Graph directory not found: {graph_dir}")
        output_file = output_dir / f"{dataset}{args.output_suffix}"

        cmd = base_cmd + [
            "--eval-file",
            str(eval_file),
            "--output-file",
            str(output_file),
            "--graph-data-dir",
            str(graph_dir),
        ]

        print(f"\n[run_eval_suite] Evaluating {dataset} -> {output_file}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"eval_gs.py failed for dataset {dataset} with exit code {result.returncode}")


if __name__ == "__main__":
    main()
