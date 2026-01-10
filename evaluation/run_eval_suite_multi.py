import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


FLAG_MAP: Dict[str, str] = {
    "datasets": "--datasets",
    "eval_dir": "--eval_dir",
    "graph_data_root": "--graph_data_root",
    "output_dir": "--output_dir",
    "output_suffix": "--output_suffix",
    "model": "--model",
    "tokenizer": "--tokenizer",
    "graph_encoder_path": "--graph_encoder_path",
    "head": "--head",
    "batch_size": "--batch-size",
    "temperature": "--temperature",
    "max_new_tokens": "--max-new-tokens",
    "stop_sequences": "--stop-sequences",
    "max_rounds": "--max-rounds",
    "graph_max_searches": "--graph-max-searches",
    "graph_topk": "--graph-topk",
    "graph_topk_similar": "--graph-topk-similar",
    "graph_topk_one_hop": "--graph-topk-one-hop",
    "graph_topk_two_hop": "--graph-topk-two-hop",
    "graph_topk_pagerank": "--graph-topk-pagerank",
    "tensor_parallel_size": "--tensor-parallel-size",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "extra_args": "--extra-args",
}

BOOL_FLAGS: Dict[str, str] = {
    "run_metrics": "--run-metrics",
}

RUN_REQUIRED_KEYS = {"model", "graph_encoder_path", "output_dir"}
GLOBAL_REQUIRED_KEYS = {"datasets", "eval_dir", "graph_data_root"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run run_eval_suite.py multiple times with different parameter sets",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="JSON config path containing shared options and runs list",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated commands without executing them",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "runs" not in cfg or not isinstance(cfg["runs"], list) or not cfg["runs"]:
        raise ValueError("Config must contain a non-empty 'runs' list")
    return cfg


def build_cli_args(options: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for key, flag in FLAG_MAP.items():
        if key not in options:
            continue
        value = options[key]
        if value is None:
            continue
        args.extend([flag, str(value)])

    for key, flag in BOOL_FLAGS.items():
        if options.get(key):
            args.append(flag)
    return args


def merge_options(shared: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(shared)
    merged.update(run)
    return merged


def ensure_run_keys(options: Dict[str, Any]) -> None:
    missing = [
        key
        for key in GLOBAL_REQUIRED_KEYS | RUN_REQUIRED_KEYS
        if key not in options or options[key] is None
    ]
    if missing:
        raise ValueError(f"Missing required keys for run: {missing}")


def run_commands(commands: Iterable[List[str]], dry_run: bool) -> None:
    for cmd in commands:
        print(f"\n[run_eval_suite_multi] Command: {' '.join(cmd)}")
        if dry_run:
            continue
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {result.returncode}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)

    shared = cfg.get("shared", {})

    runs: List[Dict[str, Any]] = cfg["runs"]
    command_list: List[List[str]] = []

    run_eval_suite_path = Path(__file__).resolve().with_name("run_eval_suite.py")

    for idx, run in enumerate(runs, start=1):
        options = merge_options(shared, run)
        ensure_run_keys(options)

        cli_args = build_cli_args(options)
        cmd = [sys.executable, str(run_eval_suite_path)] + cli_args

        run_label = options.get("name") or f"run{idx}"
        print(f"\n[run_eval_suite_multi] Preparing {run_label}")
        command_list.append(cmd)

    run_commands(command_list, args.dry_run)


if __name__ == "__main__":
    main()
