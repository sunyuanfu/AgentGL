#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly sample N examples from eval results, compare Stage1 vs Stage2 accuracy, optionally modify eval file."
    )
    parser.add_argument("--stage1", required=True, help="Path to Stage1 eval JSONL file")
    parser.add_argument("--stage2", required=True, help="Path to Stage2 eval JSONL file")
    parser.add_argument("--eval-file", required=True, help="Path to eval split JSONL file (contains node_id)")
    parser.add_argument("--modify-eval", action="store_true",
                        help="If set, overwrite eval file keeping only sampled node_ids.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of samples to draw (default: 1000)")
    parser.add_argument("--output", default="sampled_1000.jsonl", help="Output JSONL file for sampled examples")
    return parser.parse_args()

def is_correct(pred, gold):
    if isinstance(gold, list):
        return any(pred == g for g in gold)
    return pred == gold

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {ln} in {path}: bad JSON ({e})")
    return data

def main():
    args = parse_args()
    random.seed(args.seed)

    # Load data.
    s1 = load_jsonl(args.stage1)
    s2 = load_jsonl(args.stage2)
    eval_data = load_jsonl(args.eval_file)

    # Basic consistency checks.
    assert len(s1) == len(s2) == len(eval_data), \
        "Stage1, Stage2, and Eval file must have the same number of entries!"
    for i in range(len(s1)):
        assert s1[i].get("node_id") == s2[i].get("node_id") == eval_data[i].get("node_id"), \
            f"node_id mismatch at line {i+1}"

    # Random sampling.
    indices = random.sample(range(len(s1)), min(args.sample_size, len(s1)))
    sampled_node_ids = [s1[i]["node_id"] for i in indices]

    # Compute accuracy.
    records = []
    for i in indices:
        a, b = s1[i], s2[i]
        s1_correct = is_correct(a["pred_ans"], a["answer"])
        s2_correct = is_correct(b["pred_ans"], b["answer"])
        records.append({
            "node_id": a["node_id"],
            "stage1_correct": s1_correct,
            "stage2_correct": s2_correct,
            "stage1_pred": a["pred_ans"],
            "stage2_pred": b["pred_ans"],
            "answer": a["answer"]
        })

    n = len(records)
    s1_acc = sum(r["stage1_correct"] for r in records) / n * 100
    s2_acc = sum(r["stage2_correct"] for r in records) / n * 100
    improved = sum((r["stage2_correct"] and not r["stage1_correct"]) for r in records)
    degraded = sum((r["stage1_correct"] and not r["stage2_correct"]) for r in records)
    same = n - improved - degraded

    print(f"Randomly selected {n} samples (seed={args.seed})")
    print(f"Stage1 Accuracy: {s1_acc:.2f}%")
    print(f"Stage2 Accuracy: {s2_acc:.2f}%")
    print(f"Improved: {improved}, Degraded: {degraded}, Same: {same}")

    # Save sampled records.
    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved sampled examples to {args.output}")

    # Optionally rewrite the eval file.
    if args.modify_eval:
        print(f"Modifying eval file: {args.eval_file}")
        tmp_path = args.eval_file + ".bak"
        os.rename(args.eval_file, tmp_path)

        filtered = [x for x in eval_data if x["node_id"] in sampled_node_ids]
        with open(args.eval_file, "w", encoding="utf-8") as f:
            for e in filtered:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        print(f"Eval file overwritten with {len(filtered)} entries (backup saved to {tmp_path})")
    else:
        print("Eval file not modified (use --modify-eval to enable).")

if __name__ == "__main__":
    main()
