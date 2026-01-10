#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple accuracy evaluator for JSONL files.

Each line is a JSON object containing at least:
  - "pred_ans": string
  - "answer":   string (or list of strings; list is tolerated)

Rules:
  - Correct if pred_ans == answer  (exact string match)
  - If answer is a list, correct if pred_ans equals ANY element in the list
  - Accuracy printed with two decimals.

Extras:
  - --head N : only evaluate the first N valid lines
  - Robust to UTF-8 BOM using encoding="utf-8-sig"
  - Skips lines missing required fields (with a warning)
"""

import argparse
import json

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate accuracy from a JSONL file.")
    p.add_argument("--input", default="/PATH/TO/WORKSPACE/AgentGL/data/eval_results/ogbn-products_qwen_lp_stage1_3b_grpo.jsonl", help="Path to JSONL file.")
    p.add_argument("--head", type=int, default=-1,
                   help="Only evaluate the first N lines (<=0 means no limit).")
    return p.parse_args()

def is_correct(pred, gold):
    """Exact match. If gold is a list, correct if pred equals any element."""
    if isinstance(gold, list):
        return any(pred == g for g in gold)
    return pred == gold

def main():
    args = parse_args()
    total = 0
    correct = 0
    skipped = 0

    with open(args.input, "r", encoding="utf-8-sig") as f:
        for ln, raw in enumerate(f, start=1):
            if args.head > 0 and total >= args.head:
                break
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                skipped += 1
                print(f"[WARN] Line {ln}: bad JSON ({e}); skipped.")
                continue

            if "pred_ans" not in obj or "answer" not in obj:
                skipped += 1
                print(f"[WARN] Line {ln}: missing 'pred_ans' or 'answer'; skipped.")
                continue

            pred = obj["pred_ans"]
            gold = obj["answer"]

            # Count valid samples
            total += 1
            if is_correct(pred, gold):
                correct += 1

    if total == 0:
        print("No valid samples to evaluate.")
        return

    acc = correct / total * 100.0
    print(f"File: {args.input}")
    print(f"Total: {total} | Correct: {correct} | Accuracy: {acc:.2f}%")
    # If you only need the numeric value, uncomment the next line:
    # print(f"{acc:.2f}")

if __name__ == "__main__":
    main()
