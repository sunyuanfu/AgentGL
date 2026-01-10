import argparse
import json
import os
import re
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
QUERY_PATTERN = re.compile(r"<\|begin_of_query\|>.*?<\|end_of_query\|>", re.DOTALL)
DOC_PATTERN = re.compile(r"<\|begin_of_documents\|>.*?<\|end_of_documents\|>", re.DOTALL)

SEARCH_TAGS = {
    "1-hop": "<|begin_of_query|>1-hop",
    "2-hop": "<|begin_of_query|>2-hop",
    "pagerank": "<|begin_of_query|>pagerank",
    "similar": "<|begin_of_query|>similar",
}

MAX_RESPONSE_TOKENS = 2200


def _normalize_label(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


class GraphRewardServer:
    def __init__(self, args: argparse.Namespace) -> None:
        data_path = args.data_path
        if not data_path:
            raise ValueError("--data_path must point to the graph dataset directory or comma-separated directories")

        # Support comma-separated multiple datasets; concatenate labels by order
        labels_all: List[str] = []
        for part in [p.strip() for p in str(data_path).split(",") if p.strip()]:
            category_path = os.path.join(part, "category.json")
            if not os.path.exists(category_path):
                raise FileNotFoundError(f"category.json not found under {part}")
            with open(category_path, "r", encoding="utf-8") as fp:
                labels = json.load(fp)
            if labels and isinstance(labels[0], list):
                labels = [item[0] for item in labels]
            labels_all.extend([str(label) for label in labels])

        self.labels: List[str] = labels_all

        self.log_file = args.log_file
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        logger.info("Loaded %d labels for graph reward server", len(self.labels))

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    def _extract_answer(self, response: str) -> Optional[str]:
        match = ANSWER_PATTERN.search(response)
        if not match:
            return None
        return match.group(1).strip()

    def _format_reward(self, response: str) -> float:
        reward = 0.0

        think_ok = response.count("<think>") == 1 and response.count("</think>") == 1
        answer_ok = response.count("<answer>") == response.count("</answer>") == 1
        if think_ok and answer_ok:
            reward += 0.5
        else:
            reward -= 0.5

        doc_begin = response.count("<|begin_of_documents|>")
        doc_end = response.count("<|end_of_documents|>")
        query_begin = response.count("<|begin_of_query|>")
        query_end = response.count("<|end_of_query|>")

        if doc_begin == doc_end and query_begin == query_end:
            reward += 0.1
        else:
            reward -= 0.3

        answer_block = self._extract_answer(response) or ""

        if "<|begin_of_query|>" in answer_block or "<|begin_of_documents|>" in answer_block:
            reward -= 0.5

        tokens = answer_block.split()
        if len(tokens) > 12:
            reward -= 0.2

        if any("<think>" in part for part in answer_block.split("</think>")):
            reward -= 0.3

        return reward

    def _think_length_reward(self, response: str) -> float:
        match = THINK_PATTERN.search(response)
        if not match:
            return -0.3

        think_block = match.group(1)
        cleaned = DOC_PATTERN.sub(" ", think_block)
        segments = [segment.strip() for segment in QUERY_PATTERN.split(cleaned)]
        if not segments:
            return -0.3

        threshold = 30
        short_segments = 0
        for segment in segments:
            token_count = len(re.findall(r"\S+", segment))
            if token_count < threshold:
                short_segments += 1

        if short_segments == 0:
            return 0.5

        return max(-0.2 * short_segments, -0.6)

    # def _search_coverage_reward(self, response: str) -> float:
    #     matches = QUERY_PATTERN.findall(response)
    #     search_types: List[str] = []
    #     for match in matches:
    #         for name, tag in SEARCH_TAGS.items():
    #             if match.startswith(tag):
    #                 search_types.append(name)
    #                 break
    #     if not search_types:
    #         return 0.0

    #     limited = search_types[:5]
    #     reward = 0.3 * len(limited)
    #     if len(limited) >= len(SEARCH_TAGS) and len(set(limited)) == len(SEARCH_TAGS):
    #         reward += 1.0
    #     return reward
    
    def _search_coverage_reward(self, response: str) -> float:
        coverage = 0.0
        used_tags = set()
        for name, tag in SEARCH_TAGS.items():
            pattern = re.compile(re.escape(tag) + r".*?<\|end_of_query\|>", re.DOTALL)
            if pattern.search(response or ""):
                used_tags.add(name)
        if used_tags:
            coverage += 0.5 * len(used_tags)
        return min(coverage, 2.0)

    def _length_penalty(self, response: str) -> float:
        token_count = len(re.findall(r"\S+", response or ""))
        return -0.5 if token_count > MAX_RESPONSE_TOKENS else 0.0

    def _classification_reward(self, predicted: Optional[str], idx: int) -> float:
        if predicted is None:
            return -1.0
        if idx < 0 or idx >= len(self.labels):
            return -0.5
        gold = self.labels[idx]
        return 1.5 if _normalize_label(predicted) == _normalize_label(gold) else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def score_batch(self, responses: List[str], indices: List[int]) -> List[float]:
        if len(responses) != len(indices):
            raise ValueError("Length of responses and idx must match")

        rewards: List[float] = []
        for response, idx in zip(responses, indices):
            answer = self._extract_answer(response)
            reward = self._classification_reward(answer, int(idx))
            reward += self._format_reward(response)
            reward += self._think_length_reward(response)
            reward += self._search_coverage_reward(response)
            reward += self._length_penalty(response)
            rewards.append(float(max(min(reward, 15.0), -15.0)))

        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as fp:
                for response, idx, reward in zip(responses, indices, rewards):
                    record = {
                        "idx": int(idx),
                        "reward": reward,
                        "answer": self._extract_answer(response),
                        "raw": response,
                    }
                    fp.write(json.dumps(record, ensure_ascii=False) + "\n")

        return rewards


def create_app(args: argparse.Namespace) -> FastAPI:
    scorer = GraphRewardServer(args)
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        responses = data.get("query", []) or []
        idx_list = data.get("idx", []) or []
        if not isinstance(responses, list) or not isinstance(idx_list, list):
            raise ValueError("query and idx must be lists")
        rewards = scorer.score_batch(responses, [int(i) for i in idx_list])
        logger.info("Processed %d responses: avg_reward=%.3f", len(rewards), sum(rewards) / max(len(rewards), 1))
        return JSONResponse({"rewards": rewards})

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to graph dataset directory (comma-separated for multiple)")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="Unused but kept for compatibility")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--log_file", type=str, default=None, help="Optional JSONL log file")
    args = parser.parse_args()

    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
