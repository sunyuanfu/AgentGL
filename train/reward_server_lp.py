import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)
MAX_RESPONSE_TOKENS = 2000

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
QUERY_PATTERN = re.compile(r"<\|begin_of_query\|>.*?<\|end_of_query\|>", re.DOTALL)
DOC_PATTERN = re.compile(r"<\|begin_of_documents\|>.*?<\|end_of_documents\|>", re.DOTALL)

SEARCH_TAGS = {
    "1-hop": ("<|begin_of_query|>1-hop", "<|end_of_query|>"),
    "2-hop": ("<|begin_of_query|>2-hop", "<|end_of_query|>"),
    "pagerank": ("<|begin_of_query|>pagerank", "<|end_of_query|>"),
    "similar": ("<|begin_of_query|>similar", "<|end_of_query|>"),
}


def load_pair_labels(pair_data: str) -> Dict[int, int]:
    if not pair_data:
        raise ValueError("--pair_data must point to link prediction JSONL files")
    labels: Dict[int, int] = {}
    base = 0
    for raw_path in pair_data.split(","):
        path = Path(raw_path.strip())
        if not path.is_file():
            raise FileNotFoundError(f"pair_data file not found: {path}")
        count = 0
        with path.open("r", encoding="utf-8") as fp:
            for local_idx, line in enumerate(fp):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                pair_id = int(record.get("pair_id", local_idx))
                global_id = base + pair_id
                labels[global_id] = int(record.get("label", 0))
                count += 1
        base += count
    logger.info("Loaded %d link-prediction labels from %s", len(labels), pair_data)
    return labels


def extract_answer(response: str) -> Optional[str]:
    match = ANSWER_PATTERN.search(response or "")
    if not match:
        return None
    text = match.group(1).strip()
    return text or None


def normalize_answer(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    cleaned = re.sub(r"[^a-z]", "", text.lower())
    if cleaned in {"yes", "edge", "true"}:
        return "yes"
    if cleaned in {"no", "false"}:
        return "no"
    return None


class LinkPredictionRewardStage1:
    def __init__(self, args: argparse.Namespace) -> None:
        self.labels = load_pair_labels(args.pair_data)
        self.log_file = args.log_file
        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

    def _classification_reward(self, predicted: str, idx: int) -> float:
        if predicted is None or idx not in self.labels:
            return -1.0
        gold = self.labels[idx]
        pred = 1 if predicted == "yes" else 0
        return 1.5 if pred == gold else 0.0

    def _format_reward(self, response: str) -> float:
        reward = 0.0
        think_ok = response.count("<think>") == 1 and response.count("</think>") == 1
        answer_ok = response.count("<answer>") == response.count("</answer>") == 1
        reward += 0.5 if think_ok and answer_ok else -0.5
        doc_begin = response.count("<|begin_of_documents|>")
        doc_end = response.count("<|end_of_documents|>")
        query_begin = response.count("<|begin_of_query|>")
        query_end = response.count("<|end_of_query|>")
        reward += 0.1 if doc_begin == doc_end and query_begin == query_end else -0.3
        answer_block = extract_answer(response) or ""
        if "<|begin_of_query|>" in answer_block or "<|begin_of_documents|>" in answer_block:
            reward -= 0.5
        if len(answer_block.split()) > 6:
            reward -= 0.2
        if any("<think>" in part for part in answer_block.split("</think>")):
            reward -= 0.3
        return reward

    def _think_length_reward(self, response: str) -> float:
        match = THINK_PATTERN.search(response or "")
        if not match:
            return -0.3
        think_block = match.group(1)
        cleaned = DOC_PATTERN.sub(" ", think_block)
        segments = [segment.strip() for segment in QUERY_PATTERN.split(cleaned)]
        if not segments:
            return -0.3
        threshold = 30
        short_segments = sum(1 for segment in segments if len(segment.split()) < threshold)
        if short_segments == 0:
            return 0.5
        return 0

    def _search_coverage_reward(self, response: str) -> float:
        coverage = 0.0
        if not response:
            return coverage
        used_tags = set()
        for name, (begin_tag, end_tag) in SEARCH_TAGS.items():
            if begin_tag in response and end_tag in response:
                used_tags.add(name)
        if used_tags:
            coverage = 0.5 * len(used_tags)
        return min(coverage, 2.0)

    def _length_penalty(self, response: str) -> float:
        token_count = len(response.split())
        return -0.5 if token_count > MAX_RESPONSE_TOKENS else 0.0

    def score_batch(self, responses: List[str], indices: List[int]) -> List[float]:
        rewards: List[float] = []
        for response, raw_idx in zip(responses, indices):
            idx = int(raw_idx)
            answer = normalize_answer(extract_answer(response))
            reward = self._classification_reward(answer, idx)
            reward += self._format_reward(response)
            reward += self._think_length_reward(response)
            reward += self._search_coverage_reward(response)
            reward += self._length_penalty(response)
            rewards.append(float(max(min(reward, 15.0), -15.0)))

        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as fp:
                for response, idx, reward in zip(responses, indices, rewards):
                    fp.write(
                        json.dumps(
                            {
                                "idx": int(idx),
                                "reward": reward,
                                "answer": normalize_answer(extract_answer(response)),
                                "raw": response,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        return rewards


def create_app(args: argparse.Namespace) -> FastAPI:
    scorer = LinkPredictionRewardStage1(args)
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        responses = data.get("query", []) or []
        idx_list = data.get("idx", []) or []
        if not isinstance(responses, list) or not isinstance(idx_list, list):
            raise ValueError("query and idx must be lists")
        rewards = scorer.score_batch(responses, [int(i) for i in idx_list])
        logger.info("Stage1 LP rewards for %d responses: avg=%.3f", len(rewards), sum(rewards) / max(len(rewards), 1))
        return JSONResponse({"rewards": rewards})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage1 reward server for link prediction RLHF training")
    parser.add_argument(
        "--pair_data",
        type=str,
        required=True,
        help="Comma-separated link prediction JSONL files used for the current training stage",
    )
    parser.add_argument("--port", type=int, default=5003)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--log_file", type=str, default=None, help="Optional JSONL log file")
    args = parser.parse_args()

    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
