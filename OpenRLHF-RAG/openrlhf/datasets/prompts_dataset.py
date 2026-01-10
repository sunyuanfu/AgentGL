import json
import os
import re
from typing import Dict, List, Optional

from torch.utils.data import Dataset
from tqdm import tqdm


_CATEGORY_CACHE: Dict[str, List[str]] = {}


def _read_categories_from_txt(category_txt_path: str) -> Dict[str, List[str]]:
    """Parse a simple python-like mapping file to a dict of categories.

    Expected rough structure per dataset:
        'dataset-name': [ 'A', 'B', ... ],
    """
    if not os.path.exists(category_txt_path):
        return {}
    text = open(category_txt_path, "r", encoding="utf-8").read()
    mapping: Dict[str, List[str]] = {}
    # Find blocks like 'name': [ ... ]
    for m in re.finditer(r"'([^']+?)'\s*:\s*\[(.*?)\]", text, flags=re.S):
        name = m.group(1)
        body = m.group(2)
        cats = [c.strip().strip("'\"") for c in body.split(",") if c.strip()]
        mapping[name] = cats
    return mapping


def _unique_categories_from_dir(dir_path: str) -> List[str]:
    path = os.path.join(dir_path, "category.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fp:
        arr = json.load(fp)
    if arr and isinstance(arr[0], list):
        arr = [x[0] for x in arr]
    # preserve first-seen order
    seen = {}
    for x in arr:
        seen.setdefault(str(x), True)
    return list(seen.keys())


def _get_dataset_categories(dataset: str, graph_data_dir: Optional[str]) -> List[str]:
    # 1) Try cache
    if dataset in _CATEGORY_CACHE:
        return _CATEGORY_CACHE[dataset]

    cats: List[str] = []

    # 2) Try resolve from graph_data_dir (can be comma-separated paths)
    if graph_data_dir:
        parts = [p.strip() for p in str(graph_data_dir).split(",") if p.strip()]
        # If a part ends with the dataset name, use it; otherwise try join
        for p in parts:
            if p.rstrip("/").endswith("/" + dataset) or p.rstrip("/").endswith(dataset):
                cats = _unique_categories_from_dir(p)
                break
        if not cats and len(parts) == 1:
            candidate = os.path.join(parts[0], dataset)
            cats = _unique_categories_from_dir(candidate)

    # 3) Fallback to the shared category.txt at default node_data path
    if not cats:
        default_root = "/PATH/TO/WORKSPACE/node_data"
        mapping = _read_categories_from_txt(os.path.join(default_root, "category.txt"))
        cats = mapping.get(dataset, [])

    _CATEGORY_CACHE[dataset] = cats
    return cats


def _format_graph_prompt(
    node_id: int,
    summary: str,
    max_search_limit: int,
    dataset: Optional[str] = None,
    graph_data_dir: Optional[str] = None,
    pool_topk: Optional[Dict[str, Optional[int]]] = None,
) -> str:
    summary = (summary or "").strip()
    summary_snippet = summary[:500] + ("..." if len(summary) > 500 else "")
    domain = (dataset or "ogbn-arxiv").lower()

    default_topk = 5
    if pool_topk is not None:
        default_topk = int(pool_topk.get("default") or default_topk)
    def _resolve_limit(name: str) -> int:
        if pool_topk is None:
            return default_topk
        val = pool_topk.get(name)
        return default_topk if val is None else int(val)

    topk_one_hop = _resolve_limit("one_hop")
    topk_two_hop = _resolve_limit("two_hop")
    topk_pagerank = _resolve_limit("pagerank")
    topk_similar = _resolve_limit("similar")

    is_arxiv = domain in {"ogbn-arxiv", "arxiv_2023", "arxiv-2023", "arxiv"}
    is_pubmed = domain == "pubmed"
    is_amazon = (domain.startswith("amazon") or domain == "ogbn-products" or "products" in domain)
    is_reddit = domain == "reddit"

    if is_arxiv:
        task_line = (
            "Your task is to predict the category of arXiv computer science (cs) papers.\n"
        )
        output_req = "- Final output must be exactly: <answer>cs.XX</answer>\n"
        pools = (
            f"- 1-hop: direct neighbors (papers that directly cite or are cited by the anchor). Highest locality. Returns up to {topk_one_hop} nodes per search.\n"
            f"- 2-hop: neighbors of neighbors expanding the local region. Captures indirect yet related context with more diversity than 1-hop. Returns up to {topk_two_hop} nodes per search.\n"
            f"- pagerank: globally influential nodes selected by PageRank (e.g., surveys/benchmarks/frameworks). Returns up to {topk_pagerank} nodes per search.\n"
            f"- similar: globally most semantically similar nodes by embedding similarity. Returns up to {topk_similar} nodes per search.\n"
        )
        anchor_hdr = "\n Now please predict the category of the anchor node paper:\n"
    elif is_pubmed:
        task_line = (
            "Your task is to predict the category of PubMed biomedical papers.\n"
        )
        output_req = (
            "- Final output must be exactly one of the listed PubMed categories inside <answer>...</answer>.\n"
        )
        pools = (
            f"- 1-hop: direct neighbors (papers that directly cite or are cited by the anchor). Highest locality. Returns up to {topk_one_hop} nodes per search.\n"
            f"- 2-hop: neighbors of neighbors expanding the local region. Captures indirect yet related context with more diversity than 1-hop. Returns up to {topk_two_hop} nodes per search.\n"
            f"- pagerank: globally influential nodes selected by PageRank (e.g., surveys/benchmarks/frameworks). Returns up to {topk_pagerank} nodes per search.\n"
            f"- similar: globally most semantically similar nodes by embedding similarity. Returns up to {topk_similar} nodes per search.\n"
        )
        cats = _get_dataset_categories(domain, graph_data_dir)
        cats_block = "\nAvailable PubMed categories:\n- " + "\n- ".join(cats) + "\n" if cats else "\n"
        anchor_hdr = (
            "\n Now please predict the category of the anchor node paper:\n" + cats_block
        )
    elif is_amazon:
        task_line = (
            "Your task is to predict the category of Amazon products.\n"
        )
        output_req = (
            "- Final output must be exactly one of the listed Amazon categories inside <answer>...</answer>.\n"
        )
        pools = (
            f"- 1-hop: direct neighbors (products that are frequently co-purchased together with the anchor). Highest locality. Returns up to {topk_one_hop} nodes per search.\n"
            f"- 2-hop: neighbors of neighbors expanding the local region in the co-purchasing graph. Returns up to {topk_two_hop} nodes per search.\n"
            f"- pagerank: globally influential products selected by PageRank. Returns up to {topk_pagerank} nodes per search.\n"
            f"- similar: globally most semantically similar products by embedding similarity. Returns up to {topk_similar} nodes per search.\n"
        )
        cats = _get_dataset_categories(domain, graph_data_dir)
        cats_block = "\nAvailable Amazon categories:\n- " + "\n- ".join(cats) + "\n" if cats else "\n"
        anchor_hdr = (
            "\n Now please predict the category of the anchor node product:\n" + cats_block
        )
    elif is_reddit:
        task_line = (
            "Your task is to predict the subreddit category of a Reddit post.\n"
        )
        output_req = (
            "- Final output must be exactly one of the listed subreddit categories inside <answer>...</answer>.\n"
        )
        pools = (
            f"- 1-hop: other posts from the same author (also_posted set). Highest locality. Returns up to {topk_one_hop} nodes per search.\n"
            f"- 2-hop: same as 1-hop in this dataset (duplicates allowed for robustness). Returns up to {topk_two_hop} nodes per search.\n"
            f"- pagerank: globally influential posts selected by PageRank over the author cliques. Returns up to {topk_pagerank} nodes per search.\n"
            f"- similar: globally most semantically similar posts by embedding similarity. Returns up to {topk_similar} nodes per search.\n"
        )
        cats = _get_dataset_categories(domain, graph_data_dir)
        cats_block = "\nAvailable subreddit categories:\n- " + "\n- ".join(cats) + "\n" if cats else "\n"
        anchor_hdr = (
            "\n Now please predict the category of the anchor post:\n" + cats_block
        )
    else:
        # Fallback: treat as arxiv-style
        task_line = (
            "Your task is to predict the category of arXiv computer science (cs) papers.\n"
        )
        output_req = "- Final output must be exactly: <answer>cs.XX</answer>\n"
        pools = (
            f"- 1-hop: direct neighbors (papers that directly cite or are cited by the anchor). Highest locality. Returns up to {topk_one_hop} nodes per search.\n"
            f"- 2-hop: neighbors of neighbors expanding the local region. Captures indirect yet related context with more diversity than 1-hop. Returns up to {topk_two_hop} nodes per search.\n"
            f"- pagerank: globally influential nodes selected by PageRank (e.g., surveys/benchmarks/frameworks). Returns up to {topk_pagerank} nodes per search.\n"
            f"- similar: globally most semantically similar nodes by embedding similarity. Returns up to {topk_similar} nodes per search.\n"
        )
        anchor_hdr = "\n Now please predict the category of the anchor node paper:\n"

    search_limits_desc = (
        f"per-pool limits → 1-hop {topk_one_hop}, 2-hop {topk_two_hop}, pagerank {topk_pagerank}, similar {topk_similar}"
    )

    base_prompt = (
        "<|im_start|>system\n"
        "You are a reasoning assistant with the ability to perform graph searches to help you answer the user's question accurately.\n"
        f"{task_line}"
        "You can use graph search to get the information of neighbor nodes to help you answer the user's question accurately.\n"
        "\n"
        "THINK/ANSWER FORMAT:\n"
        "- Do ALL internal reasoning inside <think>...</think>.\n"
        "- Provide ONLY the final category label inside <answer>...</answer>.\n"
        "- Never reveal your chain-of-thought outside <think>...\n"
        "- When confident, output exactly one label as required.\n"
        "\n"
        "GRAPH SEARCH POOLS (for context retrieval):\n"
        f"{pools}"
        "\n"
        f"SEARCH FORMAT ({search_limits_desc}):\n"
        "- All searches MUST be performed INSIDE the <think>...</think> block.\n"
        "- Use one of the pool-specific tags below to indicate which neighbor pool to search:\n"
        "  * 1-hop:    <|begin_of_query|>1-hop:QUERY<|end_of_query|>\n"
        "  * 2-hop:    <|begin_of_query|>2-hop:QUERY<|end_of_query|>\n"
        "  * pagerank: <|begin_of_query|>pagerank:QUERY<|end_of_query|>\n"
        "  * similar:  <|begin_of_query|>similar:QUERY<|end_of_query|>\n"
        "- The search system will return results wrapped as:\n"
        "  <|begin_of_documents|> ...retrieval results... <|end_of_documents|>\n"
        "\n"
        "Guidelines:\n"
        "- Try to cover several pools across rounds for stronger reasoning.\n"
        "- You can only perform one search in each round, but you may run multiple searches overall.\n"
        f"- Max total searches = {max_search_limit}.\n"
        "- Do not output anything other than <think>...</think> and a single <answer>...</answer>.\n"
        "\n"
        "OUTPUT REQUIREMENT:\n"
        f"{output_req}"
        "\n"
        # "QUERY EXAMPLE:\n"
        # "<|begin_of_query|>1-hop:I need find some papers about the topic of GNNs<|end_of_query|>\n"
        # "<|begin_of_query|>2-hop:Give me some papers on how to use Transformers<|end_of_query|>\n"
        # "<|begin_of_query|>pagerank:I need to check which papers are the most influential in the field of Computer Vision<|end_of_query|>\n"
        # "<|begin_of_query|>similar:I need to find some most relevant papers.<|end_of_query|>\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{anchor_hdr}"
        "Anchor Node Information:\n"
        f"{summary_snippet}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    return base_prompt


def preprocess_data(
    data,
    input_template=None,
    input_key: str = "input",
    apply_chat_template=None,
    max_search_limit: int = 5,
) -> str:
    node_id = data.get("node_id")
    # accept both keys, prefer summary_en if provided
    summary = data.get("summary_en") or data.get("summary")
    dataset = data.get("dataset")
    graph_data_dir = getattr(getattr(preprocess_data, "_args", object()), "graph_data_dir", None)
    args_ref = getattr(preprocess_data, "_args", object())
    pool_topk = {
        "default": getattr(args_ref, "graph_topk", 5),
        "similar": getattr(args_ref, "graph_topk_similar", None),
        "one_hop": getattr(args_ref, "graph_topk_one_hop", None),
        "two_hop": getattr(args_ref, "graph_topk_two_hop", None),
        "pagerank": getattr(args_ref, "graph_topk_pagerank", None),
    }

    if node_id is not None and summary is not None:
        prompt = _format_graph_prompt(
            int(node_id),
            str(summary),
            max_search_limit,
            dataset=dataset,
            graph_data_dir=graph_data_dir,
            pool_topk=pool_topk,
        )
        return f"{node_id}<|idx_prompt_split|>{prompt}"

    # Fallback to the original prompt style for legacy datasets.
    if apply_chat_template:
        question = data["question"]
        idx = data["idx"]

        messages_chat = [
            {
                "role": "system",
                "content": """You are a helpful assistant.
Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, **you can perform searching for uncertain knowledge** if necessary with the format of "<|begin_of_query|> search query (only list keywords, such as **"keyword_1 keyword_2 ..."**)<|end_of_query|>". **A query must involve only a single triple**.
Then, the search system will provide you with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".""",
            },
            {"role": "user", "content": question},
        ]

        prompt = apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True) + "<think>"
    else:
        base_prompt = """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, **the Assistant can perform searching** for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only list keywords, such as **"keyword_1 keyword_2 ..."**)<|end_of_query|>". **A query must involve only a single triple**.
Then, the search system will provide the Assistant with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".

User:{question}
Assistant: <think>"""

        question = data["question"]
        idx = data["idx"]
        prompt = base_prompt.format(question=question)

    return str(idx) + "<|idx_prompt_split|>" + prompt


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        max_search_limit = getattr(self.strategy.args, "graph_max_searches", 5)

        difficulty_buckets = {0: [], 1: [], 2: []}
        difficulty_rank = {"easy": 0, "medium": 1, "hard": 2}

        # Pass args into preprocess_data via function attribute for category resolution
        preprocess_data._args = self.strategy.args  # type: ignore[attr-defined]

        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(
                data,
                input_template,
                input_key,
                apply_chat_template,
                max_search_limit=max_search_limit,
            )
            difficulty = str(data.get("difficulty", "medium")).lower()
            rank = difficulty_rank.get(difficulty, 1)
            difficulty_buckets[rank].append(prompt)

        # assemble prompts from easy -> medium -> hard for curriculum-style training
        for rank in (0, 1, 2):
            self.prompts.extend(difficulty_buckets[rank])
        # print("len(self.prompts):",len(self.prompts))
        # print("self.prompts[0:5]:",self.prompts[0:5])
        # kill

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
