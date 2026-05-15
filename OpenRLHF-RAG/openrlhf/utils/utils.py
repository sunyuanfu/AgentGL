import math
import os
import random
from fractions import Fraction
from typing import List, Tuple

from datasets import Dataset, interleave_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def get_strategy(args):
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def _lcm(values: List[int]) -> int:
    result = 1
    for value in values:
        if value == 0:
            continue
        result = abs(result * value) // math.gcd(result, value)
    return max(1, result)


def _build_round_robin_pattern(probabilities: List[float]) -> List[int]:
    fractions: List[Tuple[int, Fraction]] = []
    for idx, prob in enumerate(probabilities):
        if prob <= 0:
            continue
        fractions.append((idx, Fraction(prob).limit_denominator(1000)))

    if not fractions:
        return list(range(len(probabilities)))

    lcm_value = _lcm([frac.denominator for _, frac in fractions])
    pattern: List[int] = []
    for idx, frac in fractions:
        count = frac.numerator * (lcm_value // frac.denominator)
        count = max(1, count)
        pattern.extend([idx] * count)

    missing = [i for i in range(len(probabilities)) if all(i != idx for idx, _ in fractions)]
    pattern.extend(missing)

    return pattern or list(range(len(probabilities)))


def _balanced_interleave_datasets(dataset_list, probabilities, seed, stopping_strategy):
    rng = random.Random(seed)
    shuffled = [ds.shuffle(seed=rng.randint(0, 2**31 - 1)) for ds in dataset_list]
    lengths = [len(ds) for ds in shuffled]
    total = sum(lengths)
    if total == 0:
        return shuffled[0].select(range(0)) if shuffled else Dataset.from_list([])

    pattern = _build_round_robin_pattern(probabilities)
    cursors = [0] * len(shuffled)
    result = []
    pattern_idx = 0

    def any_exhausted():
        return any(cur >= length for cur, length in zip(cursors, lengths))

    def all_exhausted():
        return all(cur >= length for cur, length in zip(cursors, lengths))

    while len(result) < total:
        if stopping_strategy == "first_exhausted" and any_exhausted():
            break
        if stopping_strategy == "all_exhausted" and all_exhausted():
            break

        dataset_idx = pattern[pattern_idx % len(pattern)]
        pattern_idx += 1
        if dataset_idx >= len(shuffled):
            continue

        if cursors[dataset_idx] >= lengths[dataset_idx]:
            if stopping_strategy == "first_exhausted":
                break
            if all_exhausted():
                break
            continue

        result.append(shuffled[dataset_idx][cursors[dataset_idx]])
        cursors[dataset_idx] += 1

    if not result:
        return shuffled[0].select(range(0)) if shuffled else Dataset.from_list([])

    return Dataset.from_list(result)


def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
    train_split="train",
    eval_split="test",
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        if ext == ".py" or (
            os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
        ):
            data = load_dataset(dataset, trust_remote_code=True)
            strategy.print(f"loaded {dataset} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
            strategy.print(f"loaded {dataset} with data_files={dataset}")
        # local dataset saved with `datasets.Dataset.save_to_disk`
        elif os.path.isdir(dataset):
            data = load_from_disk(dataset)
            strategy.print(f"loaded {dataset} from disk")
        # remote/local folder or common file
        else:
            data = load_dataset(dataset, data_dir=data_dir)
            strategy.print(f"loaded {dataset} from files")

        if train_split and train_split in data:
            train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
        else:
            train_data = data.select(range(min(max_count, len(data))))
        train_data_list.append(train_data)

        if return_eval:
            if eval_split and eval_split in data:
                eval_data = data[eval_split].select(range(min(max_count, len(data[eval_split]))))
            # train will contains eval? TODO
            else:
                eval_data = train_data.select(range(min(max_count, int(len(train_data) * 0.03))))
            eval_data_list.append(eval_data)

    # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    balanced = bool(strategy and getattr(strategy.args, "balanced_prompt_mixing", False))

    if balanced:
        train_dataset = _balanced_interleave_datasets(
            train_data_list,
            probabilities,
            seed,
            stopping_strategy,
        )
    else:
        train_dataset = interleave_datasets(
            train_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
    if return_eval:
        if balanced:
            eval_dataset = _balanced_interleave_datasets(
                eval_data_list,
                probabilities,
                seed,
                stopping_strategy,
            )
        else:
            eval_dataset = interleave_datasets(
                eval_data_list,
                probabilities=probabilities,
                seed=seed,
                stopping_strategy=stopping_strategy,
            )
        return train_dataset, eval_dataset
    else:
        return train_dataset


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")
