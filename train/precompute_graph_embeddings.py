import argparse
import os
import time

from openrlhf.utils.graph_retriever import GraphRetriever, GraphRetrieverConfig


def build_embeddings(data_dir: str, config: GraphRetrieverConfig) -> None:
    start = time.time()
    print(f"[precompute] Loading graph from {data_dir} ...", flush=True)
    cfg = GraphRetrieverConfig(
        data_dir=data_dir,
        encoder_path=config.encoder_path,
        encode_batch_size=config.encode_batch_size,
        fusion_alpha=config.fusion_alpha,
        default_max_results=config.default_max_results,
        topk_similar=config.topk_similar,
        topk_one_hop=config.topk_one_hop,
        topk_two_hop=config.topk_two_hop,
        topk_pagerank=config.topk_pagerank,
        preview_len=config.preview_len,
        target_preview_len=config.target_preview_len,
        encoder_device=config.encoder_device,
        encoder_remote_url=config.encoder_remote_url,
        encoder_timeout=config.encoder_timeout,
    )
    GraphRetriever(cfg)
    elapsed = time.time() - start
    print(f"[precompute] Finished {data_dir} in {elapsed:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute graph node embeddings to warm up retriever cache")
    parser.add_argument(
        "--graph_data_dir",
        type=str,
        required=True,
        help="Single path or comma-separated list of graph dataset directories",
    )
    parser.add_argument(
        "--graph_encoder_path",
        type=str,
        default=None,
        help="SentenceTransformer model path if local encoding is needed",
    )
    parser.add_argument(
        "--graph_encoder_device",
        type=str,
        default="cpu",
        help="Device for local encoding (ignored when remote URL is used)",
    )
    parser.add_argument(
        "--graph_encoder_remote_url",
        type=str,
        default=None,
        help="Endpoint of remote encoding service (POST /encode)",
    )
    parser.add_argument(
        "--graph_encoder_remote_timeout",
        type=float,
        default=120.0,
        help="Timeout for remote encoding requests",
    )
    parser.add_argument(
        "--encode_batch_size",
        type=int,
        default=128,
        help="Batch size used during embedding generation",
    )
    args = parser.parse_args()

    parts = [p.strip() for p in args.graph_data_dir.split(",") if p.strip()]
    if not parts:
        raise ValueError("graph_data_dir must contain at least one directory")

    base_cfg = GraphRetrieverConfig(
        data_dir=parts[0],  # placeholder, will be replaced per dataset
        encoder_path=args.graph_encoder_path,
        encode_batch_size=args.encode_batch_size,
        fusion_alpha=0.5,
        default_max_results=5,
        preview_len=500,
        target_preview_len=120,
        encoder_device=args.graph_encoder_device,
        encoder_remote_url=args.graph_encoder_remote_url,
        encoder_timeout=args.graph_encoder_remote_timeout,
    )

    for data_dir in parts:
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Graph data directory not found: {data_dir}")
        base_cfg.data_dir = data_dir
        build_embeddings(data_dir, base_cfg)


if __name__ == "__main__":
    main()
