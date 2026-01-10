# AgentGL

AgentGL is a graph-retrieval-augmented RLHF training and evaluation project built on OpenRLHF.
It supports multi-hop graph search during generation, reward-model scoring, and vLLM-based evaluation.

## Layout
- `OpenRLHF-RAG/`: OpenRLHF core with graph-native search extensions
- `train/`: dataset builders and utility scripts
- `evaluation/`: evaluation drivers and metrics
- `scripts/`: training and orchestration scripts

## Quick Start (high level)
1. Prepare graph artifacts and model checkpoints (paths are placeholders in scripts).
2. Run training scripts under `scripts/`.
3. Use `evaluation/` tools to generate and score results.

## Notes
- This repo is sanitized for sharing; data and logs are removed.
- Absolute paths are replaced with placeholders such as `/PATH/TO/WORKSPACE`.

## License
See `OpenRLHF-RAG/LICENSE`.
