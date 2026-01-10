
# NODE_RANK=$1
# export TORCH_HOME=/PATH/TO/WORKDIR
export NUMEXPR_MAX_THREADS=128
export RAY_DEDUP_LOGS=0

# Your wandb token
wandb_token=YOUR_WANDB_TOKEN
# sudo rm -rf ~/.netrc
export WANDB_API_KEY=${wandb_token}

# Dataset roots and encoder
NODE_DATA_ROOT=/PATH/TO/WORKSPACE/node_data
GRAPH_ENCODER_PATH=/PATH/TO/WORKSPACE/all-roberta-large-v1

# Mixed datasets (can be adjusted)
DATASETS="ogbn-arxiv,ogbn-products"

# Build per-dataset JSONL prompts using splits.json constraints (idempotent)
OUT_DIR=/PATH/TO/WORKSPACE/AgentGL/data/training_set_rb
# python3 /PATH/TO/WORKSPACE/AgentGL/train/generate_datasets_from_splits.py \
#   --datasets ${DATASETS} \
#   --node-data-root ${NODE_DATA_ROOT} \
#   --mode stage1 \
#   --output-dir ${OUT_DIR}

# Resolve generated JSONLs
ARXIV_JSONL=${OUT_DIR}/ogbn-arxiv_stage1_rebalanced.jsonl
PRODUCTS_JSONL=${OUT_DIR}/ogbn-products_stage1_rebalanced.jsonl

# Mixed prompt data and probs
DATA_PATH="${ARXIV_JSONL},${PRODUCTS_JSONL}"
DATA_PROBS="0.5,0.5"

# Multi-graph data dirs (comma-separated, same order as DATASETS)
GRAPH_DATA_DIR="${NODE_DATA_ROOT}/ogbn-arxiv,${NODE_DATA_ROOT}/ogbn-products"

# Path of backbone model (Qwen)
TOKENIZER_PATH=/PATH/TO/WORKSPACE/Qwen2.5-3B-Instruct

# Remote graph encoder service (run separately, e.g. on GPU6/7)
GRAPH_ENCODER_REMOTE_URL=http://127.0.0.1:9100/encode

GRAPH_TOPK_DEFAULT=5
GRAPH_TOPK_SIMILAR=5
GRAPH_TOPK_ONE_HOP=5
GRAPH_TOPK_TWO_HOP=5
GRAPH_TOPK_PAGERANK=5

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
N_SAMPLES=16
EPISODE=1
WARMUP=0.0
TBS=128
RBS=32
KL=0.0
LR=2e-6
MAX_LENGTH=1600
PORT=1278
TEMP=1.0

SAVE_MODEL_NAME=qwen_graph_rl_stage1_3b_grpo

GROUP_METHOD=normal

LOG_BASE=log

mkdir -p /PATH/TO/WORKDIR/rag_rl/results/$SAVE_MODEL_NAME
mkdir -p /PATH/TO/WORKDIR/rag_rl/results/ckpts
mkdir -p /PATH/TO/WORKDIR/rag_rl/results/$SAVE_MODEL_NAME/server
mkdir -p $LOG_BASE/server/

ray job submit --address="http://127.0.0.1:8267" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 2 \
   --colocate_actor_ref \
   --pretrain ${TOKENIZER_PATH} \
   --remote_rm_url http://localhost:${PORT}/get_reward \
   --save_path /PATH/TO/WORKDIR/rag_rl/results/ckpts/$SAVE_MODEL_NAME \
   --ckpt_path /PATH/TO/WORKDIR/rag_rl/results/ckpts/$SAVE_MODEL_NAME \
   --micro_train_batch_size 1 \
   --train_batch_size ${TBS} \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size ${RBS} \
   --advantage_estimator group_norm \
   --max_samples 8000 \
   --max_epochs 1 \
   --num_episodes ${EPISODE} \
   --lr_warmup_ratio ${WARMUP} \
   --n_samples_per_prompt $N_SAMPLES \
   --prompt_max_len 1024 \
   --generate_max_len $MAX_LENGTH \
   --zero_stage 2 \
   --bf16 \
   --load_checkpoint \
   --actor_learning_rate $LR \
   --critic_learning_rate 9e-6 \
   --init_kl_coef $KL \
   --prompt_data $DATA_PATH \
   --prompt_data_probs $DATA_PROBS \
   --balanced_prompt_mixing \
   --input_key summary_en \
   --graph_data_dir ${GRAPH_DATA_DIR} \
   --graph_encoder_path ${GRAPH_ENCODER_PATH} \
   --graph_encoder_remote_url ${GRAPH_ENCODER_REMOTE_URL} \
   --graph_max_searches 5 \
   --graph_topk ${GRAPH_TOPK_DEFAULT} \
   --graph_topk_similar ${GRAPH_TOPK_SIMILAR} \
   --graph_topk_one_hop ${GRAPH_TOPK_ONE_HOP} \
   --graph_topk_two_hop ${GRAPH_TOPK_TWO_HOP} \
   --graph_topk_pagerank ${GRAPH_TOPK_PAGERANK} \
   --flash_attn \
   --use_kl_loss \
   --group_method $GROUP_METHOD \
   --use_length_reward_in_efficiency \
   --gradient_checkpointing \
   --save_steps 20 \
   --vllm_sync_backend nccl \
   --max_ckpt_num 3 \
   --temperature $TEMP \
   --overlap_comm \
   --packing_samples \
   --use_wandb ${wandb_token} \
   --wandb_run_name $SAVE_MODEL_NAME \
