#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export AGENTGL_PROJECT_ROOT="${AGENTGL_PROJECT_ROOT:-${PROJECT_ROOT}}"

DEFAULT_ENV="${PROJECT_ROOT}/config/agentgl.env"
if [[ -f "${DEFAULT_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${DEFAULT_ENV}"
fi
if [[ -n "${AGENTGL_ENV_FILE:-}" && "${AGENTGL_ENV_FILE}" != "${DEFAULT_ENV}" && -f "${AGENTGL_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${AGENTGL_ENV_FILE}"
fi

export PYTHONPATH="${PROJECT_ROOT}/OpenRLHF-RAG:${PYTHONPATH:-}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-128}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export CUDA_VISIBLE_DEVICES="${AGENTGL_CUDA_VISIBLE_DEVICES}"

SAVE_MODEL_NAME="${AGENTGL_LP_GRPO_STAGE2_SAVE_NAME:-agentgl_lp_grpo_stage2}"
DATA_PATH="${AGENTGL_LP_GRPO_STAGE2_DATA_PATH:-${AGENTGL_LP_STAGE_DIR}/ogbn-arxiv_stage2.jsonl,${AGENTGL_LP_STAGE_DIR}/ogbn-products_stage2.jsonl}"
GRAPH_DATA_DIR="${AGENTGL_LP_GRAPH_DATA_DIR:-${DATA_PATH}}"
PRETRAIN_PATH="${AGENTGL_LP_GRPO_STAGE2_MODEL:-${AGENTGL_OUTPUT_ROOT}/ckpts/agentgl_lp_grpo_stage1}"

mkdir -p "${AGENTGL_OUTPUT_ROOT}/ckpts" "${AGENTGL_OUTPUT_ROOT}/${SAVE_MODEL_NAME}/server" "${AGENTGL_LOG_DIR}/server"

WANDB_ARGS=()
if [[ -n "${AGENTGL_WANDB_API_KEY:-${WANDB_API_KEY:-}}" ]]; then
  export WANDB_API_KEY="${AGENTGL_WANDB_API_KEY:-${WANDB_API_KEY}}"
  WANDB_ARGS=(--use_wandb "${WANDB_API_KEY}" --wandb_run_name "${SAVE_MODEL_NAME}")
fi

cmd=(
  ray job submit --address="${AGENTGL_RAY_ADDRESS}" --
  python3 -m openrlhf.cli.train_ppo_ray
  --graph_task link
  --graph_data_dir "${GRAPH_DATA_DIR}"
  --lp_node_data_root "${AGENTGL_NODE_DATA_ROOT}"
  --lp_allowed_difficulties "${AGENTGL_LP_STAGE2_DIFFICULTIES:-medium,hard}"
  --ref_num_nodes 1
  --ref_num_gpus_per_node 4
  --actor_num_nodes 1
  --actor_num_gpus_per_node 4
  --vllm_num_engines 1
  --vllm_tensor_parallel_size 2
  --colocate_actor_ref
  --pretrain "${PRETRAIN_PATH}"
  --remote_rm_url "http://localhost:${AGENTGL_REWARD_PORT}/get_reward"
  --save_path "${AGENTGL_OUTPUT_ROOT}/ckpts/${SAVE_MODEL_NAME}"
  --ckpt_path "${AGENTGL_OUTPUT_ROOT}/ckpts/${SAVE_MODEL_NAME}"
  --micro_train_batch_size 1
  --train_batch_size "${AGENTGL_TRAIN_BATCH_SIZE}"
  --micro_rollout_batch_size 1
  --rollout_batch_size "${AGENTGL_ROLLOUT_BATCH_SIZE}"
  --advantage_estimator group_norm
  --max_samples "${AGENTGL_LP_MAX_SAMPLES:-4000}"
  --max_epochs 1
  --num_episodes "${AGENTGL_EPISODES}"
  --lr_warmup_ratio "${AGENTGL_WARMUP}"
  --n_samples_per_prompt "${AGENTGL_N_SAMPLES}"
  --prompt_max_len 1024
  --generate_max_len "${AGENTGL_LP_STAGE2_MAX_LENGTH:-1600}"
  --zero_stage 2
  --bf16
  --load_checkpoint
  --actor_learning_rate "${AGENTGL_LR}"
  --critic_learning_rate 9e-6
  --init_kl_coef "${AGENTGL_KL}"
  --curriculum_easy_to_hard
  --prompt_data "${DATA_PATH}"
  --prompt_data_probs "${AGENTGL_DATA_PROBS}"
  --balanced_prompt_mixing
  --graph_encoder_path "${AGENTGL_GRAPH_ENCODER_PATH}"
  --graph_encoder_remote_url "${AGENTGL_GRAPH_ENCODER_REMOTE_URL}"
  --graph_max_searches 5
  --graph_topk "${AGENTGL_GRAPH_TOPK}"
  --graph_topk_similar "${AGENTGL_LP_GRAPH_TOPK_SIMILAR:-3}"
  --graph_topk_one_hop "${AGENTGL_GRAPH_TOPK_ONE_HOP}"
  --graph_topk_two_hop "${AGENTGL_GRAPH_TOPK_TWO_HOP}"
  --graph_topk_pagerank "${AGENTGL_LP_GRAPH_TOPK_PAGERANK:-2}"
  --graph_reflect_after_docs
  --flash_attn
  --use_kl_loss
  --group_method "${AGENTGL_GROUP_METHOD:-normal}"
  --use_length_reward_in_efficiency
  --gradient_checkpointing
  --save_steps "${AGENTGL_LP_SAVE_STEPS:-20}"
  --vllm_sync_backend nccl
  --max_ckpt_num 3
  --temperature "${AGENTGL_TEMPERATURE}"
  --overlap_comm
  --packing_samples
)

cmd+=("${WANDB_ARGS[@]}")
"${cmd[@]}"
