#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <nc|lp> <grpo|reinforcepp> <stage1|stage2>" >&2
  exit 2
fi

TASK="$1"
ALGORITHM="$2"
STAGE="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/agentgl_local.env"

if [[ -f "${AGENTGL_CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${AGENTGL_CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${AGENTGL_CONDA_ENV}"
fi

cd "${AGENTGL_PROJECT_ROOT}"
mkdir -p "${AGENTGL_LOG_DIR}"

export AGENTGL_ENV_FILE="${SCRIPT_DIR}/agentgl_local.env"
export PYTHONPATH="${AGENTGL_PROJECT_ROOT}/OpenRLHF-RAG:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${AGENTGL_CUDA_VISIBLE_DEVICES}"

port_is_open() {
  python - "$1" <<'PY'
import socket
import sys

sock = socket.socket()
sock.settimeout(0.5)
try:
    sock.connect(("127.0.0.1", int(sys.argv[1])))
    print("1")
except OSError:
    print("0")
finally:
    sock.close()
PY
}

start_embedding_server() {
  if [[ "$(port_is_open "${AGENTGL_EMBEDDING_PORT}")" == "1" ]]; then
    echo "Embedding server already responds on port ${AGENTGL_EMBEDDING_PORT}."
    return
  fi

  echo "Starting embedding server on port ${AGENTGL_EMBEDDING_PORT}."
  nohup python train/embedding_server.py \
    --model_path "${AGENTGL_GRAPH_ENCODER_PATH}" \
    --device "${AGENTGL_EMBEDDING_DEVICE}" \
    --port "${AGENTGL_EMBEDDING_PORT}" \
    > "${AGENTGL_LOG_DIR}/embedding_server.log" 2>&1 &
  sleep 2
}

start_reward_server() {
  if [[ "$(port_is_open "${AGENTGL_REWARD_PORT}")" == "1" ]]; then
    echo "Reward server already responds on port ${AGENTGL_REWARD_PORT}."
    return
  fi

  local server_module
  local log_name
  echo "Starting ${TASK} reward server for ${STAGE} on port ${AGENTGL_REWARD_PORT}."

  if [[ "${TASK}" == "nc" ]]; then
    if [[ "${STAGE}" == "stage1" ]]; then
      server_module="train/reward_server_qwen_zero.py"
    else
      server_module="train/reward_server_qwen_stage2.py"
    fi
    log_name="reward_${TASK}_${STAGE}.jsonl"
    nohup python "${server_module}" \
      --data_path "${AGENTGL_NODE_DATA_ROOT}/ogbn-arxiv,${AGENTGL_NODE_DATA_ROOT}/ogbn-products" \
      --port "${AGENTGL_REWARD_PORT}" \
      --log_file "${AGENTGL_LOG_DIR}/${log_name}" \
      > "${AGENTGL_LOG_DIR}/reward_${TASK}_${STAGE}.out" 2>&1 &
  elif [[ "${TASK}" == "lp" ]]; then
    if [[ "${STAGE}" == "stage1" ]]; then
      server_module="train/reward_server_lp.py"
      pair_data="${AGENTGL_PROJECT_ROOT}/data/link_prediction_stage/ogbn-arxiv_stage1.jsonl,${AGENTGL_PROJECT_ROOT}/data/link_prediction_stage/ogbn-products_stage1.jsonl"
    else
      server_module="train/reward_server_lp_stage2.py"
      pair_data="${AGENTGL_PROJECT_ROOT}/data/link_prediction_stage/ogbn-arxiv_stage2.jsonl,${AGENTGL_PROJECT_ROOT}/data/link_prediction_stage/ogbn-products_stage2.jsonl"
    fi
    log_name="reward_${TASK}_${STAGE}.jsonl"
    nohup python "${server_module}" \
      --pair_data "${pair_data}" \
      --port "${AGENTGL_REWARD_PORT}" \
      --log_file "${AGENTGL_LOG_DIR}/${log_name}" \
      > "${AGENTGL_LOG_DIR}/reward_${TASK}_${STAGE}.out" 2>&1 &
  else
    echo "Unknown task: ${TASK}" >&2
    exit 2
  fi
  sleep 2
}

start_ray() {
  if ray status --address="${AGENTGL_RAY_GCS_ADDRESS}" > /dev/null 2>&1; then
    echo "Ray already responds at ${AGENTGL_RAY_GCS_ADDRESS}."
    return
  fi

  echo "Starting Ray head node."
  ray start --head \
    --num-gpus "${AGENTGL_RAY_NUM_GPUS}" \
    --port "${AGENTGL_RAY_PORT}" \
    --dashboard-port "${AGENTGL_RAY_DASHBOARD_PORT}" \
    > "${AGENTGL_LOG_DIR}/ray_start.out" 2>&1
  sleep 8
}

train_script="scripts/train_${TASK}_${ALGORITHM}_${STAGE}.sh"
if [[ ! -f "${train_script}" ]]; then
  echo "Training script not found: ${train_script}" >&2
  exit 2
fi

start_embedding_server
start_reward_server
start_ray

echo "Launching ${train_script}."
bash "${train_script}" > "${AGENTGL_LOG_DIR}/${TASK}_${ALGORITHM}_${STAGE}.out" 2>&1
echo "Done. Main log: ${AGENTGL_LOG_DIR}/${TASK}_${ALGORITHM}_${STAGE}.out"
