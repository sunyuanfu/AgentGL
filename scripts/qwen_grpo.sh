

# NODE_RANK=$1

# export TORCH_HOME=/PATH/TO/WORKDIR
export NUMEXPR_MAX_THREADS=128
export RAY_DEDUP_LOGS=0

# Your wandb token
wandb_token=YOUR_WANDB_TOKEN
# sudo rm -rf ~/.netrc
export WANDB_API_KEY=${wandb_token}

# Path of training data
DATA_PATH=xx
# /PATH/TO/HOME/user/OpenRLHF/data/demo_dataset
#
# Path of backbone model(DeepSeek-R1-Distill-Qwen-1.5B)
TOKENIZER_PATH=/PATH/TO/WORKDIR/model/Qwen2.5-7B
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
N_SAMPLES=16
EPISODE=10000
WARMUP=0.0
TBS=64
RBS=16
KL=0.0
LR=2e-6
MAX_LENGTH=29000
PORT=1278
TEMP=1.0
# REWARD_MODEL=server_false-1_true1_unknown-1-repeat-single
REWARD_MODEL=server_dpsk_tuple
SAVE_MODEL_NAME=lqwen_base_grpo_221_new_grpo_kl0

GROUP_METHOD=normal

LOG_BASE=log

mkdir -p /PATH/TO/WORKDIR/rag_rl/results/$SAVE_MODEL_NAME
mkdir -p /PATH/TO/WORKDIR/rag_rl/results/ckpts
mkdir -p /PATH/TO/WORKDIR/rag_rl/results/$SAVE_MODEL_NAME/server
mkdir -p $LOG_BASE/server/

# pkill -f ${REWARD_MODEL}
# nohup python -m openrlhf.cli.${REWARD_MODEL} --data_path $DATA_PATH --reward_pretrain $TOKENIZER_PATH --log_file /PATH/TO/WORKDIR/RAG_RL/results/$SAVE_MODEL_NAME/server/sampling.jsonl --port ${PORT} > $LOG_BASE/server/$SAVE_MODEL_NAME-node$NODE_RANK.log 2>&1 &
# echo $LOG_BASE/server/$SAVE_MODEL_NAME-node$NODE_RANK.log

ray job submit --address="http://10.119.20.150" \
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
   --max_samples 100000 \
   --max_epochs 1 \
   --num_episodes ${EPISODE} \
   --lr_warmup_ratio ${WARMUP} \
   --n_samples_per_prompt $N_SAMPLES \
   --prompt_max_len 1024 \
   --generate_max_len $MAX_LENGTH \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate $LR \
   --critic_learning_rate 9e-6 \
   --init_kl_coef $KL \
   --prompt_data $DATA_PATH \
   --input_key question \
   --flash_attn \
   --use_kl_loss \
   --gradient_checkpointing \
   --save_steps 40 \
   --vllm_sync_backend nccl \
   --max_ckpt_num 2 \
   --group_method $GROUP_METHOD \
   --use_length_reward_in_efficiency \
   --temperature $TEMP \
   --overlap_comm \
   --packing_samples \
   --use_wandb ${wandb_token} \
   --wandb_run_name $SAVE_MODEL_NAME \
   # --apply_chat_template \
