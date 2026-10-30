#!/bin/bash
# CogSense-8B: SFT of Qwen3-VL-8B with the LVIP head, 8 GPUs, DeepSpeed ZeRO-3.
# Usage (from any directory): bash qwen-vl-finetune/scripts/sft_cogsense.sh
#
# Paper setting: 8x H200, Adam, learning rate 1e-5, weight decay 1e-5, beta = 0.1.
# Batch size, epochs, warmup and scheduler are not given in the paper.
# Needs flash-attn; without it run: ATTN_IMPL=sdpa bash qwen-vl-finetune/scripts/sft_cogsense.sh

set -e
cd "$(dirname "$0")/.."

# Distributed training configuration
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=Qwen/Qwen3-VL-8B-Instruct

# Training hyperparameters
lr=1e-5
batch_size=4
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (see qwenvl/data/__init__.py)
datasets=cogsense_sft

# Output configuration
run_name="cogsense-8b-sft-lvip"
output_dir=./output/cogsense-8b-sft-lvip

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten False \
    --lvip_head True \
    --lvip_loss_weight 0.1 \
    --attn_implementation ${ATTN_IMPL:-flash_attention_2} \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate ${lr} \
    --weight_decay 1e-5 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to tensorboard"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --nnodes=${NNODES} \
         --node_rank=${NODE_RANK} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
