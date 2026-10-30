#!/bin/bash
# CogSense-8B: RL with latent rationales (GFlowNet fine-tuning, LoRA) of the SFT
# checkpoint, 8 GPUs, DeepSpeed ZeRO-3, then merge the adapter.
# Usage (from any directory): bash qwen-vl-finetune/scripts/rl_cogsense.sh
#
# SFT_MODEL is the output of scripts/sft_cogsense.sh (it must have the LVIP head).
# Paper setting: learning rate 1e-5, weight decay 1e-5. The paper does not give the
# other values: sampling, rationale length, anchor stride and LoRA follow LaCoT;
# alpha = gamma = 1, the delta_s schedule and rat_min_len are defaults of this code.
# Needs flash-attn; without it run: ATTN_IMPL=sdpa bash qwen-vl-finetune/scripts/rl_cogsense.sh

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
sft_model=${SFT_MODEL:-./output/cogsense-8b-sft-lvip}

# Training entry point
entry_file=qwenvl/train/train_gfn.py

# Dataset configuration (see qwenvl/data/__init__.py)
datasets=cogsense_sft

# Output configuration
run_name="cogsense-8b-rl"
output_dir=./output/cogsense-8b-rl-lora
merged_dir=./output/cogsense-8b-rl

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${sft_model} \
    --dataset_use ${datasets} \
    --attn_implementation ${ATTN_IMPL:-flash_attention_2} \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.0 \
    --explore_nums 6 \
    --explore_min_bs 2 \
    --rat_max_len 1024 \
    --rat_min_len 16 \
    --pf_temp_low 0.5 \
    --pf_temp_high 1.0 \
    --reward_alpha 1.0 \
    --reward_gamma 1.0 \
    --reward_anchor_stride 8 \
    --reward_batch_size 8 \
    --delta_start 0.1 \
    --delta_end 1.0 \
    --delta_horizon 50 \
    --subtb_lambda 1.0 \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --weight_decay 1e-5 \
    --warmup_ratio 0.05 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --run_name ${run_name} \
    --report_to tensorboard"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --nnodes=${NNODES} \
         --node_rank=${NODE_RANK} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}

# Merge the LoRA adapter into the SFT checkpoint
if [ "${NODE_RANK}" = "0" ]; then
    python tools/merge_lora.py --base ${sft_model} --adapter ${output_dir} --output ${merged_dir}
fi
