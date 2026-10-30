# Copyright 2026 The CogSense Authors.
# Licensed under the Apache License, Version 2.0.
"""CogSense RL with latent rationales: GFlowNet fine-tuning of an SFT checkpoint.

The rationale policy q_theta is the SFT model (scripts/sft_cogsense.sh) with LoRA
adapters on the linear layers of the language model. The frozen scorer q_theta0
is the same model with the adapters disabled, and the LVIP head g_psi stays
frozen. See qwenvl/train/gfn_trainer.py for the objective.

Usage: scripts/rl_cogsense.sh. Saves the LoRA adapter and the processor;
tools/merge_lora.py merges the adapter into a full checkpoint.
"""

import logging
import os
import pathlib
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor
from transformers.integrations import is_deepspeed_zero3_enabled

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from qwenvl.data.data_gfn import GFNDataCollator, GFNDataset  # noqa: E402
from qwenvl.models.modeling_cogsense import CogSenseForConditionalGeneration  # noqa: E402
from qwenvl.train.argument import DataArguments, GFNArguments, ModelArguments, TrainingArguments  # noqa: E402
from qwenvl.train.fix_rope_device import patch_rope_for_deepspeed_zero3  # noqa: E402
from qwenvl.train.gfn_trainer import GFNTrainer  # noqa: E402

# LoRA on every linear layer of the language model; the vision tower, the merger,
# the LVIP head and lm_head are not adapted.
LORA_TARGET_MODULES = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def rank0_print(*args):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(*args)


def load_policy(model_args, training_args, gfn_args):
    model = CogSenseForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=model_args.attn_implementation,
        dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    if gfn_args.reward_gamma != 0 and model.lvip_head is None:
        raise ValueError(
            "R_lvip needs the LVIP head: start from a CogSense SFT checkpoint "
            "(scripts/sft_cogsense.sh), or set --reward_gamma 0."
        )
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad = False
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        model.print_trainable_parameters()
    return model


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, GFNArguments))
    model_args, data_args, training_args, gfn_args = parser.parse_args_into_dataclasses()

    if training_args.per_device_train_batch_size != 1:
        raise ValueError("RL with latent rationales needs --per_device_train_batch_size 1")
    if data_args.data_flatten or data_args.data_packing:
        raise ValueError("--data_flatten / --data_packing are not supported by the RL stage")
    if model_args.lvip_head:
        rank0_print("--lvip_head is ignored: the LVIP head is taken from the SFT checkpoint and kept frozen.")
    # The batches carry the reference rationale and the answer, which the model forward does not take.
    training_args.remove_unused_columns = False
    os.makedirs(training_args.output_dir, exist_ok=True)

    if is_deepspeed_zero3_enabled():
        patch_rope_for_deepspeed_zero3()

    model = load_policy(model_args, training_args, gfn_args)
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    data_args.model_type = "qwen3vl"
    data_args.lvip = True
    train_dataset = GFNDataset(processor, data_args)

    trainer = GFNTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=GFNDataCollator(),
        processing_class=processor,
        gfn_args=gfn_args,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # Saves the LoRA adapter (and the processor).
    trainer.save_model(training_args.output_dir)
    if trainer.args.should_save:
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
