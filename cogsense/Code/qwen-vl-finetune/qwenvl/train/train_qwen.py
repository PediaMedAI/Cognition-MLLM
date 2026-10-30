# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# Adapted from QwenLM/Qwen3-VL (qwen-vl-finetune, commit 50068df), Apache-2.0.
# Modified for CogSense: --lvip_head trains Qwen3-VL with the LVIP head
# (qwenvl/models/modeling_cogsense.py), --attn_implementation, a RoPE device fix
# for DeepSpeed ZeRO-3, and set_model uses model.model.* (transformers 5) and also
# freezes lm_head.

import contextlib
import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class, CogSenseTrainer

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from transformers import AutoConfig
from transformers.integrations import is_deepspeed_zero3_enabled
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.models.modeling_cogsense import CogSenseForConditionalGeneration
from qwenvl.train.fix_rope_device import patch_rope_for_deepspeed_zero3
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.language_model.named_parameters():
            p.requires_grad = True
        for p in model.lm_head.parameters():
            p.requires_grad = True
    else:
        for n, p in model.model.language_model.named_parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = False

    # The LVIP head is always trained.
    if getattr(model, "lvip_head", None) is not None:
        for p in model.lvip_head.parameters():
            p.requires_grad = True


def init_lvip_head(model):
    """N(0, initializer_range) weights and zero biases for a newly added LVIP head."""
    head = model.lvip_head
    std = getattr(model.config.text_config, "initializer_range", 0.02)
    if is_deepspeed_zero3_enabled():
        import deepspeed

        context = deepspeed.zero.GatheredParameters(list(head.parameters()), modifier_rank=0)
    else:
        context = contextlib.nullcontext()
    with context, torch.no_grad():
        for layer in head:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, mean=0.0, std=std)
                torch.nn.init.zeros_(layer.bias)


def load_cogsense_model(model_args, training_args, attn_implementation):
    """Qwen3-VL + LVIP head. The head is stored as `model.nfp_head` (config fields
    `nfp_head`, `nfp_mse_loss_weight`), as in the released CogSense-8B checkpoint."""
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir)
    if config.model_type != "qwen3_vl":
        raise ValueError(f"--lvip_head needs a dense Qwen3-VL checkpoint, got model_type={config.model_type}")
    config.nfp_head = True
    config.nfp_mse_loss_weight = model_args.lvip_loss_weight
    model, loading_info = CogSenseForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        dtype=(torch.bfloat16 if training_args.bf16 else None),
        output_loading_info=True,
    )
    if any("nfp_head" in key for key in loading_info["missing_keys"]):
        rank0_print("LVIP head not found in the checkpoint: initialized from N(0, 0.02).")
        init_lvip_head(model)
    else:
        rank0_print("LVIP head loaded from the checkpoint.")
    rank0_print(
        f"LVIP: beta={config.nfp_mse_loss_weight}, "
        f"head {config.text_config.hidden_size} -> {config.vision_config.hidden_size}"
    )
    return model


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)
    attn_implementation = model_args.attn_implementation or attn_implementation

    if is_deepspeed_zero3_enabled():
        patch_rope_for_deepspeed_zero3()

    if model_args.lvip_head:
        model = load_cogsense_model(model_args, training_args, attn_implementation)
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    data_args.lvip = model_args.lvip_head  # add the LVIP metadata to every sample

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.lora_enable and model_args.lvip_head:
        raise ValueError("--lvip_head is trained with full fine-tuning; --lora_enable is not supported with it.")

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention linear layers of Qwen
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            model.model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()
    
    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer_cls = CogSenseTrainer if model_args.lvip_head else Trainer
    trainer = trainer_cls(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
