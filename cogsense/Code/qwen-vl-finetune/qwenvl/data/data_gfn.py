# Copyright 2026 The CogSense Authors.
# Licensed under the Apache License, Version 2.0.
"""Data for RL with latent rationales (GFlowNet fine-tuning, following LaCoT).

Uses the same annotations as SFT (tools/build_cogsense_data.py). The assistant
target "<reasoning>\\n\\nSo, the final answer is: <answer>X</answer>" of every
sample is split into

    X      the prompt: the user turn + "<|im_start|>assistant\\n", built exactly as
           at evaluation time
    Z_ref  the reference rationale: the reasoning chain, as token ids
    y      the answer: "So, the final answer is: <answer>X</answer><|im_end|>"

The end of a rationale (the termination action) is the "\\n\\n" token that separates
the reasoning from the answer in the SFT format; it is added by the trainer.
"""

from pathlib import Path
from typing import Dict, Sequence

import torch

from .data_processor import LazySupervisedDataset, _build_messages, lvip_fields

ANSWER_PREFIX = "So, the final answer is:"


def split_target(target: str):
    """Split an SFT target into the rationale text and the answer text."""
    index = target.rfind(ANSWER_PREFIX)
    if index < 0:
        raise ValueError(f"no {ANSWER_PREFIX!r} in the target")
    return target[:index].rstrip(), target[index:].strip()


class GFNDataset(LazySupervisedDataset):
    """One question per item: prompt, images, reference rationale and answer."""

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        source = sources[0]
        messages = _build_messages(source, Path(source.get("data_path", "")))
        if messages[-1]["role"] != "assistant":
            raise ValueError("the last turn must be the assistant target")
        prompt = self.processor.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        rationale, answer = split_target(source["conversations"][-1]["value"])
        tokenizer = self.processor.tokenizer
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        rationale_ids = tokenizer(rationale, add_special_tokens=False)["input_ids"] if rationale else []
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"] + [im_end]

        item = {
            "input_ids": prompt["input_ids"],
            "pixel_values": prompt["pixel_values"],
            "image_grid_thw": prompt["image_grid_thw"],
            "ref_rationale_ids": torch.tensor(rationale_ids, dtype=torch.long),
            "answer_ids": torch.tensor(answer_ids, dtype=torch.long),
        }
        item.update(lvip_fields(source, prompt))
        return item


class GFNDataCollator:
    """The GFlowNet trainer handles one question per device and step."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if len(instances) != 1:
            raise ValueError("RL with latent rationales needs --per_device_train_batch_size 1")
        return dict(instances[0])
