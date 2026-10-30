# Copyright 2026 The CogSense Authors.
# Licensed under the Apache License, Version 2.0.
"""Merge the LoRA adapter of the RL stage (scripts/rl_cogsense.sh) into its SFT
checkpoint and save a full CogSense checkpoint (with the LVIP head), loadable by
evaluation/eval_lvip.py.

    python tools/merge_lora.py --adapter ./output/cogsense-8b-rl-lora \
        --output ./output/cogsense-8b-rl
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from qwenvl.models.modeling_cogsense import CogSenseForConditionalGeneration  # noqa: E402


def merge(adapter: str, output: str, base: str = None, dtype: str = "bfloat16"):
    from peft import PeftModel
    from transformers import AutoProcessor

    if base is None:
        with open(os.path.join(adapter, "adapter_config.json")) as f:
            base = json.load(f)["base_model_name_or_path"]
    model = CogSenseForConditionalGeneration.from_pretrained(base, dtype=getattr(torch, dtype))
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(output)
    # The RL run saves its processor (with its image pixel limits) next to the adapter.
    try:
        processor = AutoProcessor.from_pretrained(adapter)
    except (OSError, ValueError):
        processor = AutoProcessor.from_pretrained(base)
    processor.save_pretrained(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, help="Output directory of scripts/rl_cogsense.sh.")
    parser.add_argument("--output", required=True, help="Directory of the merged checkpoint.")
    parser.add_argument("--base", default=None, help="SFT checkpoint (default: the adapter's base model).")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()
    print(f"Merged checkpoint saved to {merge(args.adapter, args.output, args.base, args.dtype)}")


if __name__ == "__main__":
    main()
