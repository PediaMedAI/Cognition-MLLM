# Copyright 2026 The CogSense Authors.
# Licensed under the Apache License, Version 2.0.
"""Convert CogSense records with reasoning chains into qwen-vl-finetune annotations.

Input: a .jsonl (or .json list) file, one question per record:

    {"problem_id": 0, "category": "Fluid Intelligence",
     "prompt": "Based on the pattern shown in the question image, which option completes the sequence?",
     "question": "fluid_intelligence/1/question/q.jpeg",      # str, list of str, or list of
                                                               # {"before": ..., "after": ...} pairs
     "options": ["fluid_intelligence/1/options/a.jpeg", ...],
     "solution": "<answer>A</answer>",
     "reasoning": ["step 1", "step 2", ...]}                   # list of steps or one string

Output: a .json list in the qwen-vl-finetune format, plus the LVIP fields:

    {"id": 0, "category": "Fluid Intelligence",
     "image": ["fluid_intelligence/1/question/q.jpeg", "fluid_intelligence/1/options/a.jpeg", ...],
     "conversations": [{"from": "human", "value": "<system prompt>\\n\\n<prompt>...<image>..."},
                       {"from": "gpt", "value": "step 1\\nstep 2\\n\\nSo, the final answer is: <answer>A</answer>"}],
     "strip_text": false,
     "lvip_option_indices": [1, 2, 3, 4],
     "lvip_target_index": 1}

The human turn is built with the prompt code of the evaluation script
(make_chat_input and the message conversion of call_model in
evaluation/eval_lvip.py), so the training prompt is token-identical to the
evaluation prompt. "strip_text": false keeps the whitespace around <image>.

Usage:
    python tools/build_cogsense_data.py --input cogsense_sft_records.jsonl \\
        --image_root ../data/CogSense-Dataset --output ../data/cogsense_sft.json
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation import eval_lvip  # noqa: E402  (prompt code shared with evaluation)


def to_bench_entry(record: Dict) -> Dict:
    """Convert a CogSense-Dataset record into the CogSense-Bench schema used by make_chat_input."""
    entry = {
        "problem_id": record.get("problem_id"),
        "problem": record["prompt"],
        "options": record["options"],
        "solution": record["solution"],
    }
    question = record["question"]
    if isinstance(question, list) and question and isinstance(question[0], dict):
        # Before/after example pairs (ARC-AGI style).
        entry["pattern"] = [
            {
                "before_image_path": pair.get("before", pair.get("before_image_path")),
                "after_image_path": pair.get("after", pair.get("after:", pair.get("after_image_path"))),
            }
            for pair in question
        ]
    else:
        entry["path"] = question
    return entry


def answer_letter(solution: str) -> str:
    match = re.search(r"<answer>\s*([A-Z])\s*</answer>", solution, re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot parse the answer from solution {solution!r}")
    return match.group(1).upper()


def build_target(record: Dict) -> str:
    """x = (Z, y): the reasoning steps, then the final answer."""
    reasoning = record.get("reasoning")
    if isinstance(reasoning, (list, tuple)):
        reasoning = "\n".join(str(step).strip() for step in reasoning if str(step).strip())
    reasoning = (reasoning or "").strip()
    answer = f"<answer>{answer_letter(record['solution'])}</answer>"
    if not reasoning:
        return f"So, the final answer is: {answer}"
    return f"{reasoning}\n\nSo, the final answer is: {answer}"


def chat_input_with_paths(entry: Dict, image_root: str) -> Tuple[List[Dict], List[str]]:
    """Run make_chat_input, recording the image paths instead of base64-encoding the files."""
    paths = []

    def record_path(path):
        paths.append(path)
        return str(len(paths) - 1)

    original = eval_lvip.encode_image_to_base64
    eval_lvip.encode_image_to_base64 = record_path
    try:
        messages = eval_lvip.make_chat_input(entry, image_root)
    finally:
        eval_lvip.encode_image_to_base64 = original
    return messages, paths


def human_turn(messages: List[Dict]) -> str:
    """The user turn of call_model as one string with <image> placeholders: the system
    prompt plus "\\n\\n" is prepended to the user content, which is kept verbatim."""
    system_prompt = next((m["content"] for m in messages if m["role"] == "system"), None)
    users = [m for m in messages if m["role"] == "user"]
    if len(users) != 1:
        raise ValueError(f"expected one user turn, got {len(users)}")
    parts = [f"{system_prompt}\n\n"] if system_prompt else []
    content = users[0]["content"]
    for item in content if isinstance(content, list) else [{"type": "text", "text": content}]:
        if item["type"] == "text":
            if "<image>" in item["text"] or "<video>" in item["text"]:
                raise ValueError("prompt text contains a literal <image>/<video> tag")
            parts.append(item["text"])
        elif item["type"] == "image_url" and item["image_url"]["url"].startswith("data:image"):
            parts.append("<image>")
    return "".join(parts)


def convert_record(record: Dict, image_root: str) -> Dict:
    entry = to_bench_entry(record)
    messages, paths = chat_input_with_paths(entry, image_root)
    human = human_turn(messages)
    if human.count("<image>") != len(paths):
        raise ValueError("number of <image> tags and images differ")
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"missing image {missing[0]}")

    # LVIP: the options are the last images of the prompt, in A, B, C, ... order.
    num_images, options = len(paths), entry["options"]
    first_option = num_images - len(options)
    if not options or first_option < 0:
        raise ValueError("no option images")
    for i, option in enumerate(options):
        if os.path.normpath(paths[first_option + i]) != os.path.normpath(os.path.join(image_root, option)):
            raise ValueError("option images are not the last images of the prompt")
    answer_index = ord(answer_letter(record["solution"])) - ord("A")
    if not 0 <= answer_index < len(options):
        raise ValueError(f"answer index {answer_index} out of range")

    return {
        "id": record.get("problem_id"),
        "category": record.get("category"),
        "image": [os.path.relpath(p, image_root) for p in paths],
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt", "value": build_target(record)},
        ],
        "strip_text": False,
        "lvip_option_indices": list(range(first_option, num_images)),
        "lvip_target_index": first_option + answer_index,
    }


def read_records(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="records with reasoning chains (.jsonl or .json)")
    parser.add_argument("--image_root", required=True, help="directory the image paths are relative to")
    parser.add_argument("--output", required=True, help="output annotation file (.json)")
    args = parser.parse_args()

    image_root = os.path.abspath(args.image_root)
    samples, skipped, no_reasoning = [], 0, 0
    for record in read_records(args.input):
        try:
            samples.append(convert_record(record, image_root))
        except (ValueError, KeyError, FileNotFoundError) as e:
            skipped += 1
            print(f"skip problem {record.get('problem_id')}: {e}", file=sys.stderr)
            continue
        if not record.get("reasoning"):
            no_reasoning += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)
    print(f"wrote {len(samples)} samples to {args.output} (skipped {skipped}, without reasoning {no_reasoning})")


if __name__ == "__main__":
    main()
