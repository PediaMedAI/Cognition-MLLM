# Copyright 2026 The CogSense Authors.
# Licensed under the Apache License, Version 2.0.
"""Fix a device mismatch in the Qwen3-VL RoPE under DeepSpeed ZeRO-3.

With ZeRO-3 the non-persistent RoPE buffers can stay on the CPU. Move them to
the device of the activations before they are used.
"""

from transformers.models.qwen3_vl import modeling_qwen3_vl


def patch_rope_for_deepspeed_zero3():
    original_vision_rope = modeling_qwen3_vl.apply_rotary_pos_emb_vision

    def apply_rotary_pos_emb_vision(q, k, cos, sin):
        return original_vision_rope(q, k, cos.to(q.device), sin.to(q.device))

    modeling_qwen3_vl.apply_rotary_pos_emb_vision = apply_rotary_pos_emb_vision

    original_text_rope_forward = modeling_qwen3_vl.Qwen3VLTextRotaryEmbedding.forward

    def text_rope_forward(self, x, position_ids, *args, **kwargs):
        if self.inv_freq.device != position_ids.device:
            self.inv_freq = self.inv_freq.to(position_ids.device)
        return original_text_rope_forward(self, x, position_ids, *args, **kwargs)

    modeling_qwen3_vl.Qwen3VLTextRotaryEmbedding.forward = text_rope_forward
