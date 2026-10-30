# Adapted from QwenLM/Qwen3-VL (qwen-vl-finetune, commit 50068df), Apache-2.0.
# Modified for CogSense: LVIP and attention-implementation arguments, GFNArguments (RL).

import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={"help": "flash_attention_2 (required by --data_flatten/--data_packing), sdpa or eager."},
    )

    # CogSense: Latent Visual Imagery Prediction (LVIP), Qwen3-VL only
    lvip_head: bool = field(
        default=False,
        metadata={"help": "Attach the LVIP head and train with CE + lvip_loss_weight * MSE."},
    )
    lvip_loss_weight: float = field(
        default=0.1, metadata={"help": "beta in L_SFT = CE + beta * MSE(g(mean(H_opt)), Enc_vis(V_y))."}
    )

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    video_fps: float = 2


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None

    ## Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)


@dataclass
class GFNArguments:
    """CogSense RL with latent rationales (qwenvl/train/train_gfn.py)."""

    explore_nums: int = field(default=6, metadata={"help": "m: rationales sampled from q_theta per question."})
    explore_min_bs: int = field(
        default=2, metadata={"help": "Rationales sampled in parallel; explore_nums must be a multiple of it."}
    )
    rat_max_len: int = field(default=1024, metadata={"help": "Maximum rationale length (tokens)."})
    rat_min_len: int = field(default=16, metadata={"help": "Minimum length of sampled rationales (tokens)."})
    pf_temp_low: float = field(default=0.5, metadata={"help": "Lowest sampling temperature (tempered samples)."})
    pf_temp_high: float = field(default=1.0, metadata={"help": "Highest sampling temperature (tempered samples)."})
    reward_alpha: float = field(default=1.0, metadata={"help": "alpha: weight of log q_theta0(y | X, Z)."})
    reward_gamma: float = field(default=1.0, metadata={"help": "gamma: weight of R_lvip."})
    reward_anchor_stride: int = field(default=8, metadata={"help": "lambda: stride of the reward anchors."})
    reward_batch_size: int = field(default=8, metadata={"help": "Prefixes scored per forward pass of the scorer."})
    reward_temp_start: float = field(default=1.0, metadata={"help": "Rewards are divided by this temperature."})
    reward_temp_end: float = field(default=1.0)
    reward_temp_horizon: int = field(default=50)
    delta_start: float = field(
        default=0.1, metadata={"help": "delta_s at step 0: keep Z_i if R(Z_i) >= R(Z_ref) + log(delta_s)."}
    )
    delta_end: float = field(default=1.0, metadata={"help": "delta_s after delta_horizon optimizer steps."})
    delta_horizon: int = field(default=50)
    subtb_lambda: float = field(default=1.0, metadata={"help": "Weight lambda^(j-i-1) of sub-trajectory (i, j)."})

    def __post_init__(self):
        if self.explore_nums < 1 or self.explore_min_bs < 1 or self.explore_nums % self.explore_min_bs:
            raise ValueError("explore_nums must be a positive multiple of explore_min_bs")
        if not (0 < self.delta_start <= 1 and 0 < self.delta_end <= 1):
            raise ValueError("delta_start and delta_end must be in (0, 1]")
        if self.reward_anchor_stride < 1 or self.rat_max_len < 1:
            raise ValueError("reward_anchor_stride and rat_max_len must be positive")
