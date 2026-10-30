# Copyright 2026 The CogSense Authors.
# Licensed under the Apache License, Version 2.0.
"""CogSense: Qwen3-VL with a Latent Visual Imagery Prediction (LVIP) head.

SFT with LVIP from "Toward Cognitive Supersensing in Multimodal Large Language
Model" (arXiv:2602.01541), Sec. 4.3:

    H_opt      final-layer backbone hidden states of all option-image tokens
    h_opt_bar  = mean(H_opt)                          (average pooling)
    h_y_hat    = g_psi(h_opt_bar)                     (two-layer MLP, the LVIP head)
    h_y        = Enc_vis(V_y)                         (frozen visual-encoder embedding
                                                       of the ground-truth option image)
    L_SFT      = CE(x = (Z, y) | X) + beta * MSE(h_y_hat, h_y)

Enc_vis is the Qwen3-VL vision transformer *before* the vision-language
projector (the patch merger, P in the paper). Its patch features of V_y are
mean-pooled into one vector and used as a detached target.

Checkpoint compatibility: the LVIP head is stored as ``model.nfp_head`` and is
configured by the config fields ``nfp_head`` (bool) and ``nfp_mse_loss_weight``
(beta). These names match the released ``PediaMedAI/CogSense-8B`` checkpoint,
so it loads into this class together with its LVIP head weights.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast


class LVIPHead(nn.Sequential):
    """g_psi: Linear(d, d) -> GELU -> Linear(d, d_vis)."""

    def __init__(self, text_hidden_size: int, vision_hidden_size: int):
        super().__init__(
            nn.Linear(text_hidden_size, text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, vision_hidden_size),
        )


class CogSenseForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Qwen3-VL + LVIP head. Generation is unchanged: the head is only used for the
    auxiliary training loss."""

    # The Trainer divides the loss by the gradient-accumulation steps (no num_items_in_batch).
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "nfp_head", False):
            self.model.nfp_head = LVIPHead(config.text_config.hidden_size, config.vision_config.hidden_size)
            # N(0, 0.02) weights, zero bias. When loading a checkpoint without the head
            # (e.g. Qwen/Qwen3-VL-8B-Instruct), transformers' default Linear init for the
            # missing weights gives the same N(0, initializer_range=0.02) / zero bias.
            std = getattr(config.text_config, "initializer_range", 0.02)
            for layer in self.model.nfp_head:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=std)
                    nn.init.zeros_(layer.bias)
        else:
            self.model.nfp_head = None

    @property
    def lvip_head(self) -> Optional[LVIPHead]:
        return self.model.nfp_head

    def save_pretrained(self, save_directory, *args, **kwargs):
        """Save as a standard Qwen3-VL checkpoint (``architectures`` stays
        ``Qwen3VLForConditionalGeneration``, like the released CogSense-8B), so it also
        loads with stock Qwen3-VL code, which ignores the extra LVIP head weights."""
        super().save_pretrained(save_directory, *args, **kwargs)
        self.config.architectures = ["Qwen3VLForConditionalGeneration"]
        is_main_process = kwargs.get("is_main_process", args[0] if args else True)
        if is_main_process:
            self.config.save_pretrained(save_directory)

    # ------------------------------------------------------------------
    # LVIP
    # ------------------------------------------------------------------
    def compute_lvip(
        self,
        input_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        image_grid_thw: torch.LongTensor,
        vision_features: torch.Tensor,
        lvip_option_image_mask: torch.Tensor,
        lvip_target_image_index: torch.LongTensor,
        lvip_image_sample_index: Optional[torch.LongTensor] = None,
    ):
        """Return (prediction h_y_hat [S, d_vis], target h_y [S, d_vis], valid [S]) for the
        S samples of the batch.

        Args:
            input_ids: [B, L] token ids (image placeholders are ``config.image_token_id``).
                With ``data_flatten``/``data_packing`` several samples share one row.
            hidden_states: [B, L, d] final-layer hidden states of the LLM backbone.
            image_grid_thw: [N_img, 3] grids of all images in the batch (batch order).
            vision_features: [sum_i t*h*w, d_vis] vision-encoder patch features before
                the patch merger, for all images in the batch (batch order).
            lvip_option_image_mask: [N_img] bool, True for candidate option images.
            lvip_target_image_index: [S] index (into the batch image list) of the
                ground-truth option image V_y of each sample, or -1 if none.
            lvip_image_sample_index: [N_img] sample (0..S-1) of each image. If None, every
                row of ``input_ids`` is one sample.
        """
        device = hidden_states.device
        hidden_size = hidden_states.shape[-1]
        num_images = image_grid_thw.shape[0]
        merge_size = self.model.visual.spatial_merge_size
        patches_per_image = image_grid_thw.prod(-1).to(device)
        tokens_per_image = patches_per_image // (merge_size**2)
        image_ids = torch.arange(num_images, device=device)

        # Target h_y: mean-pooled frozen visual-encoder features of each image.
        vision_features = vision_features.detach().float().to(device)
        if vision_features.shape[0] != int(patches_per_image.sum()):
            raise ValueError("Vision features do not match image_grid_thw.")
        patch_image_id = torch.repeat_interleave(image_ids, patches_per_image)
        image_sum = vision_features.new_zeros(num_images, vision_features.shape[-1])
        image_sum.index_add_(0, patch_image_id, vision_features)
        image_mean = image_sum / patches_per_image.unsqueeze(-1).float()

        # H_opt: hidden states of option-image tokens, average-pooled per sample.
        image_token_mask = input_ids == self.config.image_token_id
        token_hidden = hidden_states[image_token_mask]  # row-major = image order
        if token_hidden.shape[0] != int(tokens_per_image.sum()):
            raise ValueError("Image placeholder tokens do not match image_grid_thw (truncated sequence?).")
        token_image_id = torch.repeat_interleave(image_ids, tokens_per_image)
        if lvip_image_sample_index is None:
            token_sample_id = image_token_mask.nonzero(as_tuple=True)[0]
        else:
            token_sample_id = lvip_image_sample_index.to(device).long()[token_image_id]
        num_samples = lvip_target_image_index.numel()
        is_option_token = lvip_option_image_mask.to(device).bool()[token_image_id]

        pooled_sum = hidden_states.new_zeros(num_samples, hidden_size, dtype=torch.float32)
        pooled_sum.index_add_(0, token_sample_id[is_option_token], token_hidden[is_option_token].float())
        option_count = hidden_states.new_zeros(num_samples, dtype=torch.float32)
        option_count.index_add_(
            0, token_sample_id[is_option_token], torch.ones_like(token_sample_id[is_option_token], dtype=torch.float32)
        )
        pooled = pooled_sum / option_count.clamp(min=1.0).unsqueeze(-1)

        # The head runs on every sample (ZeRO-3 needs every rank to call it).
        head_dtype = self.model.nfp_head[0].weight.dtype
        prediction = self.model.nfp_head(pooled.to(head_dtype))

        target_index = lvip_target_image_index.to(device).long().flatten()
        valid = (target_index >= 0) & (option_count > 0)
        target = image_mean[target_index.clamp(min=0)]
        return prediction, target, valid

    @staticmethod
    def lvip_mse(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """MSE(h_y_hat, h_y), averaged over feature dims and valid samples."""
        per_sample = F.mse_loss(prediction.float(), target.float(), reduction="none").mean(-1)
        valid = valid.float()
        return (per_sample * valid).sum() / valid.sum().clamp(min=1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        logits_to_keep=0,
        lvip_option_image_mask: Optional[torch.Tensor] = None,
        lvip_target_image_index: Optional[torch.LongTensor] = None,
        lvip_image_sample_index: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        lvip_batch = self.model.nfp_head is not None and lvip_target_image_index is not None
        compute_lvip = (
            lvip_batch and lvip_option_image_mask is not None and pixel_values is not None and input_ids is not None
        )

        # Capture Enc_vis(V): the input of the vision patch merger (the projector P).
        captured = {}
        hook = None
        if compute_lvip:

            def _capture(module, args, hook_kwargs):
                captured["vision_features"] = args[0] if args else hook_kwargs["hidden_states"]

            hook = self.model.visual.merger.register_forward_pre_hook(_capture, with_kwargs=True)

        # Only forward version-specific arguments when they are set
        # (cache_position: transformers 4.x, mm_token_type_ids: transformers 5.x).
        if cache_position is not None:
            kwargs["cache_position"] = cache_position
        if mm_token_type_ids is not None:
            kwargs["mm_token_type_ids"] = mm_token_type_ids
        try:
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        finally:
            if hook is not None:
                hook.remove()

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        if compute_lvip:
            prediction, target, valid = self.compute_lvip(
                input_ids,
                hidden_states,
                image_grid_thw,
                captured["vision_features"],
                lvip_option_image_mask,
                lvip_target_image_index,
                lvip_image_sample_index,
            )
            lvip_loss = self.lvip_mse(prediction, target, valid)
            beta = float(getattr(self.config, "nfp_mse_loss_weight", 0.0))
            loss = beta * lvip_loss if loss is None else loss + beta * lvip_loss
            self._last_lvip_loss = lvip_loss.detach()
            # Used by the RL stage for the LVIP reward.
            self._last_lvip_outputs = (prediction.detach(), target.detach(), valid)
        elif lvip_batch and loss is not None:
            # A batch without images: still call the head (with zero weight) so that
            # every rank runs the same modules under DeepSpeed ZeRO-3.
            head_dtype = self.model.nfp_head[0].weight.dtype
            loss = loss + 0.0 * self.model.nfp_head(hidden_states.new_zeros(1, hidden_states.shape[-1], dtype=head_dtype)).sum()

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )
