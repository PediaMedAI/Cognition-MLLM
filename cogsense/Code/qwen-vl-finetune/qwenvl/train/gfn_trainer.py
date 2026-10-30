# Copyright 2026 The CogSense Authors.
# Licensed under the Apache License, Version 2.0.
"""RL with latent rationales: GFlowNet fine-tuning of the rationale policy.

Adapted from LaCoT ("Latent Chain-of-Thought for Visual Reasoning", Sun et al.,
NeurIPS 2025, https://github.com/heliossun/LaCoT), with the CogSense reward of
arXiv:2602.01541 (Sec. 4.4):

    p*(Z | X, y)  ~  exp(R(Z; X, y))
    R(tau_t)      =  alpha * log q_theta0(y | X, z_1:t)  +  gamma * R_lvip(tau_t)
    R_lvip        =  -|| g_psi(mean(H_opt(X, Z; theta))) - Enc_vis(V_y) ||_2^2

    q_theta    the rationale policy: the SFT model with LoRA adapters (trained)
    q_theta0   the frozen scorer: the same model with the adapters disabled,
               i.e. exactly the SFT model
    g_psi      the frozen LVIP head of the SFT model

A rationale Z = (z_1, ..., z_n, T) is the reasoning part of the SFT target. The
termination action T is the "\\n\\n" token that separates the reasoning from the
answer y ("So, the final answer is: <answer>X</answer>"). A sampled token that
ends with "\\n\\n" (e.g. ".\\n\\n") ends the rationale: its text is kept and T is
appended. A sample that reaches rat_max_len is terminated there.

One training step, for one question X per device:
  1. Rationales: the reference rationale Z_ref (from the SFT data) and m samples
     Z_1..Z_m ~ q_theta(. | X): explore_nums samples in groups of explore_min_bs, at
     temperature 1, or with probability 1/4 at a temperature in
     [pf_temp_low, pf_temp_high] (as in LaCoT).
  2. Token-wise marginal reward estimation: R(tau_t) is computed with the scorer at
     the anchors t = 0, lambda, 2*lambda, ... < n and at t = n, and linearly
     interpolated in between.
  3. Reference-guided filtering: Z_i is kept if R(Z_i) >= R(Z_ref) + log(delta_s),
     with delta_s scheduled linearly from delta_start to delta_end over
     delta_horizon optimizer steps.
  4. Loss: sum_i 1[Z_i kept] * L_SubTB(Z_i), with the sub-trajectory balance loss
     (Madan et al., 2023) on the prefix scores in the "modified SubTB" form used by
     LaCoT (Hu et al., 2024): log F(s_t) = R(tau_t) - log q_theta(T | s_t).

R_lvip depends on Z through the backbone conditioned on [X; Z]. With causal
attention the option-image tokens (part of X) do not attend to Z, so H_opt, and
therefore R_lvip, is the same for every prefix z_1:t. It is read from the policy
forward pass over [X; Z].
"""

import contextlib
import inspect
import math
import random
from collections import defaultdict
from typing import Dict, List

import torch
from transformers import GenerationConfig, Trainer
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

# Later transformers 5.x releases take mm_token_type_ids for M-RoPE.
_NEEDS_MM_TOKEN_TYPES = "mm_token_type_ids" in inspect.signature(Qwen3VLModel.forward).parameters


def _base_model(model):
    """The CogSense model inside a (PEFT-wrapped) model."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


@contextlib.contextmanager
def _eval_mode(model):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


@contextlib.contextmanager
def _generation_config(model, config):
    """Generate with exactly `config`: passing a GenerationConfig to generate() would
    fill its default-valued fields (temperature, top_k, top_p) from the model's own
    generation config."""
    base = _base_model(model)
    saved = base.generation_config
    model.generation_config = config
    base.generation_config = config
    try:
        yield
    finally:
        base.generation_config = saved
        if model is not base:
            model.generation_config = saved


def _all_reduce_max(value: int, device) -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        t = torch.tensor([value], device=device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        return int(t.item())
    return value


def anchor_indices(n: int, stride: int) -> List[int]:
    """Anchors t = 0, stride, 2*stride, ... < n, plus the full rationale t = n."""
    return sorted(set(range(0, n, stride)) | {n})


def interpolate_anchors(values: Dict[int, float], n: int) -> torch.Tensor:
    """R(tau_t) for t = 0..n by linear interpolation between the anchor values."""
    anchors = sorted(values)
    out = torch.empty(n + 1, dtype=torch.float32)
    for a, b in zip(anchors[:-1], anchors[1:]):
        steps = torch.arange(0, b - a + 1, dtype=torch.float32)
        out[a : b + 1] = values[a] + steps / (b - a) * (values[b] - values[a])
    out[anchors[-1]] = values[anchors[-1]]
    return out


def subtb_loss(log_rewards: torch.Tensor, log_pf: torch.Tensor, log_term: torch.Tensor, lam: float) -> torch.Tensor:
    """Modified sub-trajectory balance loss of one trajectory s_0 -> ... -> s_n -> T.

    For every pair i < j of states:
        log R(tau_i) + sum_{i <= t < j} log q(z_t+1 | s_t) + log q(T | s_j)
            = log R(tau_j) + log q(T | s_i),
    weighted by lam ** (j - i - 1) and averaged.

    Args:
        log_rewards: [n+1] R(tau_t) for the prefixes s_0..s_n.
        log_pf:      [n]   log q_theta(z_t+1 | s_t).
        log_term:    [n+1] log q_theta(T | s_t).
        lam:         sub-trajectory weight.
    """
    n = log_pf.shape[0]
    if n == 0:
        return log_pf.sum() * 0.0 + log_term.sum() * 0.0
    log_flow = log_rewards.to(log_term) - log_term
    delta = log_flow[:-1] + log_pf - log_flow[1:]
    cum = torch.cat([delta.new_zeros(1), delta.cumsum(0)])
    diff = cum.unsqueeze(0) - cum.unsqueeze(1)  # diff[i, j] = sum_{i <= t < j} delta_t
    idx = torch.arange(n + 1, device=cum.device)
    length = idx.unsqueeze(0) - idx.unsqueeze(1)  # j - i
    weight = torch.where(length > 0, lam ** (length - 1).clamp(min=0).to(cum.dtype), 0.0)
    return (weight * diff.pow(2)).sum() / weight.sum()


class GFNTrainer(Trainer):
    """Trainer for RL with latent rationales. Needs a PEFT (LoRA) model, the
    Qwen3-VL processor as `processing_class`, GFNArguments as `gfn_args` and
    the batches of qwenvl/data/data_gfn.py."""

    def __init__(self, *args, gfn_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gfn_args = gfn_args
        tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)
        self.pad_id = tokenizer.pad_token_id
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        stop = tokenizer("\n\n", add_special_tokens=False)["input_ids"]
        if len(stop) != 1:
            raise ValueError("expected '\\n\\n' to be a single token")
        self.stop_id = stop[0]
        # A sampled rationale ends at any token whose text ends with "\n\n", or at <|im_end|>.
        self.boundary_ids = sorted(
            {i for i in range(len(tokenizer)) if tokenizer.decode([i]).endswith("\n\n")} | {self.im_end_id}
        )
        self.boundary_prefix_ids = {}  # boundary token -> tokens of its text without the newlines
        for i in self.boundary_ids:
            text = "" if i == self.im_end_id else tokenizer.decode([i]).rstrip("\n")
            self.boundary_prefix_ids[i] = tokenizer(text, add_special_tokens=False)["input_ids"] if text else []
        self._metrics = defaultdict(list)

    # ------------------------------------------------------------------
    # Schedules (in optimizer steps)
    # ------------------------------------------------------------------
    def _linear(self, start, end, horizon):
        frac = min(1.0, self.state.global_step / max(1, horizon))
        return start + (end - start) * frac

    def log_delta(self) -> float:
        a = self.gfn_args
        return math.log(self._linear(a.delta_start, a.delta_end, a.delta_horizon))

    def reward_temperature(self) -> float:
        a = self.gfn_args
        return self._linear(a.reward_temp_start, a.reward_temp_end, a.reward_temp_horizon)

    def _mm_kwargs(self, input_ids):
        if not _NEEDS_MM_TOKEN_TYPES:
            return {}
        image_token_id = _base_model(self.accelerator.unwrap_model(self.model)).config.image_token_id
        return {"mm_token_type_ids": (input_ids == image_token_id).long()}

    # ------------------------------------------------------------------
    # 1. Rationales
    # ------------------------------------------------------------------
    def canonical_rationale(self, tokens: List[int]) -> List[int]:
        """The rationale tokens z_1..z_n of a sampled continuation: everything before
        the first boundary token, plus the text of that token without its newlines."""
        out = []
        for tok in tokens:
            if tok == self.pad_id:
                break
            if tok in self.boundary_prefix_ids:
                out += self.boundary_prefix_ids[tok]
                break
            out.append(tok)
        return out[: self.gfn_args.rat_max_len]

    @torch.no_grad()
    def sample_rationales(self, model, inputs) -> List[List[int]]:
        a = self.gfn_args
        temperature = 1.0
        if random.randint(0, 3) == 1:  # LaCoT: tempered sampling with probability 1/4
            temperature = random.uniform(a.pf_temp_low, a.pf_temp_high)
        config = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            max_new_tokens=a.rat_max_len,
            min_new_tokens=min(a.rat_min_len, a.rat_max_len),
            eos_token_id=self.boundary_ids,
            pad_token_id=self.pad_id,
            suppress_tokens=[self.pad_id],
            use_cache=True,
        )
        prompt = inputs["input_ids"]
        bs = a.explore_min_bs
        prompts = prompt.repeat(bs, 1)
        rationales = []
        with _eval_mode(model), _generation_config(model, config):
            for _ in range(a.explore_nums // bs):
                out = model.generate(
                    input_ids=prompts,
                    attention_mask=torch.ones_like(prompts),
                    pixel_values=inputs["pixel_values"].repeat(bs, 1),
                    image_grid_thw=inputs["image_grid_thw"].repeat(bs, 1),
                    **self._mm_kwargs(prompts),
                )
                for row in out[:, prompt.shape[1] :].tolist():
                    rationales.append(self.canonical_rationale(row))
        return rationales

    # ------------------------------------------------------------------
    # Policy terms (with gradient) and R_lvip
    # ------------------------------------------------------------------
    def policy_terms(self, model, inputs, rationales: List[List[int]]):
        """log q_theta(z_t+1 | s_t) [n], log q_theta(T | s_t) [n+1] and R_lvip for every
        rationale, from one forward pass of the policy over [X; Z] (right padded)."""
        device = inputs["input_ids"].device
        prompt = inputs["input_ids"][0]
        num = len(rationales)
        max_z = max(len(z) for z in rationales)
        input_ids = torch.full((num, prompt.shape[0] + max_z), self.pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(input_ids)
        for r, z in enumerate(rationales):
            seq = torch.cat([prompt, torch.tensor(z, dtype=torch.long, device=device)])
            input_ids[r, : seq.shape[0]] = seq
            attention_mask[r, : seq.shape[0]] = 1

        base = _base_model(self.accelerator.unwrap_model(model))
        use_lvip = base.lvip_head is not None and self.gfn_args.reward_gamma != 0
        lvip_kwargs = {}
        if use_lvip:
            num_images = inputs["image_grid_thw"].shape[0]
            target = inputs["lvip_target_image_index"].view(-1)[0]
            offsets = num_images * torch.arange(num, device=device)
            lvip_kwargs = dict(
                lvip_option_image_mask=inputs["lvip_option_image_mask"].repeat(num),
                lvip_target_image_index=torch.where(target >= 0, target + offsets, -1),
                lvip_image_sample_index=torch.arange(num, device=device).repeat_interleave(num_images),
            )
        # logits[:, t] is the next-token distribution at state s_t (prefix z_1:t).
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=inputs["pixel_values"].repeat(num, 1),
            image_grid_thw=inputs["image_grid_thw"].repeat(num, 1),
            logits_to_keep=max_z + 1,
            use_cache=False,
            **self._mm_kwargs(input_ids),
            **lvip_kwargs,
        ).logits

        log_pf, log_term = [], []
        for r, z in enumerate(rationales):
            n = len(z)
            row = logits[r, : n + 1].float()
            lse = row.logsumexp(-1)
            log_term.append(row[:, self.stop_id] - lse)
            tokens = torch.tensor(z, dtype=torch.long, device=device)
            log_pf.append(row[:n].gather(-1, tokens[:, None]).squeeze(-1) - lse[:n])

        r_lvip = torch.zeros(num, device=device)
        if use_lvip:
            prediction, h_y, valid = base._last_lvip_outputs
            r_lvip = -(prediction.float() - h_y.float()).pow(2).sum(-1) * valid.float()
            base._last_lvip_outputs = None
        return log_pf, log_term, r_lvip.detach()

    # ------------------------------------------------------------------
    # 2. Prefix rewards with the frozen scorer
    # ------------------------------------------------------------------
    @torch.no_grad()
    def answer_log_likelihoods(self, model, inputs, prefixes: List[List[int]]) -> List[float]:
        """log q_theta0(y | X, z_1:t, T) for every prefix z_1:t (adapters disabled)."""
        device = inputs["input_ids"].device
        prompt = inputs["input_ids"][0]
        answer = inputs["answer_ids"]
        bs = self.gfn_args.reward_batch_size
        num_batches = (len(prefixes) + bs - 1) // bs
        # Under DeepSpeed ZeRO-3 every rank must run the same number of forward passes.
        total_batches = _all_reduce_max(num_batches, device)
        stop = torch.tensor([self.stop_id], dtype=torch.long, device=device)
        scores = []
        disable = model.disable_adapter() if hasattr(model, "disable_adapter") else contextlib.nullcontext()
        with disable, _eval_mode(model):
            for b in range(total_batches):
                chunk = prefixes[b * bs : (b + 1) * bs] if b < num_batches else prefixes[:1]
                seqs = [torch.cat([prompt, torch.tensor(z, dtype=torch.long, device=device), stop, answer]) for z in chunk]
                length = max(s.shape[0] for s in seqs)
                # Left padding: the answer is at the end of every row.
                input_ids = torch.full((len(seqs), length), self.pad_id, dtype=torch.long, device=device)
                attention_mask = torch.zeros_like(input_ids)
                for i, s in enumerate(seqs):
                    input_ids[i, length - s.shape[0] :] = s
                    attention_mask[i, length - s.shape[0] :] = 1
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=inputs["pixel_values"].repeat(len(seqs), 1),
                    image_grid_thw=inputs["image_grid_thw"].repeat(len(seqs), 1),
                    logits_to_keep=answer.shape[0] + 1,
                    use_cache=False,
                    **self._mm_kwargs(input_ids),
                ).logits[:, :-1].float()
                logp = logits.log_softmax(-1).gather(-1, answer.view(1, -1, 1).expand(len(seqs), -1, 1)).squeeze(-1)
                if b < num_batches:
                    scores += logp.sum(-1).tolist()
        return scores

    def prefix_log_rewards(self, model, inputs, rationales, r_lvip) -> List[torch.Tensor]:
        """R(tau_t), t = 0..n, of every rationale: scorer at the anchors, interpolated."""
        a = self.gfn_args
        jobs = [(r, t) for r, z in enumerate(rationales) for t in anchor_indices(len(z), a.reward_anchor_stride)]
        scores = self.answer_log_likelihoods(model, inputs, [rationales[r][:t] for r, t in jobs])
        temperature = self.reward_temperature()
        anchors = defaultdict(dict)
        for (r, t), s in zip(jobs, scores):
            anchors[r][t] = (a.reward_alpha * s + a.reward_gamma * float(r_lvip[r])) / temperature
        return [interpolate_anchors(anchors[r], len(z)) for r, z in enumerate(rationales)]

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        a = self.gfn_args
        unwrapped = self.accelerator.unwrap_model(model)
        reference = inputs["ref_rationale_ids"].tolist()[: a.rat_max_len]
        rationales = [reference] + self.sample_rationales(unwrapped, inputs)

        log_pf, log_term, r_lvip = self.policy_terms(model, inputs, rationales)
        log_rewards = self.prefix_log_rewards(unwrapped, inputs, rationales, r_lvip)

        # 3. Reference-guided filtering on the full-rationale scores R(Z).
        final = torch.tensor([lr[-1].item() for lr in log_rewards])
        accept = final[1:] >= final[0] + self.log_delta()

        # 4. sum_i 1[Z_i kept] * L_SubTB(Z_i) over the sampled rationales. The zero terms
        # keep every rationale in the graph (same backward on every rank).
        loss = sum(lp.sum() * 0.0 for lp in log_pf) + sum(lt.sum() * 0.0 for lt in log_term)
        for i in range(1, len(rationales)):
            if accept[i - 1]:
                loss = loss + subtb_loss(log_rewards[i], log_pf[i], log_term[i], a.subtb_lambda)

        self._metrics["gfn_loss"].append(float(loss.detach()))
        self._metrics["accept_rate"].append(float(accept.float().mean()))
        self._metrics["reward_ref"].append(float(final[0]))
        self._metrics["reward_sampled"].append(float(final[1:].mean()))
        self._metrics["r_lvip"].append(float(r_lvip.mean()))
        self._metrics["rationale_len"].append(sum(len(z) for z in rationales[1:]) / max(1, len(rationales) - 1))
        return (loss, None) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if self._metrics and "loss" in logs:
            for key, values in self._metrics.items():
                stats = torch.tensor([sum(values), len(values)], dtype=torch.float64, device=self.args.device)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(stats)
                logs[key] = round(float(stats[0] / stats[1]), 6)
            self._metrics.clear()
        super().log(logs, *args, **kwargs)
