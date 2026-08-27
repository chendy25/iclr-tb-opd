# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    if response_mask.is_nested:
        distillation_losses_response = distillation_losses[response_mask.bool().to_padded_tensor(False)]
    else:
        distillation_losses_response = distillation_losses[response_mask.bool()]
    # A micro-batch may be fully masked (e.g. every sequence dropped by
    # mask_truncated_no_answer), leaving an empty selection. min()/max() over an
    # empty tensor raises, so fall back to aggregation-neutral sentinels (+inf for
    # MIN, -inf for MAX); any non-empty micro-batch in the global batch dominates.
    if distillation_losses_response.numel() == 0:
        loss_min = distillation_losses.new_tensor(float("inf"))
        loss_max = distillation_losses.new_tensor(float("-inf"))
    else:
        loss_min = distillation_losses_response.min()
        loss_max = distillation_losses_response.max()
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, loss_min),
        "distillation/loss_max": Metric(AggregationType.MAX, loss_max),
    }


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    match config.strategy:
        # VeOmni uses FSDP2 internally, so its loss computation is identical to FSDP.
        case "fsdp" | "veomni":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
        case "megatron":
            import verl.trainer.distillation.megatron.losses as megatron_losses

            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
    )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # Called as final policy loss
    distillation_loss_config = distillation_config.distillation_loss
    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
    if not distillation_loss_config.use_task_rewards and not distillation_loss_config.use_policy_gradient:
        # no need to compute policy loss
        policy_loss = 0.0
        policy_metrics = {}
    else:
        policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
        if not distillation_loss_config.use_task_rewards:
            policy_loss = 0.0

    # Combine distillation with policy loss
    policy_metrics.update(distill_metrics)
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
    )
    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)

    return policy_loss, policy_metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode

    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    if loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = data["old_log_probs"].to_padded_tensor(0.0)
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        rollout_is_weights = data.get("rollout_is_weights", None)
        # Token-level advantage in OPD is the negative distillation loss. Optionally
        # subtract a per-token repetition penalty (computed on rollout token ids, see
        # single_turn_agent_loop / utils.compute_repetition_penalty) so degenerate
        # repetition spans get a negative advantage (a gradient "wall" at the entry
        # token) instead of being reinforced.
        advantages = -distillation_losses.detach()
        # Penalty scale: repetition / overlong penalties are expressed as *multiples of
        # the current advantage scale* rather than absolute values. The OPD token
        # advantage is the raw per-token KL, whose magnitude shrinks ~5-6x over training
        # as the student approaches the teacher; a fixed absolute penalty would therefore
        # silently ramp up (from ~1x to ~7x the signal) over a run. Scaling by the batch
        # mean |advantage| keeps each penalty a constant fraction of the signal. The
        # advantage itself is left un-normalized so OPD's absolute "-KL as reward"
        # semantics are preserved (no whitening / de-meaning).
        valid_mask = response_mask.bool()
        adv_scale = advantages[valid_mask].abs().mean().detach() if valid_mask.any() else advantages.new_tensor(0.0)
        adv_scale = adv_scale.clamp(min=1e-6)
        distillation_metrics["distillation/adv_scale"] = adv_scale.item()
        # Fraction of sequences fully dropped from the loss (all-zero response mask),
        # e.g. by mask_truncated_no_answer (no-answer wall-hits). Monitors method D.
        dropped_seq = valid_mask.sum(dim=-1) == 0
        distillation_metrics["distillation/dropped_seq_frac"] = dropped_seq.float().mean().item()
        rep_penalty = data.get("rep_penalty", None)
        if rep_penalty is not None:
            if rep_penalty.is_nested:
                rep_penalty = rep_penalty.to_padded_tensor(0.0)
            rep_penalty = rep_penalty.to(device=advantages.device, dtype=advantages.dtype)
            if rep_penalty.shape == advantages.shape:
                # Relative penalty: the per-token weights are multiples of adv_scale.
                rep_penalty = rep_penalty * adv_scale
                advantages = advantages - rep_penalty
                pen_mask = (rep_penalty > 0) & valid_mask
                num_resp = valid_mask.sum().clamp(min=1)
                distillation_metrics["distillation/rep_penalty_token_frac"] = (
                    pen_mask.sum().float() / num_resp
                ).item()
                # Effective (post-scaling) penalty magnitude actually subtracted.
                distillation_metrics["distillation/rep_penalty_mean"] = (
                    rep_penalty[pen_mask].mean().item() if pen_mask.any() else 0.0
                )
        # Format-shaping penalty: rollouts with no answer-shaped final answer (no valid
        # \boxed{}/Answer: -- see rollout.format_penalty_enable) carry a uniform per-token
        # weight; subtract it (scaled by adv_scale, like rep_penalty) so the whole
        # no-answer trajectory gets a negative advantage and the model learns to emit a
        # box. Distinct from mask_truncated_no_answer, which only *drops* such rollouts
        # (neutral); use one or the other (masked tokens make this a no-op).
        format_penalty = data.get("format_penalty", None)
        if format_penalty is not None:
            if format_penalty.is_nested:
                format_penalty = format_penalty.to_padded_tensor(0.0)
            format_penalty = format_penalty.to(device=advantages.device, dtype=advantages.dtype)
            if format_penalty.shape == advantages.shape:
                format_penalty = format_penalty * adv_scale
                advantages = advantages - format_penalty
                fmt_mask = (format_penalty > 0) & valid_mask
                # Fraction of *sequences* carrying the format penalty (any penalized token).
                distillation_metrics["distillation/format_penalty_seq_frac"] = (
                    (fmt_mask.any(dim=-1).float().mean().item()) if fmt_mask.numel() else 0.0
                )
                distillation_metrics["distillation/format_penalty_mean"] = (
                    format_penalty[fmt_mask].mean().item() if fmt_mask.any() else 0.0
                )
        # DAPO-style soft overlong punishment on the *effective* (post-mask) response
        # length: subtract a per-sequence non-negative penalty (broadcast over tokens)
        # from the advantage so overlong sequences are pushed down. With mask_after_answer
        # on, post-answer refrain is already masked out of response_mask, so eff_len here
        # measures the length that is NOT post-answer refrain (pre-answer bloat / ramble).
        # The penalty is likewise scaled by adv_scale (a multiple of the advantage scale).
        if getattr(loss_config, "overlong_enable", False):
            max_len = loss_config.overlong_max_len or response_mask.shape[-1]
            buffer_len = max(int(loss_config.overlong_buffer_len), 1)
            expected_len = max_len - buffer_len
            eff_len = response_mask.sum(dim=-1).to(advantages.dtype)  # (bsz,)
            over = ((eff_len - expected_len).clamp(min=0.0) / buffer_len).clamp(max=1.0)
            overlong_pen = over * float(loss_config.overlong_penalty_factor) * adv_scale  # (bsz,) >= 0
            advantages = advantages - overlong_pen.unsqueeze(-1)
            over_mask = over > 0
            distillation_metrics["distillation/overlong_seq_frac"] = over_mask.float().mean().item()
            distillation_metrics["distillation/overlong_pen_mean"] = (
                overlong_pen[over_mask].mean().item() if over_mask.any() else 0.0
            )
        # TB-OPD Rao-Blackwell branch weighting: a per-sequence multiplier (broadcast over
        # tokens) reweighting the k+1 continuations of a fork by the student's own
        # probability of the token each one was forced to take. It rides the existing
        # importance-weight hook, so it multiplies the per-token loss but NOT the token
        # count in the denominator; the weights have mean 1 within a group, which keeps
        # the aggregate loss scale (and effective LR) identical to uniform weighting.
        branch_weight = data.get("branch_weight", None)
        if branch_weight is not None:
            if branch_weight.is_nested:
                branch_weight = branch_weight.to_padded_tensor(1.0)
            branch_weight = branch_weight.to(device=advantages.device, dtype=advantages.dtype)
            if branch_weight.shape == advantages.shape:
                rollout_is_weights = (
                    branch_weight if rollout_is_weights is None else rollout_is_weights * branch_weight
                )
                # Weights are per-token, not per-sequence: under dedup_shared_prefix the
                # main slot's pre-fork tokens sit at 1.0 while the fork onward carries the
                # RB weight, so position 0 is no longer representative. Summarize over the
                # tokens that actually reach the loss instead.
                w_mask = response_mask.to(branch_weight.dtype)
                n_in_loss = w_mask.sum()
                if n_in_loss > 0:
                    distillation_metrics["distillation/branch_weight_mean"] = (
                        (branch_weight * w_mask).sum() / n_in_loss
                    ).item()
                    distillation_metrics["distillation/branch_weight_min"] = (
                        branch_weight.masked_fill(w_mask == 0, float("inf")).min().item()
                    )
                    distillation_metrics["distillation/branch_weight_max"] = (
                        branch_weight.masked_fill(w_mask == 0, float("-inf")).max().item()
                    )

        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)

        # Learn-EOS auxiliary loss: teach the model to emit EOS right after a (correct)
        # final answer. ``eos_sft_mask`` marks the supervised EOS position for each
        # eligible rollout (produced + correctness-gated in the agent loop). The EOS
        # positions are already excluded from ``response_mask`` above, so this CE is
        # their only signal.
        #
        # Normalization: mirror the main token-mean loss (see agg_loss) -- normalize the
        # SUM of the per-EOS NLL by the GLOBAL batch size (number of sequences) and
        # compensate FSDP grad-averaging with ``* dp_size``. Two reasons this matters:
        #   1. A plain local ``.mean()`` over the micro-batch EOS count divides by a tiny
        #      number (~1-2 positions), giving each EOS ~100x a normal token's gradient
        #      weight, so a handful of EOS positions dominated the (grad-clipped) update
        #      and starved the OPD objective early in training.
        #   2. That local mean also drifts with the (dynamic) micro-batch count and dp
        #      size, so ``learn_eos_coef`` was not a stable knob.
        # With this global normalization the effective term is ``seq_frac * mean_NLL``:
        # it auto-scales with how many rollouts actually need the stop signal and cannot
        # be dominated by a few positions.
        learn_eos_coef = float(getattr(loss_config, "learn_eos_coef", 0.0) or 0.0)
        eos_sft_mask = data.get("eos_sft_mask", None)
        if learn_eos_coef > 0.0 and eos_sft_mask is not None:
            if eos_sft_mask.is_nested:
                eos_sft_mask = eos_sft_mask.to_padded_tensor(0.0)
            eos_sft_mask = eos_sft_mask.to(device=log_prob.device, dtype=log_prob.dtype)
            if eos_sft_mask.shape == log_prob.shape:
                eos_bool = eos_sft_mask > 0
                num_eos = eos_bool.sum()
                if num_eos > 0:
                    eos_nll = -log_prob[eos_bool]  # per-EOS negative log-likelihood
                    gbs = config.global_batch_info.get("global_batch_size") or log_prob.shape[0]
                    dp = config.global_batch_info.get("dp_size") or 1
                    eos_term = eos_nll.sum() / gbs * dp
                    distillation_loss = distillation_loss + learn_eos_coef * eos_term
                    # raw per-EOS mean NLL (diagnostic: -> 0 as the model learns to stop)
                    distillation_metrics["distillation/learn_eos_ce"] = eos_nll.mean().detach().item()
                    # normalized term actually added (pre-coef); comparable to distillation/loss
                    distillation_metrics["distillation/learn_eos_term"] = eos_term.detach().item()
                    distillation_metrics["distillation/learn_eos_seq_frac"] = (
                        num_eos.float() / log_prob.shape[0]
                    ).item()
                else:
                    distillation_metrics["distillation/learn_eos_seq_frac"] = 0.0
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )

    return distillation_loss, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    overlap_count = model_output.get("overlap_count")
    overlap_token_advantage = model_output.get("overlap_token_advantage")
    if overlap_count is not None and overlap_token_advantage is not None:
        overlap_count = no_padding_2_padding(overlap_count, data)
        overlap_token_advantage = no_padding_2_padding(overlap_token_advantage, data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    overlap_metrics = {}
    if overlap_count is not None and overlap_token_advantage is not None:
        assert overlap_count.shape == overlap_token_advantage.shape == response_mask_bool.shape
        valid_overlap_count = overlap_count[response_mask_bool]
        k = distillation_config.distillation_loss.topk
        assert k is not None
        # Diagnostics for tracking teacher/student top-k overlap in OPD, following
        # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016):
        # overlap ratio and average teacher-token KL contribution on overlapped tokens.
        overlap_metrics["distillation/overlap_ratio"] = (valid_overlap_count.float().mean() / k).item()
        overlap_position_mask = response_mask_bool & (overlap_count > 0)
        if overlap_position_mask.any():
            overlap_metrics["distillation/overlap_token_advantage"] = (
                overlap_token_advantage[overlap_position_mask].mean().item()
            )
        else:
            overlap_metrics["distillation/overlap_token_advantage"] = 0.0

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
        **overlap_metrics,
    }

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, distillation_metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
    )
    # Since k1 can be negative, log the mean absolute loss. Guard the empty case
    # (a fully-masked micro-batch, e.g. all sequences dropped) so .mean() is not NaN.
    _abs_sel = distillation_losses[response_mask_bool].abs()
    metrics = {
        "distillation/abs_loss": Metric(
            AggregationType.MEAN,
            _abs_sel.mean() if _abs_sel.numel() > 0 else distillation_losses.new_tensor(0.0),
        ),
    }
    return distillation_losses, metrics
