# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig
from verl.utils.config import omega_conf_to_dataclass

from .rollout import RolloutConfig

__all__ = ["DistillationLossConfig", "DistillationTeacherModelConfig", "DistillationConfig", "TBOPDConfig"]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class DistillationLossConfig(BaseConfig):
    """Configuration for distillation loss settings.

    loss_mode (str):
        Distillation loss function to use.
    topk (int, optional):
        Number of top tokens to consider for top-k distillation losses.
    use_task_rewards (bool):
        Whether to include task rewards alongside distillation loss.
    distillation_loss_coef (float):
        Coefficient for distillation loss when combined with task rewards.
    loss_max_clamp (float, optional):
        Maximum value to clamp distillation loss. If None, no clamping is applied.
    log_prob_min_clamp (float, optional):
        Minimum value to clamp log probabilities for stability, e.g., log q - log p where p or q are
        very close to zero. If None, no clamping is applied.
    use_policy_gradient (bool):
        Whether to incorporate distillation loss as a reward, as done
        by https://thinkingmachines.ai/blog/on-policy-distillation/. Recommended to use loss_mode=k1.
        Otherwise, distillation loss is directly backpropagated as a supervised loss,
        as in https://arxiv.org/abs/2306.13649. Recommended to use loss_mode=k3 or forward_kl_topk.
    policy_loss_mode (str):
        Name of the policy loss to use when use_policy_gradient is true.
    clip_ratio (float):
        PPO clipping ratio for policy loss.
    clip_ratio_low (float):
        Lower bound for PPO clipping ratio.
    clip_ratio_high (float):
        Upper bound for PPO clipping ratio.
    loss_settings (DistillationLossSettings, optional):
        Runtime-populated settings based on loss_mode. Not set by user.
    """

    loss_mode: str = "k3"
    topk: Optional[int] = 128
    use_task_rewards: bool = True
    distillation_loss_coef: float = 1.0
    loss_max_clamp: Optional[float] = 10.0
    log_prob_min_clamp: Optional[float] = -10.0

    # Chunked top-K log-probs (opt-in, avoids [B, T, V] log_softmax buffer
    # at long context). Only consumed by ``loss_mode='forward_kl_topk'``.
    # Default ``False`` to preserve short-context performance (chunked path
    # has ~6x time overhead at N=14K, V=152K). Set ``True`` when hitting OOM
    # at long context (>=64K tokens, V=152K) where the baseline path OOMs.
    use_chunked_topk: bool = False
    # Tokens per chunk along (B*T) when ``use_chunked_topk=True``. Larger
    # chunks reduce kernel-launch overhead but increase per-chunk memory;
    # smaller chunks reduce per-chunk memory but increase kernel-launch
    # overhead (saved-tensor total stays constant either way).
    chunked_topk_chunk_size: int = 4096

    use_policy_gradient: bool = True
    policy_loss_mode: str = "vanilla"
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2

    # ---- DAPO-style soft overlong punishment (advantage shaping) ----
    # In reward-free OPD (use_task_rewards=False) there is no reward channel, so the
    # length penalty is applied directly to the token-level advantage in the policy
    # gradient path (see verl/trainer/distillation/losses.py), analogous to rep_penalty.
    # The penalty is a function of the *effective* (post-mask) response length: with
    # mask_after_answer on, post-answer refrain is already zeroed, so this term guards
    # the remaining length (pre-answer bloat / no-answer ramble) rather than double-
    # charging a correct prefix for refrain the mask already dropped.
    # penalty(L) = clamp((L - (max_len - buffer)) / buffer, 0, 1) * factor, subtracted
    # per (unmasked) token, so overlong sequences get a uniform negative advantage.
    overlong_enable: bool = False
    # Length (in response tokens) of the soft ramp before the hard cap.
    overlong_buffer_len: int = 4096
    # Peak penalty magnitude at/above the cap. DAPO uses 1.0; OPD advantages are ~O(0.1),
    # so a smaller value (e.g. 0.5) is usually a safer first try.
    overlong_penalty_factor: float = 1.0
    # Hard cap = max response length. If None, inferred from response_mask.shape[-1]
    # (padded response length of the micro-batch); set explicitly to the configured
    # max_response_length to keep the threshold batch-independent.
    overlong_max_len: Optional[int] = None

    # ---- Learn-EOS auxiliary loss ----
    # Weight (lambda) of the auxiliary cross-entropy that teaches the model to emit EOS
    # right after a final answer. Active only when the rollout produced an ``eos_sft_mask``
    # (see rollout.learn_eos_after_answer): for the marked EOS position(s) we add
    # ``learn_eos_coef * (-log p_student(EOS | prefix))`` to the loss. The mask is already
    # correctness-gated at rollout time (learn_eos_require_correct), so this only fires on
    # the EOS token, never on the reasoning tokens. 0.0 disables. OPD token advantages are
    # ~O(0.1); a small value (e.g. 0.1-0.5) keeps the stop signal from dominating.
    learn_eos_coef: float = 0.0

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    # Store distillation loss settings for computing the specified loss_mode
    # Not set by user, populated at runtime
    loss_settings: Optional[dict] = None

    def __post_init__(self):
        self._mutable_fields.add("loss_settings")
        from verl.trainer.distillation.losses import DistillationLossSettings, get_distillation_loss_settings

        self.loss_settings: DistillationLossSettings = get_distillation_loss_settings(self.loss_mode)

        if self.policy_loss_mode != "vanilla":
            raise NotImplementedError(
                f"Only vanilla policy loss is currently supported when use_policy_gradient is True, "
                f"but got {self.policy_loss_mode}."
            )

        if self.use_policy_gradient and self.loss_mode == "forward_kl_topk":
            print(
                "WARNING: forward_kl_topk is most effective as a supervised distillation loss "
                "(use_policy_gradient=False). With policy gradient, the update uses only the sampled"
                " token's logprob ∇logπ(a), so the top-k distributional signal (how non-sampled logits "
                "should move) is largely unused."
            )

        if not self.use_policy_gradient and self.loss_mode == "k1":
            raise ValueError(
                "Directly backpropagating k1 loss is incorrect since gradient of k1 loss"
                " wrt model weights does not depend on teacher log probabilities."
            )


@dataclass
class DistillationTeacherModelConfig(BaseConfig):
    """Configuration for on-policy distillation teacher.

    key (str, optional):
        Identifier to route examples to the teacher model in multi-teacher setting.
    model_path (str, optional):
        Model path for the teacher model. Can be a local path or a Hugging Face model
    inference (RolloutConfig):
        Rollout configuration for the teacher model inference during distillation.
    num_replicas (int):
        Number of inference replicas of this teacher to launch. Each replica occupies
        `per_replica_world_size` GPUs (= inference.data_parallel_size *
        inference.tensor_model_parallel_size * inference.pipeline_model_parallel_size),
        so the teacher's total GPU footprint is
        `num_replicas * per_replica_world_size`.
    """

    _mutable_fields = BaseConfig._mutable_fields | {"num_replicas", "key"}

    key: Optional[str] = None
    model_path: Optional[str] = None
    inference: RolloutConfig = field(default_factory=RolloutConfig)
    num_replicas: Optional[int] = 0

    @property
    def per_replica_world_size(self) -> int:
        return (
            self.inference.tensor_model_parallel_size
            * self.inference.data_parallel_size
            * self.inference.pipeline_model_parallel_size
        )

    @property
    def world_size(self) -> int:
        return self.num_replicas * self.per_replica_world_size

    def check_configured(self):
        if self.model_path is None:
            raise ValueError("model_path must be specified for distillation teacher model config.")
        if self.key is None:
            raise ValueError("key must be specified for distillation teacher model config.")
        if self.num_replicas is None:
            raise ValueError("num_replicas must be specified for distillation teacher model config.")

    def validate_and_prepare_for_distillation(self, use_topk: bool, topk: Optional[int]) -> None:
        # Prompt + Response from student are fed into teacher as context
        max_model_len = self.inference.max_model_len
        student_prompt_length = self.inference.prompt_length
        student_response_length = self.inference.response_length
        required_context_len = student_prompt_length + student_response_length + 1
        if max_model_len is not None and required_context_len > max_model_len:
            raise ValueError(
                "Distillation teacher inference requires room for the student prompt, the full student "
                f"response, and one generated token, but got {student_prompt_length=}, "
                f"{student_response_length=}, {required_context_len=}, {max_model_len=}."
            )
        self.inference.prompt_length = self.inference.prompt_length + self.inference.response_length
        self.inference.response_length = 1
        self._validate_topk_logprobs(use_topk=use_topk, topk=topk)

    def _validate_topk_logprobs(self, use_topk: bool, topk: Optional[int]) -> None:
        if not use_topk:
            return
        if topk is None:
            raise ValueError("topk must be specified when use_topk is True.")

        engine_name = self.inference.name
        engine_kwargs = self.inference.engine_kwargs
        match engine_name:
            case "vllm":
                vllm_engine_kwargs = dict(engine_kwargs.get("vllm", {}))
                max_logprobs = vllm_engine_kwargs.get("max_logprobs")
                if max_logprobs is None:
                    vllm_engine_kwargs["max_logprobs"] = topk
                    max_logprobs = topk
                if max_logprobs < topk:
                    raise ValueError(
                        f"VLLM max_logprobs ({max_logprobs}) must be >= distillation_loss topk "
                        f"({topk}) to enable distillation loss computation."
                    )
                engine_kwargs["vllm"] = vllm_engine_kwargs
            case "sglang":
                # SGLang's top_logprobs_num is a per-request parameter, so there is no
                # engine-boot cap to align (unlike vLLM's max_logprobs). The async
                # server translates sampling_params["prompt_logprobs"] into
                # return_logprob + logprob_start_len=0 + top_logprobs_num at call time.
                pass
            case _:
                raise NotImplementedError(
                    f"DistillationTeacherModelConfig does not support inference engine {engine_name}"
                )


@dataclass
class TBOPDConfig(BaseConfig):
    """Configuration for token-level branching on-policy distillation (TB-OPD).

    Default method M*: Only-fail + B=1 + k=2 full expand. Disabled by default so
    that standard OPD (B1) is completely unaffected.

    enable (bool):
        Master switch. When False, rollout falls back to standard per-row generation.
    k (int):
        Number of alternative tokens expanded into full branch continuations at each
        fork position. With ``rollout.n = 1 + num_forks * k``, slot 0 is the main
        trajectory and slot ``1 + b*k + j`` is the j-th alternative at fork ``b``.
        See ``num_forks`` on why spending budget on B beats spending it on k.
    only_fail (bool):
        If True, only branch when the main trajectory is judged incorrect by the
        rule-based reward; otherwise the extra slots are filled with plain rollouts.
    fork_metric (str):
        How to score fork uncertainty at each response position: "entropy"
        (truncated top-k entropy) or "topk_gap" (top1-top2 logprob margin).
    topk_logprobs (int):
        Number of top-k prompt logprobs requested in the second student forward
        used to locate the fork and read candidate tokens.
    branch_min_tokens (int):
        Minimum remaining response budget required at a fork position; positions
        too close to the end are skipped.
    correct_threshold (float):
        Reward score >= threshold counts the main trajectory as correct.
    fork_select (str):
        Fork-position selection among filtered candidates: "argmax" (highest
        uncertainty; deterministic) or "topk_uniform" (uniform-random within the
        top-``fork_topk_positions`` most uncertain positions; CURE-aligned).
    fork_topk_positions (int):
        Number of highest-uncertainty positions kept before uniform sampling when
        ``fork_select="topk_uniform"``. Ignored for "argmax".
    fork_skip_first (int):
        Number of leading response positions to exclude from forking (CURE skips
        position 0; default 1). Guards against branching on opener boilerplate.
    fork_min_token_strip_len (int):
        A candidate position is skipped unless its decoded token, stripped of
        whitespace, is strictly longer than this (CURE uses >1). Filters
        punctuation / single-char / whitespace high-entropy noise.
    fork_min_entropy (float):
        Minimum fork uncertainty (only applied when ``fork_metric="entropy"``);
        if the best filtered position scores below this, no fork is emitted and
        the extra slots degrade to plain rollouts. 0.0 disables the gate.
    fork_dedup_main (bool):
        If True, exclude the token actually sampled by the main trajectory at the
        fork position from the expanded candidate set, so every branch is a
        genuine alternative rather than a copy of the main path.
    fork_token_filter (str):
        Eligibility filter for fork positions. "math_aware" (default) keeps
        math-bearing single-char tokens (digits/operators) and reflection words,
        dropping only special/whitespace/pure-punctuation tokens -- the entropy
        metric then selects. "strip_len" is the legacy CURE-style filter that
        drops tokens whose stripped text is not longer than
        ``fork_min_token_strip_len`` (which discards single-char math tokens).
    scheme_b (bool):
        If True, read fork candidates from the main rollout's own per-token top-k
        logprobs (requested via ``logprobs=topk_logprobs`` during generation)
        instead of running a second student forward. Saves one full-sequence
        prefill per branched prompt; the win grows with response length. Entropy
        is then measured on the actual decoding distribution (temperature ``T``,
        ``processed_logprobs``), matching the Beyond-80/20 forking-token
        definition. Falls back to the second forward if top-k is unavailable.
    scheme_b_validate (bool):
        Validation-only. When ``scheme_b`` is on, also run the legacy second
        forward and log ``tb_opd_schemeb_pos_match`` (fork-position agreement) so
        the temperature / ``logprobs_mode`` consistency can be checked before
        trusting Scheme B. Doubles the selection cost; keep off for real runs.
    branch_mode (str):
        How branch continuations are generated after fork selection.
        ``forced_topk`` (default, M*): force each top-k candidate token at the
        fork position then continue-generate. ``resample`` (B4 / CURE-style):
        continue-generate from the shared prefix without forcing a token, using
        stochastic sampling (``resample_temperature`` or rollout temperature).
    resample_temperature (float):
        Sampling temperature for ``branch_mode="resample"``. Values < 0 reuse the
        rollout temperature from ``actor_rollout_ref.rollout``.
    shape_branches (bool):
        If True, apply the same post-generation answer shaping to TB-OPD branch
        continuations as the main rollout gets in ``SingleTurnAgentLoop.run`` --
        i.e. ``mask_truncated_no_answer`` / ``mask_after_answer`` / learn-EOS
        (``eos_sft_mask``) and the ``rep_penalty`` / ``format_penalty`` columns.
        When False (default) branches keep a full all-ones ``response_mask`` and no
        EOS/penalty columns, so mask/EOS only constrain the main trajectory.
    dedup_shared_prefix (bool):
        Every branch replays the main trajectory's tokens before the fork, so with
        ``rollout.n = 1 + k`` that shared prefix is forward/backward-ed and enters the
        loss ``1 + k`` times -- a silent ``(1 + k)x`` up-weighting of early tokens (and
        of the learn-EOS supervision, whenever the fork lands past the final answer).
        When True (default) the prefix is masked out of every *branch* row, so it is
        trained exactly once via the main slot and each branch only supervises the
        forced token onward. Under ``branch_weight_mode="rb"`` the main slot's prefix
        also drops back to weight 1.0 (the RB weights sum to ``n``, so leaving it at
        the main slot's weight would still over-count it). False restores the legacy
        duplicated-prefix behaviour.
    fork_respect_mask (bool):
        Restrict fork candidates to positions that are still in-loss on the main
        trajectory, i.e. where ``response_mask == 1``. Without this the fork can land
        inside the post-answer span that ``mask_after_answer`` / learn-EOS already
        zeroed: the branch then re-derives the *same* answer from the shared prefix,
        everything it generates after the fork is masked away, and its injected EOS
        duplicates the main trajectory's at the same index. Default True; False
        restores the legacy behaviour of scanning the whole response.
    num_forks (int):
        Number of fork *positions* opened per trajectory (``B``). Each fork expands
        into ``k`` forced alternatives, so ``rollout.n`` should be ``1 + B * k``.
        ``1`` (default) reproduces the single-fork method exactly.

        Raising ``B`` is a strictly better use of a given rollout budget than raising
        ``k``. RB weights renormalize to sum ``n`` *within one fork*, so at ``B=1``
        the extra slots compete for a fixed pool: at ``k=6`` the rank-5 and rank-6
        alternatives land near ``0.1`` while the main trajectory inflates past ``3``,
        i.e. a full-length rollout is paid for ~1% of the gradient mass. Splitting the
        same budget across ``B`` distinct fork positions gives every branch its own
        normalization, so all of them carry usable weight.
    fork_min_gap (int):
        Minimum token distance between two fork positions when ``num_forks > 1``.
        Without it the top-``B`` ranked positions are typically adjacent, and the
        branches degenerate into near-copies of each other. Ignored when ``B == 1``.
    fork_alpha (float):
        Blend between the two fork-ranking signals:
        ``alpha * rank(entropy) + (1 - alpha) * rank(teacher_disagreement)``.
        ``1.0`` (default) is pure entropy -- where the student is *unsure*. ``0.0`` is
        pure teacher disagreement -- where the student is *wrong* relative to the
        teacher, i.e. where the OPD signal actually lives. Values below 1.0 require
        the teacher, so the main trajectory's teacher forward is issued before fork
        selection (it is needed for the loss regardless, so this costs nothing extra).
    fork_kl_window (int):
        Window (in tokens) over which teacher/student disagreement is averaged when
        ranking fork positions. A single token's log-ratio is far too noisy; the fork
        should land where the *upcoming* span diverges. Ignored when ``fork_alpha=1``.
    fork_fuse (str):
        How entropy and teacher-disagreement combine after the entropy gate.
        ``"blend"`` (default) averages the two ranks -- a position must do well on
        both to stay in the top-k. ``"max"`` keeps whichever rank is larger, so
        either extreme survives. ``"union"`` takes the top half by entropy and the
        top half by disagreement, then samples from that union -- this is the mode
        that lets *both* kinds of fork actually get chosen. Requires a teacher
        signal (same as ``fork_alpha < 1``).
    branch_weight_mode (str):
        How the ``k+1`` continuations of a fork are weighted in the loss.
        ``"off"`` (default) keeps the legacy uniform weighting.         ``"rb"`` applies
        Rao-Blackwellized weights ``pi_theta(a_j) / sum_i pi_theta(a_i)`` (rescaled to
        mean 1), so a forced alternative contributes in proportion to how likely the
        student was to emit it. Only applies to ``branch_mode="forced_topk"``;
        ``"resample"`` branches are already drawn from ``pi_theta`` and stay uniform.

        The estimator necessarily differs between ``num_forks == 1`` and ``> 1``:

        - ``B == 1``: the exact conditional expectation. Normalization runs over
          ``{main} + k`` alternatives, so the main trajectory's post-fork span carries
          its own ``pi_theta(a_0)`` weight.
        - ``B > 1``: the main trajectory sits in ``B`` fork groups at once and a single
          scalar per token cannot carry ``B`` weights. It is therefore pinned at ``1.0``
          -- the same "count it once" accounting as ``dedup_shared_prefix`` -- and each
          fork's ``k`` forced branches are normalized among themselves to total mass
          ``k``. The weights still sum to ``1 + B * k = n``, so the mean stays 1 and the
          effective learning rate is unchanged.
    branch_weight_temp (float):
        Temperature on the RB weights. ``1.0`` is the exact estimator; larger values
        interpolate towards uniform.
    branch_weight_floor (float):
        Minimum per-slot weight (pre-renormalization) so a very unlikely branch still
        contributes something after we paid to generate it. ``0.0`` = exact RB.
    """

    enable: bool = False
    k: int = 2
    only_fail: bool = True
    fork_metric: str = "entropy"
    topk_logprobs: int = 20
    branch_min_tokens: int = 8
    correct_threshold: float = 1.0
    fork_select: str = "argmax"
    fork_topk_positions: int = 20
    fork_skip_first: int = 1
    fork_min_token_strip_len: int = 1
    fork_min_entropy: float = 0.0
    fork_dedup_main: bool = True
    fork_token_filter: str = "math_aware"
    scheme_b: bool = False
    scheme_b_validate: bool = False
    branch_mode: str = "forced_topk"
    resample_temperature: float = -1.0
    shape_branches: bool = False
    dedup_shared_prefix: bool = True
    fork_respect_mask: bool = True
    num_forks: int = 1
    fork_min_gap: int = 16
    fork_alpha: float = 1.0
    fork_kl_window: int = 128
    fork_fuse: str = "blend"
    branch_weight_mode: str = "off"
    branch_weight_temp: float = 1.0
    branch_weight_floor: float = 0.0

    def __post_init__(self):
        if not self.enable:
            return
        if self.k < 1:
            raise ValueError(f"tb_opd.k must be >= 1, got {self.k}")
        if self.branch_mode not in ("forced_topk", "resample"):
            raise ValueError(
                f"tb_opd.branch_mode must be 'forced_topk' or 'resample', got {self.branch_mode}"
            )
        if self.fork_metric not in ("entropy", "topk_gap"):
            raise ValueError(f"tb_opd.fork_metric must be 'entropy' or 'topk_gap', got {self.fork_metric}")
        if self.topk_logprobs < 2:
            raise ValueError(f"tb_opd.topk_logprobs must be >= 2, got {self.topk_logprobs}")
        if self.fork_select not in ("argmax", "topk_uniform"):
            raise ValueError(f"tb_opd.fork_select must be 'argmax' or 'topk_uniform', got {self.fork_select}")
        if self.fork_token_filter not in ("math_aware", "strip_len"):
            raise ValueError(
                f"tb_opd.fork_token_filter must be 'math_aware' or 'strip_len', got {self.fork_token_filter}"
            )
        if self.fork_topk_positions < 1:
            raise ValueError(f"tb_opd.fork_topk_positions must be >= 1, got {self.fork_topk_positions}")
        if self.fork_skip_first < 0:
            raise ValueError(f"tb_opd.fork_skip_first must be >= 0, got {self.fork_skip_first}")
        if self.num_forks < 1:
            raise ValueError(f"tb_opd.num_forks must be >= 1, got {self.num_forks}")
        if self.fork_min_gap < 0:
            raise ValueError(f"tb_opd.fork_min_gap must be >= 0, got {self.fork_min_gap}")
        if not 0.0 <= self.fork_alpha <= 1.0:
            raise ValueError(f"tb_opd.fork_alpha must be in [0, 1], got {self.fork_alpha}")
        if self.fork_kl_window < 1:
            raise ValueError(f"tb_opd.fork_kl_window must be >= 1, got {self.fork_kl_window}")
        if self.fork_fuse not in ("blend", "max", "union"):
            raise ValueError(f"tb_opd.fork_fuse must be 'blend', 'max', or 'union', got {self.fork_fuse}")
        if self.branch_weight_mode not in ("off", "rb"):
            raise ValueError(f"tb_opd.branch_weight_mode must be 'off' or 'rb', got {self.branch_weight_mode}")
        if self.branch_weight_temp <= 0.0:
            raise ValueError(f"tb_opd.branch_weight_temp must be > 0, got {self.branch_weight_temp}")
        if not 0.0 <= self.branch_weight_floor < 1.0:
            raise ValueError(f"tb_opd.branch_weight_floor must be in [0, 1), got {self.branch_weight_floor}")


@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for on-policy distillation.

    enabled (bool):
        Whether on-policy distillation is enabled.
    n_gpus_per_node (int):
        Number of GPUs per node in the teacher resource pool.
    nnodes (int):
        Number of nodes in the teacher resource pool.
    teacher_models (dict[str, TeacherModelConfig]):
        Configurations for teacher models used for multi-teacher distillation.
    teacher_key (str):
        Key to route examples to the appropriate teacher model in multi-teacher setups. Should correspond to a field in
        the data proto, e.g., data_source.
    distillation_loss (DistillationLossConfig):
    Configuration for distillation loss settings.

    NOTE: The `teacher_model` entry is in the `teacher_models` dict by default.
    Since it is popped when other teacher entries are added, using `teacher_model` as
    one of several keys silently drops it. For example, the following CLI overrides result
    in ONLY `teacher_model2` being used:

    ```bash
    distillation.teacher_models.teacher_model.key=openai/gsm8k
    distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    Instead, give the first teacher a different name:

    ```bash
    +distillation.teacher_models.teacher_model1.key=openai/gsm8k
    +distillation.teacher_models.teacher_model1.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    """

    _mutable_fields = BaseConfig._mutable_fields | {"teacher_models", "n_gpus_per_node", "nnodes"}

    enabled: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0
    teacher_models: dict[str, DistillationTeacherModelConfig] = field(default_factory=dict)
    teacher_key: str = "data_source"
    distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)
    tb_opd: TBOPDConfig = field(default_factory=TBOPDConfig)

    def __post_init__(self):
        if not self.enabled:
            return

        self.teacher_models = self._resolve_teacher_models()
        teacher_world_size_sum = 0
        for teacher_model in self.teacher_models.values():
            teacher_model.validate_and_prepare_for_distillation(
                use_topk=self.distillation_loss.loss_settings.use_topk,
                topk=self.distillation_loss.topk,
            )
            teacher_world_size_sum += teacher_model.world_size
        total_pool_size = self.n_gpus_per_node * self.nnodes
        if teacher_world_size_sum != total_pool_size:
            raise ValueError(
                f"Sum of teacher (num_replicas * per_replica_world_size) ({teacher_world_size_sum}) must match "
                f"the distillation resource pool size "
                f"({self.n_gpus_per_node=} * {self.nnodes=} = {total_pool_size})."
            )

    def _resolve_teacher_models(self) -> dict[str, DistillationTeacherModelConfig]:
        assert "teacher_model" in self.teacher_models
        if len(self.teacher_models) == 1:
            # Single teacher occupies the entire teacher resource pool.
            teacher_model = self.teacher_models["teacher_model"]
            inference = teacher_model.inference
            per_replica = (
                inference.tensor_model_parallel_size
                * inference.data_parallel_size
                * inference.pipeline_model_parallel_size
            )
            pool_size = self.n_gpus_per_node * self.nnodes
            if pool_size % per_replica != 0:
                raise ValueError(
                    f"Single teacher's per_replica_world_size ({per_replica}) must divide the distillation "
                    f"resource pool size ({self.n_gpus_per_node=} * {self.nnodes=} = {pool_size})."
                )
            teacher_model.num_replicas = pool_size // per_replica
            teacher_model.key = "default"
        else:
            # Multiple teachers: remove default single teacher config
            self.teacher_models.pop("teacher_model")

        # Teacher models dict is keyed by teacher_key instead of YAML entry name
        teacher_models = {}
        for teacher_config in self.teacher_models.values():
            teacher_config = omega_conf_to_dataclass(teacher_config, dataclass_type=DistillationTeacherModelConfig)
            teacher_config.check_configured()
            if teacher_config.key in teacher_models:
                raise ValueError(f"Duplicate teacher key {teacher_config.key} found in teacher models.")
            teacher_models[teacher_config.key] = teacher_config
        return teacher_models
