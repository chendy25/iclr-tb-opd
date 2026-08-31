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
        Number of alternative tokens expanded into full branch continuations at the
        fork position. With rollout.n = 1 + k, slot 0 is the main trajectory and
        slots 1..k are the branches.
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
        Shared by both fork units: a floor on the *raw* uncertainty of a fork
        candidate, applied before any normalization. Gating a normalized score is
        impossible -- rank and min-max both send the best candidate to 1.0 -- so this
        is the only knob that can actually suppress forking on a trajectory the
        student was confident throughout. For ``fork_unit="turn"`` the threshold is
        in the units of whichever estimator ran (truncated entropy, or the mean-NLL
        proxy when the rollout carried no top-k); ``tb_opd_fork_estimator`` reports
        which one. Original (token-path) meaning follows:
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

    -- Turn-level branching (TB-OPD-Turn, ``fork_unit="turn"``) --

    fork_unit (str):
        Granularity of the fork. ``token`` (default,母方法 TB-OPD-Token) forks at
        a single response position on the flat token sequence and continues with a
        single generate. ``turn`` (TB-OPD-Turn) forks at the *start of a
        high-uncertainty assistant turn* in a multi-turn tool-use trajectory and
        re-enters the full tool loop for the remaining turns (E4 breakpoint
        resume). Turn forking requires ``agent_name=tool_agent`` and per-token
        ``response_logprobs`` (set ``actor_rollout_ref.rollout.calculate_log_probs=True``).
    turn_first_k (int):
        Number of leading assistant tokens of a turn used to compute its
        uncertainty signal (ARPO measures the entropy spike over the first tokens
        after a tool response). <=0 uses the whole turn.
    turn_only_post_tool (bool):
        If True, only turns that immediately follow a tool response are eligible
        fork points (ARPO's "post-tool decision turn"). Default False: the token
        path imposes no such positional prior, and whether the post-tool prior helps
        is one of the things the fork experiments are meant to measure. Superseded
        by ``fork_eligibility`` when that is set explicitly.
    turn_skip_first (int):
        Number of leading assistant turns to exclude from forking. Default 0, to
        match the token path's ``fork_skip_first=1`` (which skips a single *token*,
        not a whole turn's worth of candidates).
    max_branches_per_traj (int):
        Budget B: maximum number of turns forked per trajectory. Phase 1' uses 1.
    consecutive_high_entropy_penalty (bool):
        AEPO-style guard: down-weight a turn's fork signal when the immediately
        preceding turn was also high-signal, to avoid over-branching on a run of
        consecutive high-entropy turns.
    consecutive_penalty_weight (float):
        Multiplicative penalty applied to a turn signal when the preceding turn is
        also above the median signal (only when the penalty is enabled).

    -- Shared fork scoring axes --

    Both fork units rank candidates by ``Fuse(norm(U), norm(D))``: ``U`` is student
    uncertainty, ``D`` is teacher-student disagreement. The axes below expose that
    functional so the two granularities are one operator with different candidate
    sets, rather than two separately-tuned heuristics. Each defaults to ``None`` =
    "use whatever ``fork_metric`` implies", so existing arms are unchanged.

    fork_eligibility (str | None):
        Which candidates exist for ``fork_unit="turn"``: ``post_tool`` (ARPO's
        decision turn), ``turn_open`` (any turn's first token), ``reasoning``
        (inside the thinking span), ``action`` (inside the tool-call / env action
        span), or ``all`` (no positional prior, which is what the token path does).
        ``reasoning``/``action`` require the agent loop to report per-turn action
        spans; without them those candidates are skipped. ``None`` derives
        ``post_tool``/``all`` from ``turn_only_post_tool``.
    fork_alpha (float):
        Weight on ``U`` in ``fuse="blend"``: ``alpha*U + (1-alpha)*D``. ``1.0``
        (default, as on the token path) is pure uncertainty -- where the student is
        *unsure*. ``0.0`` is pure teacher disagreement -- where it is *wrong*
        relative to the teacher, i.e. where the OPD signal lives. Below 1.0 the fork
        depends on the teacher, so the main trajectory's teacher forward is issued
        before fork selection (needed for the loss regardless, so it costs nothing).
    fork_fuse (str):
        ``blend`` (default) averages the two ranks, so a candidate must do well on
        both. ``max`` keeps the larger rank, so either extreme survives. ``union``
        keeps the top half by each signal separately and ranks that union, which is
        the mode where both kinds of fork actually get chosen. ``soft_or`` is ATOD's
        ``1-(1-U)(1-D)``. Everything except ``blend`` with ``alpha=1`` needs a
        teacher signal.
    fork_kl_window (int):
        Forward window over which the signed disagreement is averaged. A single
        token's ``D`` is too noisy to rank on; the fork is worth taking where the
        *upcoming* span diverges.
    fork_normalize (str):
        Per-trajectory normalization before fusing: ``rank`` (default, matching the
        token path -- robust to the two signals living on different scales) or
        ``minmax`` (ATOD's choice, which keeps Soft-OR probability-like).
    disagreement_signed (bool):
        ``True`` scores ``logp_student - logp_teacher``, isolating the "student is
        over-confident here" direction. ``False`` uses ATOD's unsigned ``|delta|``,
        which merges over-confidence with needless timidity.
    fork_min_turn_gap (int):
        Minimum turn distance between two selected fork points when
        ``max_branches_per_traj > 1``, so the budget spreads over the trajectory.
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

    # Turn-level branching (TB-OPD-Turn).
    fork_unit: str = "token"
    turn_first_k: int = 16
    turn_only_post_tool: bool = False
    turn_skip_first: int = 0
    max_branches_per_traj: int = 1
    consecutive_high_entropy_penalty: bool = False
    consecutive_penalty_weight: float = 0.5

    # Shared fork scoring axes (both fork units). ``None`` = take the fork_metric
    # preset, so existing arms keep their behavior unless an axis is set explicitly.
    fork_eligibility: Optional[str] = None
    fork_alpha: float = 1.0
    fork_fuse: str = "blend"
    fork_kl_window: int = 128
    fork_normalize: str = "rank"
    disagreement_signed: bool = True
    fork_min_turn_gap: int = 1

    # -- Turn-level loss reweighting (B-A1, ATOD T-DUR style; no branch expansion) --
    # Independent of ``enable``: emphasize the KD token loss on high-uncertainty
    # assistant turns *without* forking new sub-trajectories. This is the loss-side
    # "reweight-only" ablation opposite the "expand" method M. It reuses the same
    # per-turn signal (entropy proxy) computed at loss time from ``old_log_probs``
    # and ``response_mask``, so it needs neither the rollout branch path nor
    # teacher-at-rollout disagreement.
    turn_reweight: bool = False
    reweight_alpha: float = 1.0
    reweight_metric: str = "ent"  # ent | dHtool  (entropy-only; no disagreement)

    # Fork metrics valid per granularity.
    _TOKEN_METRICS = ("entropy", "topk_gap")
    # ``entropy`` is the token path's spelling of ``ent``; accepting both means the
    # shared ``fork_metric`` default is valid whichever fork unit is selected.
    _TURN_METRICS = ("ent", "entropy", "dHtool")
    _REWEIGHT_METRICS = ("ent", "dHtool")
    _ELIGIBILITY = ("post_tool", "turn_open", "reasoning", "action", "all")
    _FUSE = ("blend", "max", "union", "soft_or")
    # Removed metric values and the axes that now express them. Each conflated the
    # uncertainty statistic with the fusion, which is why fork_metric=hybrid could
    # silently drop the disagreement term when no teacher was wired up.
    _RETIRED_TURN_METRICS = {
        "disagree": "fork_metric=ent fork_alpha=0.0",
        "hybrid": "fork_metric=dHtool fork_alpha=0.5 fork_fuse=soft_or fork_normalize=minmax",
    }

    def __post_init__(self):
        # Loss-side reweighting is validated independently of the rollout switch
        # (B-A1 runs with enable=False).
        if self.turn_reweight:
            if self.reweight_metric not in self._REWEIGHT_METRICS:
                raise ValueError(
                    f"tb_opd.reweight_metric must be one of {self._REWEIGHT_METRICS}, got {self.reweight_metric}"
                )
            if self.turn_skip_first < 0:
                raise ValueError(f"tb_opd.turn_skip_first must be >= 0, got {self.turn_skip_first}")
        if not self.enable:
            return
        if self.k < 1:
            raise ValueError(f"tb_opd.k must be >= 1, got {self.k}")
        if self.fork_unit not in ("token", "turn"):
            raise ValueError(f"tb_opd.fork_unit must be 'token' or 'turn', got {self.fork_unit}")
        if self.branch_mode not in ("forced_topk", "resample"):
            raise ValueError(
                f"tb_opd.branch_mode must be 'forced_topk' or 'resample', got {self.branch_mode}"
            )
        if self.fork_unit == "turn":
            if self.fork_metric in self._RETIRED_TURN_METRICS:
                raise ValueError(
                    f"tb_opd.fork_metric='{self.fork_metric}' was retired: fork_metric now names only "
                    f"the uncertainty statistic (as on the token path), and fusion is set independently. "
                    f"Use {self._RETIRED_TURN_METRICS[self.fork_metric]} instead."
                )
            if self.fork_metric not in self._TURN_METRICS:
                raise ValueError(
                    f"tb_opd.fork_metric for fork_unit='turn' must be one of {self._TURN_METRICS}, "
                    f"got {self.fork_metric}"
                )
            if self.max_branches_per_traj < 1:
                raise ValueError(
                    f"tb_opd.max_branches_per_traj must be >= 1, got {self.max_branches_per_traj}"
                )
            if self.turn_skip_first < 0:
                raise ValueError(f"tb_opd.turn_skip_first must be >= 0, got {self.turn_skip_first}")
            if self.fork_eligibility is not None and self.fork_eligibility not in self._ELIGIBILITY:
                raise ValueError(
                    f"tb_opd.fork_eligibility must be one of {self._ELIGIBILITY}, got {self.fork_eligibility}"
                )
            if self.fork_min_turn_gap < 0:
                raise ValueError(f"tb_opd.fork_min_turn_gap must be >= 0, got {self.fork_min_turn_gap}")
        else:
            if self.fork_metric not in self._TOKEN_METRICS:
                raise ValueError(
                    f"tb_opd.fork_metric for fork_unit='token' must be one of {self._TOKEN_METRICS}, "
                    f"got {self.fork_metric}"
                )
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
        if self.fork_fuse not in self._FUSE:
            raise ValueError(f"tb_opd.fork_fuse must be one of {self._FUSE}, got {self.fork_fuse}")
        if not 0.0 <= self.fork_alpha <= 1.0:
            raise ValueError(f"tb_opd.fork_alpha must be in [0, 1], got {self.fork_alpha}")
        if self.fork_normalize not in ("minmax", "rank"):
            raise ValueError(f"tb_opd.fork_normalize must be 'minmax' or 'rank', got {self.fork_normalize}")
        if self.fork_kl_window < 1:
            raise ValueError(f"tb_opd.fork_kl_window must be >= 1, got {self.fork_kl_window}")


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
