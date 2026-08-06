# LiveCodeBench code scorer, ported from SOD (SOD/verl/utils/reward_score/livecodebench).
#
# ``unit_test`` (LiveCodeBench pass@k harness) pulls in the heavy ``lcb_runner``
# runner/eval stack (``datasets``, ``psutil``, model backends). The agentic
# TB-OPD reward path only needs ``code_math.compute_score``'s SandboxFusion-style
# code execution branches, so guard the ``unit_test`` re-export: importing this
# package on a CPU reward worker must not fail when those deps are absent.
try:
    from verl.utils.reward_score.livecodebench.unit_test import lcb_compute_score, prepare_unit_test_data
except Exception:  # noqa: BLE001
    lcb_compute_score = None
    prepare_unit_test_data = None

from verl.utils.reward_score.livecodebench.code_math import compute_score
