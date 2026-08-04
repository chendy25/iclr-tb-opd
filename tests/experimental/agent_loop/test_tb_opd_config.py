"""Unit tests for TB-OPD config and branch-mode semantics."""

from __future__ import annotations

import pytest

from verl.workers.config.distillation import TBOPDConfig


def test_tbopd_default_branch_mode():
    cfg = TBOPDConfig(enable=True)
    assert cfg.branch_mode == "forced_topk"
    assert cfg.resample_temperature == -1.0


def test_tbopd_resample_mode():
    cfg = TBOPDConfig(enable=True, branch_mode="resample", resample_temperature=1.0)
    assert cfg.branch_mode == "resample"
    assert cfg.resample_temperature == 1.0


def test_tbopd_invalid_branch_mode():
    with pytest.raises(ValueError, match="branch_mode"):
        TBOPDConfig(enable=True, branch_mode="invalid")


def test_tbopd_disabled_skips_branch_mode_validation():
    TBOPDConfig(enable=False, branch_mode="invalid")
