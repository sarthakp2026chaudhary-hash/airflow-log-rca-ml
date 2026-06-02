"""Tests for the failure-mode taxonomy."""

from __future__ import annotations

import math

from log_rca.datagen.failure_modes import (
    FAILURE_MODES,
    RETRY_PLACEHOLDER,
    by_code,
    weights,
)


def test_failure_modes_have_distinct_codes():
    codes = [m.code for m in FAILURE_MODES]
    assert len(codes) == len(set(codes)), "failure-mode codes must be unique"


def test_failure_weights_sum_to_one():
    total = sum(weights())
    assert math.isclose(total, 1.0, abs_tol=0.001), f"weights sum to {total}, not 1.0"


def test_each_mode_has_at_least_one_error_line():
    for m in FAILURE_MODES:
        assert m.error_lines, f"{m.code} has no error_lines"


def test_by_code_finds_known_mode():
    m = by_code("OOM")
    assert m is not None
    assert m.code == "OOM"


def test_by_code_returns_none_for_unknown():
    assert by_code("DOES_NOT_EXIST") is None


def test_retry_placeholder_is_not_in_main_taxonomy():
    """The retry placeholder must not pollute the real taxonomy."""
    assert RETRY_PLACEHOLDER not in FAILURE_MODES
    assert by_code("__retry") is None
