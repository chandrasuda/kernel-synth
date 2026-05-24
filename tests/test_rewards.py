"""Reward function — correctness gate + clamp + degenerate-denominator path."""

from __future__ import annotations

import pytest

from kernel_synth.rl.rewards import (
    PROGRESS_MAX,
    PROGRESS_MIN,
    WRONG_PENALTY,
    compute_reward,
)


def test_incorrect_solution_returns_wrong_penalty() -> None:
    r = compute_reward(correct=False, eager_ms=10.0, compile_ms=5.0, solution_ms=4.0)
    assert r["reward"] == WRONG_PENALTY
    assert r["components"]["correct"] is False


def test_matching_eager_yields_zero_reward() -> None:
    r = compute_reward(correct=True, eager_ms=10.0, compile_ms=5.0, solution_ms=10.0)
    assert r["reward"] == pytest.approx(0.0)


def test_matching_compile_yields_one_reward() -> None:
    r = compute_reward(correct=True, eager_ms=10.0, compile_ms=5.0, solution_ms=5.0)
    assert r["reward"] == pytest.approx(1.0)


def test_reward_is_clamped_to_progress_max() -> None:
    # Solution is 10x faster than compile — clip should kick in at PROGRESS_MAX.
    r = compute_reward(correct=True, eager_ms=10.0, compile_ms=5.0, solution_ms=0.01)
    assert r["reward"] == pytest.approx(PROGRESS_MAX)


def test_reward_is_clamped_to_progress_min_when_slower_than_eager() -> None:
    # Solution slower than eager — should hit PROGRESS_MIN.
    r = compute_reward(correct=True, eager_ms=10.0, compile_ms=5.0, solution_ms=100.0)
    assert r["reward"] == pytest.approx(PROGRESS_MIN)


def test_missing_compile_falls_back_to_eager_denominator() -> None:
    r = compute_reward(correct=True, eager_ms=10.0, compile_ms=None, solution_ms=5.0)
    assert r["reward"] == pytest.approx(0.5)


def test_degenerate_denominator_is_flagged() -> None:
    # compile within ~0.2% of eager -> denominator collapses, flag set.
    r = compute_reward(
        correct=True, eager_ms=1.000, compile_ms=0.998, solution_ms=0.500
    )
    assert r["components"]["compile_denominator_degenerate"] is True
    # Reward is meaningful but reduced (uses eager-only denominator).
    assert 0.0 < r["reward"] <= 1.0


def test_components_carry_ratios_and_inputs() -> None:
    r = compute_reward(correct=True, eager_ms=2.0, compile_ms=1.0, solution_ms=0.5)
    c = r["components"]
    assert c["eager_ms"] == 2.0
    assert c["compile_ms"] == 1.0
    assert c["solution_ms"] == 0.5
    assert c["eager_speedup"] == pytest.approx(4.0)
    assert c["compile_ratio"] == pytest.approx(2.0)


def test_non_finite_timings_are_treated_as_missing() -> None:
    r = compute_reward(
        correct=True, eager_ms=float("nan"), compile_ms=None, solution_ms=1.0
    )
    # Cannot compute progress -> defaults to 0.0.
    assert r["reward"] == 0.0
    assert r["components"]["progress"] is None
