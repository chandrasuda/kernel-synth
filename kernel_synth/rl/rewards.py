"""Reward function for the kernel-engineering RL task.

The reward maps a single benchmark result -> a scalar in roughly ``[-0.2, 1.5]``.
The scale is calibrated so:

    0.0  ->  matched eager PyTorch
    1.0  ->  matched torch.compile
    >1   ->  beat torch.compile
    <0   ->  slower than eager (small penalty)
    -0.1 ->  numerically incorrect output (sparse failure signal)

The components dict carries every input + every derived ratio so it's easy to
attribute the reward later.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


class RewardComponents(TypedDict, total=False):
    """Fully-attributed breakdown of a reward computation.

    Every field that ``compute_reward`` may populate is enumerated here so
    downstream consumers (the SPA, RL post-training) can type-check access
    instead of stringly-typed dict lookups. Optional fields are ``None``
    when the corresponding timing is missing or non-finite.
    """

    correct: bool
    eager_ms: float | None
    compile_ms: float | None
    solution_ms: float | None
    progress: float | None
    eager_speedup: float | None
    compile_ratio: float | None
    wrong_penalty: float
    progress_clip: list[float]
    compile_denominator_degenerate: bool
    benchmark: dict[str, Any]  # set by KernelEnv.finalize, not compute_reward


class RewardResult(TypedDict):
    """Return shape of :func:`compute_reward`."""

    reward: float
    components: RewardComponents


WRONG_PENALTY = -0.1
PROGRESS_MIN = -0.2
PROGRESS_MAX = 1.5

# If torch.compile is within this relative margin of eager, the denominator
# (eager - compile) is degenerate and the progress ratio is meaningless.
# Below this threshold we fall back to the eager-only denominator and tag
# the components dict so downstream consumers (SPA, RL postprocessing)
# can flag the rollout instead of treating it as a real score.
COMPILE_DEGENERATE_REL = 0.02


@dataclass
class RewardBreakdown:
    reward: float
    components: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"reward": self.reward, "components": self.components}


def _finite(x: float | None) -> float | None:
    """Coerce ``x`` to a finite float or ``None``."""
    if x is None:
        return None
    try:
        fx = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fx):
        return None
    return fx


def compute_reward(
    *,
    correct: bool,
    eager_ms: float | None,
    compile_ms: float | None,
    solution_ms: float | None,
) -> RewardResult:
    """Compute the rollout reward + a fully-attributed components dict.

    Robustness:
        * Any non-finite or non-positive timing is treated as missing.
        * If the solution is incorrect, the reward is ``WRONG_PENALTY``.
        * If ``compile_ms`` is missing or doesn't beat eager, the denominator
          collapses to a small epsilon, so any speedup vs. eager still gives
          a healthy positive reward, capped at ``PROGRESS_MAX``.

    Returns
    -------
    ``{"reward": float, "components": {...}}``
    """
    eager = _finite(eager_ms)
    compile_ = _finite(compile_ms)
    soln = _finite(solution_ms)

    # Guard against zero/negative timings before any ratio.
    eager_pos = eager if (eager is not None and eager > 0.0) else None
    compile_pos = compile_ if (compile_ is not None and compile_ > 0.0) else None
    soln_pos = soln if (soln is not None and soln > 0.0) else None

    eager_speedup: float | None = None
    if eager_pos is not None and soln_pos is not None:
        eager_speedup = eager_pos / soln_pos

    compile_ratio: float | None = None
    if compile_pos is not None and soln_pos is not None:
        compile_ratio = compile_pos / soln_pos

    progress: float | None = None
    degenerate = False
    if eager_pos is not None and soln_pos is not None:
        # If torch.compile didn't beat eager (or is too close to it), the
        # standard denominator collapses; fall back to the eager-only one and
        # warn so callers know the resulting ratio is approximate.
        if compile_pos is not None and (eager_pos - compile_pos) > 1e-6:
            relative_gap = (eager_pos - compile_pos) / max(eager_pos, 1e-6)
            if relative_gap < COMPILE_DEGENERATE_REL:
                degenerate = True
                denom = max(eager_pos, 1e-6)
                logger.warning(
                    "rewards: degenerate denominator — eager_ms=%.4f and "
                    "compile_ms=%.4f are within %.1f%% of each other; "
                    "falling back to eager-only progress.",
                    eager_pos,
                    compile_pos,
                    COMPILE_DEGENERATE_REL * 100.0,
                )
            else:
                denom = eager_pos - compile_pos
        else:
            denom = max(eager_pos, 1e-6)
            if compile_pos is not None:
                degenerate = True
                logger.warning(
                    "rewards: torch.compile is no faster than eager "
                    "(eager=%.4f, compile=%.4f); using eager-only progress.",
                    eager_pos,
                    compile_pos,
                )
        progress = (eager_pos - soln_pos) / denom

    if not correct:
        reward = WRONG_PENALTY
    elif progress is None:
        # Correct but un-timeable: treat as eager-equivalent.
        reward = 0.0
    else:
        reward = max(PROGRESS_MIN, min(PROGRESS_MAX, progress))

    components: RewardComponents = {
        "correct": bool(correct),
        "eager_ms": eager,
        "compile_ms": compile_,
        "solution_ms": soln,
        "progress": progress,
        "eager_speedup": eager_speedup,
        "compile_ratio": compile_ratio,
        "wrong_penalty": WRONG_PENALTY,
        "progress_clip": [PROGRESS_MIN, PROGRESS_MAX],
        "compile_denominator_degenerate": degenerate,
    }
    return {"reward": float(reward), "components": components}


__all__ = [
    "RewardBreakdown",
    "RewardComponents",
    "RewardResult",
    "compute_reward",
    "WRONG_PENALTY",
    "PROGRESS_MIN",
    "PROGRESS_MAX",
]
