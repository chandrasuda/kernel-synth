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

import math
from dataclasses import asdict, dataclass
from typing import Any


WRONG_PENALTY = -0.1
PROGRESS_MIN = -0.2
PROGRESS_MAX = 1.5


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
) -> dict[str, Any]:
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
    if eager_pos is not None and soln_pos is not None:
        # If torch.compile didn't beat eager (or is missing), use a tiny
        # epsilon so we still reward speedups against eager.
        if compile_pos is not None and (eager_pos - compile_pos) > 1e-6:
            denom = eager_pos - compile_pos
        else:
            denom = max(eager_pos, 1e-6)
        progress = (eager_pos - soln_pos) / denom

    if not correct:
        reward = WRONG_PENALTY
    elif progress is None:
        # Correct but un-timeable: treat as eager-equivalent.
        reward = 0.0
    else:
        reward = max(PROGRESS_MIN, min(PROGRESS_MAX, progress))

    components: dict[str, Any] = {
        "correct": bool(correct),
        "eager_ms": eager,
        "compile_ms": compile_,
        "solution_ms": soln,
        "progress": progress,
        "eager_speedup": eager_speedup,
        "compile_ratio": compile_ratio,
        "wrong_penalty": WRONG_PENALTY,
        "progress_clip": [PROGRESS_MIN, PROGRESS_MAX],
    }
    return RewardBreakdown(reward=float(reward), components=components).to_dict()


__all__ = ["RewardBreakdown", "compute_reward", "WRONG_PENALTY", "PROGRESS_MIN", "PROGRESS_MAX"]
