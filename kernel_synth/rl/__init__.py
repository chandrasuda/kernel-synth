"""RL layer: turn an extracted env folder into a Harbor/ATIF-compatible
kernel-optimization environment.

Public surface:
    KernelEnv           — gym-style env wrapping one env folder
    KernelAgentTools    — bundle of sandboxed tools for one env
    rollout             — produce one ATIF trajectory from an env folder
    RolloutResult       — return value of ``rollout()``
    compute_reward      — eager/torch.compile speedup -> scalar reward
    Trajectory / ATIF   — Pydantic models + TrajectoryBuilder + validate
"""

from .agent_loop import AgentRollout, RolloutMode, RolloutResult, rollout
from .atif import (
    Agent,
    AtifAgent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
    TrajectoryBuilder,
    validate,
)
from .env import HistoryEntry, KernelEnv, StepResult
from .prompts import KERNEL_AGENT_SYSTEM_PROMPT, render_user_prompt
from .rewards import RewardBreakdown, RewardComponents, RewardResult, compute_reward
from .tools import KernelAgentTools, ToolError
from .triton_examples import TRITON_SOFTMAX_SOURCE, is_triton_available, triton_softmax

__all__ = [
    "Agent",
    "AgentRollout",
    "AtifAgent",
    "FinalMetrics",
    "HistoryEntry",
    "KERNEL_AGENT_SYSTEM_PROMPT",
    "KernelAgentTools",
    "KernelEnv",
    "Metrics",
    "Observation",
    "ObservationResult",
    "RewardBreakdown",
    "RewardComponents",
    "RewardResult",
    "RolloutMode",
    "RolloutResult",
    "Step",
    "StepResult",
    "TRITON_SOFTMAX_SOURCE",
    "ToolCall",
    "ToolError",
    "Trajectory",
    "TrajectoryBuilder",
    "compute_reward",
    "is_triton_available",
    "render_user_prompt",
    "rollout",
    "triton_softmax",
    "validate",
]
