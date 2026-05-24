"""RL layer: turn an extracted env folder into a Harbor/ATIF-compatible
kernel-optimization environment.

Public surface:
    KernelEnv         — gym-style env wrapping one env folder
    AgentRollout      — one episode (LLM agent + tools + env)
    compute_reward    — eager <-> torch.compile reward function
    Trajectory        — ATIF v1.7 root model
"""

from .atif import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from .env import KernelEnv
from .agent_loop import AgentRollout, RolloutResult
from .rewards import RewardBreakdown, compute_reward

__all__ = [
    "Agent",
    "AgentRollout",
    "FinalMetrics",
    "KernelEnv",
    "Metrics",
    "Observation",
    "ObservationResult",
    "RewardBreakdown",
    "RolloutResult",
    "Step",
    "ToolCall",
    "Trajectory",
    "compute_reward",
]
