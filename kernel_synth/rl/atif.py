"""Agent Trajectory Interchange Format (ATIF) v1.7.

Subset of the spec we need for kernel-engineering rollouts. Includes the
validators that matter for downstream RL post-training:
    * sequential ``step_id`` starting at 1
    * agent-only fields refused on system/user steps
    * tool_call → observation.results.source_call_id correlation
    * ISO-8601 timestamps

Spec: https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "ATIF-v1.7"
ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+\-]\d{2}:\d{2})$"
)


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    function_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] | None = None


class ObservationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_call_id: str | None = None
    content: str | None = None
    subagent_trajectory_ref: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[ObservationResult] = Field(default_factory=list)


class Metrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    extra: dict[str, Any] | None = None  # RL fields land here (reward, etc.)


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: int = Field(..., ge=1)
    timestamp: str | None = None
    source: Literal["system", "user", "agent"]
    model_name: str | None = None
    reasoning_effort: str | float | None = None
    message: str | list[dict[str, Any]] = ""
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    observation: Observation | None = None
    metrics: Metrics | None = None
    extra: dict[str, Any] | None = None
    llm_call_count: int | None = None

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not ISO_RE.match(v):
            raise ValueError(f"timestamp not ISO 8601: {v!r}")
        return v

    @model_validator(mode="after")
    def _agent_only_fields(self) -> "Step":
        if self.source != "agent":
            if self.model_name is not None:
                raise ValueError("model_name only allowed on source='agent'")
            if self.reasoning_content is not None:
                raise ValueError(
                    "reasoning_content only allowed on source='agent'"
                )
            if self.tool_calls:
                raise ValueError("tool_calls only allowed on source='agent'")
            if self.metrics is not None:
                raise ValueError("metrics only allowed on source='agent'")
            if self.reasoning_effort is not None:
                raise ValueError("reasoning_effort only allowed on source='agent'")
        return self


class FinalMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_cached_tokens: int | None = None
    total_cost_usd: float | None = None
    total_steps: int | None = None
    extra: dict[str, Any] | None = None


class Agent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    model_name: str | None = None
    tool_definitions: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None


class Trajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    session_id: str | None = None
    trajectory_id: str | None = None
    agent: Agent
    steps: list[Step] = Field(default_factory=list)
    notes: str | None = None
    final_metrics: FinalMetrics | None = None
    continued_trajectory_ref: str | None = None
    extra: dict[str, Any] | None = None
    subagent_trajectories: list["Trajectory"] | None = None

    @model_validator(mode="after")
    def _validate_steps(self) -> "Trajectory":
        for i, step in enumerate(self.steps, start=1):
            if step.step_id != i:
                raise ValueError(
                    f"step_ids must be sequential starting at 1; "
                    f"got {step.step_id} at index {i - 1}"
                )

        # Tool calls must be answered by observation.results entries within
        # the SAME step (the spec correlates via source_call_id).
        for step in self.steps:
            if not step.tool_calls:
                continue
            call_ids = {tc.tool_call_id for tc in step.tool_calls}
            if step.observation is None:
                continue
            for r in step.observation.results:
                if r.source_call_id is not None and r.source_call_id not in call_ids:
                    raise ValueError(
                        f"observation result references unknown tool_call_id"
                        f" {r.source_call_id!r} in step {step.step_id}"
                    )
        return self

    def to_json_dict(self, *, exclude_none: bool = True) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=exclude_none)


Trajectory.model_rebuild()  # support self-referential subagent_trajectories
