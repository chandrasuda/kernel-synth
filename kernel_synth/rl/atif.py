"""Agent Trajectory Interchange Format (ATIF) v1.7.

Subset of the spec we need for kernel-engineering rollouts. Includes the
validators that matter for downstream RL post-training:
    * sequential ``step_id`` starting at 1
    * agent-only fields refused on system/user steps
    * tool_call -> observation.results.source_call_id correlation
    * ISO-8601 timestamps

Spec: https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


SCHEMA_VERSION = "ATIF-v1.7"
ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+\-]\d{2}:\d{2})$"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
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


class AtifAgent(BaseModel):
    """Agent metadata block. Named ``AtifAgent`` per the v1.7 spec."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    model_name: str | None = None
    tool_definitions: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None


# Back-compat alias — the spec calls this ``AtifAgent`` but earlier drafts
# of this file used ``Agent``. Keep both importable.
Agent = AtifAgent


class Trajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    session_id: str | None = None
    trajectory_id: str | None = None
    agent: AtifAgent
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

    def summary(self) -> dict[str, Any]:
        """Return a cheap (n_steps, n_tool_calls, total_tokens, final_reward) view.

        Designed for CLI / dashboard display where you want a one-line
        snapshot of a trajectory without re-walking every step downstream.
        """
        n_tool_calls = 0
        total_tokens = 0
        any_tokens = False
        for s in self.steps:
            if s.tool_calls:
                n_tool_calls += len(s.tool_calls)
            m = s.metrics
            if m is None:
                continue
            if m.prompt_tokens is not None:
                total_tokens += int(m.prompt_tokens)
                any_tokens = True
            if m.completion_tokens is not None:
                total_tokens += int(m.completion_tokens)
                any_tokens = True

        final_reward: float | None = None
        fm = self.final_metrics
        if fm is not None and fm.extra:
            r = fm.extra.get("reward")
            if isinstance(r, (int, float)):
                final_reward = float(r)
        if final_reward is None:
            for s in reversed(self.steps):
                m = s.metrics
                if m is None or not m.extra:
                    continue
                r = m.extra.get("reward")
                if isinstance(r, (int, float)):
                    final_reward = float(r)
                    break

        return {
            "n_steps": len(self.steps),
            "n_tool_calls": n_tool_calls,
            "total_tokens": total_tokens if any_tokens else None,
            "final_reward": final_reward,
        }


Trajectory.model_rebuild()  # support self-referential subagent_trajectories


# ---------------------------------------------------------------------------
# TrajectoryBuilder — incremental construction


class TrajectoryBuilder:
    """Append-only trajectory builder with monotonic ``step_id`` and
    auto-timestamped steps.

    Typical usage::

        b = TrajectoryBuilder(agent=AtifAgent(name="kernel-agent", version="0.1"))
        b.add_step(source="system", message=KERNEL_AGENT_SYSTEM_PROMPT)
        b.add_step(source="user", message=initial_prompt)
        # ... loop ...
        b.add_step(
            source="agent",
            model_name="claude-sonnet-4-5",
            message=assistant_text,
            tool_calls=[ToolCall(...)],
            observation=Observation(results=[ObservationResult(...)]),
        )
        traj = b.build(final_metrics=FinalMetrics(...))
        path.write_text(json.dumps(b.to_json_dict(), indent=2))
    """

    def __init__(
        self,
        *,
        agent: AtifAgent,
        session_id: str | None = None,
        trajectory_id: str | None = None,
        notes: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.agent = agent
        self.session_id = session_id or f"sess-{uuid.uuid4().hex[:12]}"
        self.trajectory_id = trajectory_id or f"traj-{uuid.uuid4().hex[:12]}"
        self.notes = notes
        self.extra = extra
        self._steps: list[Step] = []
        self._final_metrics: FinalMetrics | None = None

    # ---- step API ----

    def add_step(
        self,
        *,
        source: Literal["system", "user", "agent"],
        message: str | list[dict[str, Any]] = "",
        model_name: str | None = None,
        reasoning_effort: str | float | None = None,
        reasoning_content: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        observation: Observation | None = None,
        metrics: Metrics | None = None,
        extra: dict[str, Any] | None = None,
        llm_call_count: int | None = None,
        timestamp: str | None = None,
    ) -> Step:
        step = Step(
            step_id=len(self._steps) + 1,
            timestamp=timestamp or _now_iso(),
            source=source,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            message=message,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            observation=observation,
            metrics=metrics,
            extra=extra,
            llm_call_count=llm_call_count,
        )
        self._steps.append(step)
        return step

    @property
    def steps(self) -> list[Step]:
        return self._steps

    def last_step(self) -> Step | None:
        return self._steps[-1] if self._steps else None

    # ---- finalize ----

    def set_final_metrics(self, fm: FinalMetrics) -> None:
        self._final_metrics = fm

    def aggregate_metrics(self, extra: dict[str, Any] | None = None) -> FinalMetrics:
        """Sum per-step token counts/costs into a fresh ``FinalMetrics``.

        Pass ``extra`` to overlay rollout-level fields (e.g. ``{"reward": ...}``).
        """
        tot_p = tot_c = tot_cached = 0
        tot_cost = 0.0
        any_tokens = any_cost = False
        for s in self._steps:
            m = s.metrics
            if m is None:
                continue
            if m.prompt_tokens is not None:
                tot_p += m.prompt_tokens
                any_tokens = True
            if m.completion_tokens is not None:
                tot_c += m.completion_tokens
                any_tokens = True
            if m.cached_tokens is not None:
                tot_cached += m.cached_tokens
                any_tokens = True
            if m.cost_usd is not None:
                tot_cost += m.cost_usd
                any_cost = True
        return FinalMetrics(
            total_prompt_tokens=tot_p if any_tokens else None,
            total_completion_tokens=tot_c if any_tokens else None,
            total_cached_tokens=tot_cached if any_tokens else None,
            total_cost_usd=tot_cost if any_cost else None,
            total_steps=len(self._steps),
            extra=extra,
        )

    def build(self, final_metrics: FinalMetrics | None = None) -> Trajectory:
        if final_metrics is not None:
            self._final_metrics = final_metrics
        return Trajectory(
            schema_version=SCHEMA_VERSION,
            session_id=self.session_id,
            trajectory_id=self.trajectory_id,
            agent=self.agent,
            steps=list(self._steps),
            notes=self.notes,
            final_metrics=self._final_metrics,
            extra=self.extra,
        )

    def to_json_dict(self, *, exclude_none: bool = True) -> dict[str, Any]:
        return self.build().to_json_dict(exclude_none=exclude_none)

    def write_json(self, path: Path | str, *, indent: int = 2) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json_dict(), indent=indent), encoding="utf-8")
        return p


# ---------------------------------------------------------------------------
# validate() helper


def validate(path_or_dict: Path | str | dict[str, Any]) -> tuple[bool, list[str]]:
    """Round-trip a trajectory through the Pydantic models.

    Returns ``(ok, errors)``. ``errors`` is a flat list of "loc: msg" strings.
    """
    errors: list[str] = []
    payload: dict[str, Any] | None = None
    try:
        if isinstance(path_or_dict, dict):
            payload = path_or_dict
        else:
            p = Path(path_or_dict)
            if not p.is_file():
                return False, [f"{p}: file not found"]
            payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return False, [f"load failed: {e!r}"]

    try:
        Trajectory.model_validate(payload)
    except ValidationError as e:
        for err in e.errors():
            loc = ".".join(str(x) for x in err.get("loc", ()))
            errors.append(f"{loc}: {err.get('msg', '')}")
        return False, errors
    return True, []


def _cli(argv: list[str] | None = None) -> int:
    """``python -m kernel_synth.rl.atif <trace.json> [<trace.json> ...]``.

    Round-trip every trace JSON file through :func:`validate` and exit
    non-zero on the first failure. Designed for CI use; prints a short
    ``OK`` / ``FAIL`` line per file plus the validator errors.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m kernel_synth.rl.atif",
        description="Validate ATIF v1.7 trajectory JSON file(s).",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more trace JSON file paths.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only failures.",
    )
    args = parser.parse_args(argv)

    rc = 0
    for raw in args.paths:
        path = Path(raw)
        ok, errs = validate(path)
        if ok:
            if not args.quiet:
                print(f"OK   {path}")
        else:
            rc = 1
            print(f"FAIL {path}", file=sys.stderr)
            for e in errs[:20]:
                print(f"     · {e}", file=sys.stderr)
            if len(errs) > 20:
                print(f"     ... +{len(errs) - 20} more", file=sys.stderr)
    return rc


if __name__ == "__main__":
    import sys

    raise SystemExit(_cli(sys.argv[1:]))
