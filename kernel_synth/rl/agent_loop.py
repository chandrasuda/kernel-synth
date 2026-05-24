"""Rollouts that produce ATIF v1.7 trajectories from a ``KernelEnv``.

Three modes:

* ``"baseline"`` — no LLM. One ``run_benchmark`` + ``finish`` call. Proves
  the format end-to-end and gives a reward of ~0 (the baseline solution
  just wraps reference).
* ``"torch_compile"`` — also scriptless. The agent step writes a
  ``torch.compile``-wrapped solution, benchmarks it, then finishes.
  Demonstrates a reward near 1.0 on machines where ``torch.compile``
  actually helps.
* ``"agent"`` — real LLM tool-use loop via ``kernel_synth.llm.LLMClient``.
  Each LLM turn becomes one agent ATIF step; per-call observations land in
  the same step's ``observation.results``.

Every rollout terminates by writing
``env_dir/traces/<UTC-timestamp>__<mode>.json``.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ..llm import LLMClient, _redact_secrets
from .atif import (
    AtifAgent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    ToolCall,
    Trajectory,
    TrajectoryBuilder,
)
from .env import KernelEnv
from .prompts import KERNEL_AGENT_SYSTEM_PROMPT
from .tools import KernelAgentTools


# Re-export legacy names ``__init__.py`` expects.
RolloutMode = Literal["baseline", "torch_compile", "agent"]


class RolloutResult:
    """Lightweight container for the result of one ``rollout()`` call.

    Attributes
    ----------
    trajectory : Trajectory
        The validated ATIF trajectory.
    reward : float
        The final scalar reward.
    components : dict
        The reward components dict (timings, ratios, raw benchmark output).
    trace_path : Path
        Where the trajectory JSON was written.
    mode : str
        Which mode produced this rollout.
    """

    def __init__(
        self,
        *,
        trajectory: Trajectory,
        reward: float,
        components: dict[str, Any],
        trace_path: Path,
        mode: str,
    ) -> None:
        self.trajectory = trajectory
        self.reward = float(reward)
        self.components = components
        self.trace_path = Path(trace_path)
        self.mode = mode

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"RolloutResult(mode={self.mode!r}, reward={self.reward:.3f}, "
            f"trace={self.trace_path.name!r})"
        )


# AgentRollout is the legacy name __init__.py expects; expose as alias.
AgentRollout = RolloutResult


# ---------------------------------------------------------------------------
# Public API


# Independent cap so a single chatty turn can't fire 100 tool calls and
# blow past the per-turn step budget. Tuned to comfortably accommodate
# realistic kernel-engineering rollouts (read + write + benchmark loops).
DEFAULT_MAX_TOOL_CALLS = 80


def rollout(
    env_dir: Path | str,
    *,
    mode: RolloutMode = "baseline",
    llm: LLMClient | None = None,
    max_steps: int = 20,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    model_label: str | None = None,
    final_runs: int = 20,
    benchmark_timeout_s: float = 120.0,
) -> RolloutResult:
    """Run one rollout against the env at ``env_dir`` and persist the trace.

    Parameters
    ----------
    env_dir : Path
        The env folder (must contain ``benchmark.py``).
    mode : {"baseline", "torch_compile", "agent"}
        See module docstring.
    llm : LLMClient | None
        Required only for ``mode="agent"``.
    max_steps : int
        Hard cap on LLM-driven turns in ``agent`` mode.
    max_tool_calls : int
        Hard cap on total tool invocations across the rollout, independent
        of ``max_steps``. Defends against a chatty turn issuing dozens of
        tool calls.
    model_label : str | None
        Optional override for the agent's ``model_name`` (otherwise pulled
        from ``llm.model``).
    final_runs : int
        Number of timing samples for the final benchmark.
    benchmark_timeout_s : float
        Subprocess timeout for each ``run_benchmark`` invocation.
    """
    env = KernelEnv(
        env_dir,
        benchmark_timeout_s=benchmark_timeout_s,
    )
    if mode == "baseline":
        return _rollout_baseline(env, final_runs=final_runs)
    if mode == "torch_compile":
        return _rollout_torch_compile(env, final_runs=final_runs)
    if mode == "agent":
        if llm is None:
            raise ValueError("mode='agent' requires a LLMClient")
        return _rollout_agent(
            env,
            llm=llm,
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            model_label=model_label,
            final_runs=final_runs,
        )
    raise ValueError(f"unknown rollout mode: {mode!r}")


# ---------------------------------------------------------------------------
# Mode: baseline


def _rollout_baseline(env: KernelEnv, *, final_runs: int) -> RolloutResult:
    obs0 = env.reset()
    builder = _new_builder(
        env, agent_name="kernel-synth-baseline", model_name=None, mode="baseline"
    )
    builder.add_step(source="system", message=KERNEL_AGENT_SYSTEM_PROMPT)
    builder.add_step(source="user", message=obs0["prompt"])

    bench_call = _tc("run_benchmark", {"runs": final_runs})
    finish_call = _tc("finish", {"notes": "baseline rollout — no edits."})
    bench_result = env.step({"name": "run_benchmark", "arguments": {"runs": final_runs}})
    finish_result = env.step({"name": "finish", "arguments": {"notes": "baseline rollout — no edits."}})

    builder.add_step(
        source="agent",
        model_name="baseline-noop",
        message=(
            "Running the baseline benchmark and finishing — solution.py "
            "wraps the reference verbatim."
        ),
        tool_calls=[bench_call, finish_call],
        observation=Observation(
            results=[
                ObservationResult(
                    source_call_id=bench_call.tool_call_id,
                    content=json.dumps(_safe_json(bench_result.observation), indent=2),
                ),
                ObservationResult(
                    source_call_id=finish_call.tool_call_id,
                    content=str(finish_result.observation),
                ),
            ]
        ),
        metrics=Metrics(extra={"per_step_reward": 0.0}),
    )

    return _finalize(env, builder, mode="baseline", final_runs=final_runs)


# ---------------------------------------------------------------------------
# Mode: torch_compile


_TORCH_COMPILE_SOLUTION_TEMPLATE = '''\
"""Auto-written by kernel-synth torch_compile rollout."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import reference  # noqa: E402


def build(**kwargs):
    mod = reference.{class_name}(**kwargs)
    mod.eval()
    try:
        return torch.compile(mod, dynamic=True, fullgraph=False)
    except Exception:
        return mod
'''


def _rollout_torch_compile(env: KernelEnv, *, final_runs: int) -> RolloutResult:
    obs0 = env.reset()
    builder = _new_builder(
        env, agent_name="kernel-synth-torch-compile", model_name=None,
        mode="torch_compile",
    )
    builder.add_step(source="system", message=KERNEL_AGENT_SYSTEM_PROMPT)
    builder.add_step(source="user", message=obs0["prompt"])

    soln = _TORCH_COMPILE_SOLUTION_TEMPLATE.format(class_name=env.class_name)
    write_call = _tc(
        "write_file",
        {"path": "solution.py", "content": soln},
    )
    bench_call = _tc("run_benchmark", {"runs": final_runs})
    finish_call = _tc(
        "finish",
        {"notes": "Wrapped reference with torch.compile."},
    )

    write_obs = env.step({"name": "write_file", "arguments": {"path": "solution.py", "content": soln}})
    bench_obs = env.step({"name": "run_benchmark", "arguments": {"runs": final_runs}})
    finish_obs = env.step({"name": "finish", "arguments": {"notes": "Wrapped reference with torch.compile."}})

    builder.add_step(
        source="agent",
        model_name="torch.compile-shim",
        message=(
            "Replacing solution.py with a torch.compile() wrapper around the "
            "reference module and re-benchmarking."
        ),
        tool_calls=[write_call, bench_call, finish_call],
        observation=Observation(
            results=[
                ObservationResult(
                    source_call_id=write_call.tool_call_id,
                    content=str(write_obs.observation),
                ),
                ObservationResult(
                    source_call_id=bench_call.tool_call_id,
                    content=json.dumps(_safe_json(bench_obs.observation), indent=2),
                ),
                ObservationResult(
                    source_call_id=finish_call.tool_call_id,
                    content=str(finish_obs.observation),
                ),
            ]
        ),
        metrics=Metrics(extra={"per_step_reward": 0.0}),
    )

    return _finalize(env, builder, mode="torch_compile", final_runs=final_runs)


# ---------------------------------------------------------------------------
# Mode: agent (real LLM tool-use)


def _rollout_agent(
    env: KernelEnv,
    *,
    llm: LLMClient,
    max_steps: int,
    max_tool_calls: int,
    model_label: str | None,
    final_runs: int,
) -> RolloutResult:
    obs0 = env.reset()
    builder = _new_builder(
        env,
        agent_name="kernel-synth-agent",
        model_name=model_label or llm.model,
        mode="agent",
    )
    builder.add_step(source="system", message=KERNEL_AGENT_SYSTEM_PROMPT)
    builder.add_step(source="user", message=obs0["prompt"])

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": obs0["prompt"]}]}
    ]
    llm_call_count = 0
    total_tool_calls = 0
    tool_cap_hit = False

    for _turn in range(max_steps):
        try:
            resp = llm.chat(
                system=KERNEL_AGENT_SYSTEM_PROMPT,
                messages=messages,
                tools=KernelAgentTools.TOOL_SCHEMAS,
            )
        except Exception as e:  # noqa: BLE001
            redacted = _redact_secrets(repr(e))
            builder.add_step(
                source="agent",
                model_name=llm.model,
                message=f"(LLM error: {redacted})",
                metrics=Metrics(extra={"per_step_reward": 0.0, "llm_error": redacted}),
            )
            break
        llm_call_count += 1

        # Mirror the assistant turn into our running message list in
        # Anthropic-shaped form so the next LLM call sees it.
        asst_blocks: list[dict[str, Any]] = []
        if resp.text:
            asst_blocks.append({"type": "text", "text": resp.text})
        for tc in resp.tool_calls:
            asst_blocks.append(
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
            )
        messages.append({"role": "assistant", "content": asst_blocks})

        atif_tool_calls = [
            ToolCall(
                tool_call_id=tc.id,
                function_name=tc.name,
                arguments=tc.arguments or {},
            )
            for tc in resp.tool_calls
        ]

        results: list[ObservationResult] = []
        tool_results_for_next: list[dict[str, Any]] = []
        finish_called = False
        for tc in resp.tool_calls:
            step_out = env.step(
                {"name": tc.name, "arguments": tc.arguments or {}}
            )
            content = step_out.observation
            content_str = (
                json.dumps(_safe_json(content), indent=2)
                if isinstance(content, dict)
                else str(content)
            )
            results.append(
                ObservationResult(
                    source_call_id=tc.id,
                    content=content_str,
                )
            )
            tool_results_for_next.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": content_str,
                }
            )
            if step_out.done:
                finish_called = True

        usage = _usage_from_response(resp)
        metrics = Metrics(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            cached_tokens=usage.get("cached_tokens"),
            extra={"per_step_reward": 0.0},
        )

        builder.add_step(
            source="agent",
            model_name=llm.model,
            message=resp.text or "",
            tool_calls=atif_tool_calls or None,
            observation=Observation(results=results) if results else None,
            metrics=metrics,
            llm_call_count=llm_call_count,
        )

        total_tool_calls += len(resp.tool_calls)
        if not resp.tool_calls:
            break
        if finish_called:
            break
        if total_tool_calls >= max_tool_calls:
            tool_cap_hit = True
            builder.add_step(
                source="user",
                message=(
                    f"(max_tool_calls={max_tool_calls} reached after "
                    f"{total_tool_calls} calls — finalizing automatically)"
                ),
            )
            break

        messages.append({"role": "user", "content": tool_results_for_next})
    else:
        # max_steps exhausted without a finish — append a sentinel step.
        builder.add_step(
            source="user",
            message=(
                f"(max_steps={max_steps} exhausted — finalizing automatically)"
            ),
        )

    result = _finalize(
        env, builder, mode="agent", final_runs=final_runs,
        llm_call_count=llm_call_count,
    )
    if tool_cap_hit:
        result.components.setdefault("tool_cap_hit", True)
    return result


# ---------------------------------------------------------------------------
# Common finalize


def _finalize(
    env: KernelEnv,
    builder: TrajectoryBuilder,
    *,
    mode: str,
    final_runs: int,
    llm_call_count: int | None = None,
) -> RolloutResult:
    reward_dict = env.finalize(runs=final_runs)
    reward = float(reward_dict["reward"])
    components = reward_dict["components"]

    # Stash the reward on the last agent-or-user step's metrics.extra and
    # on the trajectory-level final_metrics.extra (per spec convention).
    for step in reversed(builder.steps):
        if step.source == "agent":
            extra = dict(step.extra or {})
            extra["rollout_reward"] = reward
            extra["rollout_components"] = components
            step.extra = extra
            # Also patch metrics.extra for downstream RL post-training
            # tools that look there.
            if step.metrics is None:
                step.metrics = Metrics(extra={"reward": reward, "components": components})
            else:
                m_extra = dict(step.metrics.extra or {})
                m_extra["reward"] = reward
                m_extra["components"] = components
                step.metrics.extra = m_extra
            break

    aggregate = builder.aggregate_metrics(
        extra={
            "reward": reward,
            "components": components,
            "mode": mode,
        }
    )
    builder.set_final_metrics(aggregate)
    if llm_call_count is not None:
        builder.extra = {**(builder.extra or {}), "llm_call_count": llm_call_count}

    trace_path = _trace_path(env.env_dir, mode=mode)
    builder.write_json(trace_path)
    traj = builder.build()

    return RolloutResult(
        trajectory=traj,
        reward=reward,
        components=components,
        trace_path=trace_path,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Helpers


def _new_builder(
    env: KernelEnv,
    *,
    agent_name: str,
    model_name: str | None,
    mode: str,
) -> TrajectoryBuilder:
    return TrajectoryBuilder(
        agent=AtifAgent(
            name=agent_name,
            version="0.1.0",
            model_name=model_name,
            tool_definitions=KernelAgentTools.TOOL_SCHEMAS,
            extra={"env": env.env_dir.name, "class_name": env.class_name},
        ),
        notes=f"kernel-synth rollout (mode={mode}) on {env.env_dir.name}",
        extra={
            "env_name": env.env_dir.name,
            "env_path": str(env.env_dir),
            "class_name": env.class_name,
            "mode": mode,
        },
    )


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _trace_path(env_dir: Path, *, mode: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{ts}__{_SLUG_RE.sub('_', mode)}__{uuid.uuid4().hex[:6]}.json"
    return env_dir / "traces" / name


def _tc(name: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(
        tool_call_id=f"call_{uuid.uuid4().hex[:12]}",
        function_name=name,
        arguments=arguments,
    )


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _usage_from_response(resp: Any) -> dict[str, int | None]:
    raw = getattr(resp, "raw", None)
    usage = getattr(raw, "usage", None)
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "cached_tokens": None}
    pt = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
    ct = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
    cached = getattr(usage, "cache_read_input_tokens", None)
    return {
        "prompt_tokens": int(pt) if pt is not None else None,
        "completion_tokens": int(ct) if ct is not None else None,
        "cached_tokens": int(cached) if cached is not None else None,
    }


__all__ = ["AgentRollout", "RolloutMode", "RolloutResult", "rollout"]
