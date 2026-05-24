"""Gym-style environment wrapping ONE kernel-synth env folder.

The environment is intentionally thin: tool dispatch + reset / finalize are
all that's specific. The rollout loop in ``agent_loop`` drives it.

Per-step reward is always 0 — the reward is sparse and computed in
``finalize()`` from the final benchmark output via ``rewards.compute_reward``.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .prompts import KERNEL_AGENT_SYSTEM_PROMPT, render_user_prompt
from .rewards import compute_reward
from .tools import KernelAgentTools


@dataclass
class StepResult:
    """One environment step return — mimics gym's (obs, reward, done, info)."""

    observation: str | dict[str, Any]
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class KernelEnv:
    """One env folder, exposed as a gym-style environment.

    Lifecycle::

        env = KernelEnv(env_dir)
        obs = env.reset()              # baseline benchmark + initial prompt
        for tool_call in agent_loop:
            step = env.step(tool_call) # dispatch one tool call
            if step.done: break
        final = env.finalize()         # reward + components
    """

    def __init__(
        self,
        env_dir: Path | str,
        *,
        python: str | None = None,
        benchmark_timeout_s: float = 120.0,
    ) -> None:
        self.env_dir = Path(env_dir).resolve()
        if not self.env_dir.is_dir():
            raise FileNotFoundError(f"env_dir not found: {self.env_dir}")
        self.python = python
        self.benchmark_timeout_s = benchmark_timeout_s
        self.tools = KernelAgentTools(
            self.env_dir,
            python=self.python,
            benchmark_timeout_s=self.benchmark_timeout_s,
        )

        self._originals: dict[str, str] = {}
        self._baseline: dict[str, Any] | None = None
        self._reference_source: str | None = None
        self._inputs_source: str | None = None
        self._class_name: str | None = None
        self._seed: int | None = None

        self._load_env_metadata()
        self._cache_originals()

    # ------------------------------------------------------------------
    # Public surface

    @property
    def class_name(self) -> str:
        return self._class_name or "Unknown"

    @property
    def baseline(self) -> dict[str, Any] | None:
        return self._baseline

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        """Restore solution + triton_kernels to originals, clear workspace,
        run a baseline benchmark, and return the initial observation.

        Parameters
        ----------
        seed : int | None
            If provided, propagated to the benchmark subprocess via the
            ``KERNEL_SYNTH_SEED`` env var so the same random module
            init / input tensors are used across this rollout's baseline
            run AND any subsequent ``run_benchmark`` calls the agent
            makes. Useful for making A/B comparisons reproducible.
        """
        self._restore_originals()
        self._ensure_workspace()
        if seed is not None:
            os.environ["KERNEL_SYNTH_SEED"] = str(int(seed))
            self._seed = int(seed)
        self.tools = KernelAgentTools(
            self.env_dir,
            python=self.python,
            benchmark_timeout_s=self.benchmark_timeout_s,
        )

        baseline = self.tools.run_benchmark(runs=10)
        self._baseline = baseline

        prompt = self.make_initial_prompt(baseline=baseline)
        return {
            "prompt": prompt,
            "eager_ms": _f(baseline.get("eager_ms")),
            "compile_ms": _f(baseline.get("compile_ms")),
            "solution_ms": _f(baseline.get("solution_ms")),
            "correct": bool(baseline.get("correct", False)),
            "baseline_raw": baseline,
            "seed": self._seed,
        }

    def step(self, tool_call: dict[str, Any]) -> StepResult:
        """Dispatch one tool call. Per-step reward is always 0.

        ``tool_call`` shape::

            {"name": "read_file", "arguments": {"path": "reference.py"}}
            {"name": "write_file", "arguments": {"path": "...", "content": "..."}}
            {"name": "run_benchmark", "arguments": {"runs": 10}}
            {"name": "finish", "arguments": {"notes": "..."}}
        """
        name = str(tool_call.get("name", ""))
        args = tool_call.get("arguments") or {}
        output = self.tools.dispatch(name, args)

        if isinstance(output, dict):
            observation: str | dict[str, Any] = output
            info = {"tool": name, "output_kind": "json"}
        else:
            observation = str(output)
            info = {"tool": name, "output_kind": "text"}

        done = bool(self.tools.finished)
        if done:
            info["finish_notes"] = self.tools.finish_notes
        return StepResult(observation=observation, reward=0.0, done=done, info=info)

    def finalize(self, *, runs: int = 20) -> dict[str, Any]:
        """Run the final benchmark, compute the reward, and return the dict.

        Returned shape (from ``compute_reward``)::

            {"reward": float, "components": {...timings, ratios...}}

        Also surfaces the raw benchmark JSON under ``components.benchmark``.
        """
        result = self.tools.run_benchmark(runs=runs)
        reward = compute_reward(
            correct=bool(result.get("correct", False)),
            eager_ms=_f(result.get("eager_ms")),
            compile_ms=_f(result.get("compile_ms")),
            solution_ms=_f(result.get("solution_ms")),
        )
        reward["components"]["benchmark"] = result
        return reward

    def make_initial_prompt(
        self,
        *,
        baseline: dict[str, Any] | None = None,
    ) -> str:
        if baseline is None:
            baseline = self._baseline or {}
        return render_user_prompt(
            class_name=self.class_name,
            reference_source=self._reference_source or "(reference.py missing)",
            inputs_source=self._inputs_source or "(inputs.py missing)",
            eager_ms=_f(baseline.get("eager_ms")) or 0.0,
            compile_ms=_f(baseline.get("compile_ms")),
            solution_ms=_f(baseline.get("solution_ms")) or 0.0,
        )

    # ------------------------------------------------------------------
    # Internals

    def _load_env_metadata(self) -> None:
        meta_path = self.env_dir / "env.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self._class_name = meta.get("class_name")
            except (OSError, json.JSONDecodeError):
                pass
        for fname, attr in (("reference.py", "_reference_source"),
                            ("inputs.py", "_inputs_source")):
            p = self.env_dir / fname
            if p.is_file():
                try:
                    setattr(self, attr, p.read_text(encoding="utf-8"))
                except OSError:
                    pass

    def _cache_originals(self) -> None:
        for fname in ("solution.py", "triton_kernels.py"):
            p = self.env_dir / fname
            if p.is_file():
                try:
                    self._originals[fname] = p.read_text(encoding="utf-8")
                except OSError:
                    pass

    def _restore_originals(self) -> None:
        for fname, content in self._originals.items():
            (self.env_dir / fname).write_text(content, encoding="utf-8")

    def _ensure_workspace(self) -> None:
        """Recreate workspace/ from scratch.

        We re-create the directory unconditionally so a missing or partially
        deleted ``workspace/`` between rollouts heals on the next ``reset()``.
        If something in there is held open by another process the rmtree
        ignores the failure and we still ``mkdir(parents=True, exist_ok=True)``
        to guarantee the path exists.
        """
        ws = self.env_dir / "workspace"
        if ws.is_dir():
            shutil.rmtree(ws, ignore_errors=True)
        elif ws.exists():
            try:
                ws.unlink()
            except OSError:
                pass
        ws.mkdir(parents=True, exist_ok=True)


def _f(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v


__all__ = ["KernelEnv", "StepResult"]
