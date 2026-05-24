"""Agent harness: a tool-use loop that picks unique nn.Modules from a buffer.

The harness is deliberately scalable:
    * Tools are pure functions of ``CodeBuffer`` — no I/O outside the buffer.
    * The agent loop is provider-agnostic (see ``llm.LLMClient``).
    * Per-step output is captured in ``agent_log`` so the FastAPI viewer can
      replay the search.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .code_buffer import CodeBuffer, FileEntry, slice_source
from .llm import ChatResponse, LLMClient, ToolCall
from .models import ModuleCandidate


SYSTEM_PROMPT = """\
You are the *synthesizer* for an RL-environment factory aimed at custom-kernel
engineering. You are looking at one cloned GitHub repository at a time.

Your single job: select a *diverse, non-obvious* set of `torch.nn.Module`
subclasses from this repo that would make GOOD targets for someone writing
custom CUDA / Triton / Metal kernels as an RL task.

What "good" means here:
- The module does something the community does NOT already have an
  off-the-shelf fused kernel for. Examples of what to AVOID picking:
    * vanilla multi-head attention / self-attention / cross-attention
    * plain MLP / FeedForward / GLU stacks
    * Conv2d / BatchNorm / LayerNorm wrappers
    * thin embedding / linear projections
- Strongly PREFER modules that involve any of:
    * custom mixing patterns (selective scans, state-space ops, gated
      recurrences, Hyena/Monarch/Mamba-style ops, MoE routing math)
    * non-standard normalizations (RMS variants, dynamic-tanh, group-conditional)
    * unusual fused arithmetic in `forward` (lots of explicit einsum / cumsum
      / topk / FFT / complex-valued ops, custom positional encodings, rotary
      variants, sparse/structured masks)
    * imports of triton, flash_attn, xformers, mamba_ssm, causal_conv1d,
      or hand-written CUDA glue
    * meaningful, self-contained `forward` (≈30–400 LOC) with real arithmetic,
      not just composition of standard layers
- Diversity matters more than count. Aim for 4–10 modules total, spread
  across DIFFERENT mechanisms. Do not flag two near-duplicates of the same
  idea (e.g. two slightly different attention variants).

WORKFLOW

1. Start by calling `list_files`. It returns every Python file that defines
   at least one nn.Module subclass, with the class names inline.
2. Read promising files with `read_file`. Prefer files whose class names
   suggest unusual mechanisms. You may pass `start_line`/`end_line` to read
   a slice.
3. When you find a winner, call `mark_module` with the file path, class
   name, start/end line, a SHORT reason, a `novelty_score` in [0, 1], and
   2–5 short `tags`. Each `mark_module` call records exactly one candidate.
4. When you are done — or after you have marked ~8 strong candidates — call
   `finish`. Do not keep exploring once you have a good, diverse set.

Be decisive. Mark, don't deliberate. Skip the boring stuff.
"""


# ---------------------------------------------------------------------------
# Tool schemas

TOOLS: list[dict] = [
    {
        "name": "list_files",
        "description": (
            "List Python files that define at least one nn.Module subclass, "
            "with the class names inline. Optionally filter by a path prefix "
            "(e.g. 'mamba_ssm/modules')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path_prefix": {
                    "type": "string",
                    "description": "Optional repo-relative prefix to filter by.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of files to return (default 60).",
                    "minimum": 1,
                    "maximum": 200,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read the contents of a Python file from the repo. Optionally pass "
            "start_line/end_line (1-indexed, inclusive) to read a slice."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Repo-relative path, e.g. 'mamba_ssm/modules/mamba_simple.py'.",
                },
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "mark_module",
        "description": (
            "Record one nn.Module class as a candidate RL-environment target. "
            "Call once per candidate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "class_name": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
                "reason": {
                    "type": "string",
                    "description": "Why this module is a good custom-kernel target. 1–3 sentences.",
                },
                "novelty_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "0 = boring/standard, 1 = highly unique mechanism.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2–5 short tags (e.g. 'selective-scan', 'rotary', 'moe').",
                },
            },
            "required": [
                "file_path",
                "class_name",
                "start_line",
                "end_line",
                "reason",
                "novelty_score",
            ],
            "additionalProperties": False,
        },
    },
    {
        "name": "finish",
        "description": "Terminate exploration. Call this once you have a diverse, strong set.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "One sentence summary of the picks.",
                }
            },
            "additionalProperties": False,
        },
    },
]


# ---------------------------------------------------------------------------
# Harness


@dataclass
class HarnessResult:
    candidates: list[ModuleCandidate] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)
    summary: str = ""
    stopped_reason: str = ""


class AgentHarness:
    """One-shot agent loop over a single :class:`CodeBuffer`."""

    def __init__(
        self,
        buffer: CodeBuffer,
        *,
        llm: LLMClient,
        max_steps: int | None = None,
        max_read_bytes: int = 24_000,
        max_candidates: int = 10,
    ):
        self.buffer = buffer
        self.llm = llm
        self.max_steps = max_steps or int(
            os.environ.get("KERNEL_SYNTH_MAX_AGENT_STEPS", "24")
        )
        self.max_read_bytes = max_read_bytes
        self.max_candidates = max_candidates

    def run(self) -> HarnessResult:
        result = HarnessResult()
        seed = (
            "Here is the repo overview. Use `list_files`, `read_file`, "
            "`mark_module`, and `finish` to do your job.\n\n"
            f"```\n{self.buffer.overview()}\n```"
        )
        messages: list[dict] = [
            {"role": "user", "content": [{"type": "text", "text": seed}]}
        ]
        result.log.append({"step": 0, "kind": "seed", "text": seed[:400]})

        for step in range(1, self.max_steps + 1):
            try:
                resp: ChatResponse = self.llm.chat(
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    tools=TOOLS,
                )
            except Exception as e:  # noqa: BLE001
                result.stopped_reason = f"llm_error: {e!s}"
                result.log.append({"step": step, "kind": "error", "text": str(e)})
                break

            # Record the assistant turn in Anthropic-shaped form so we can
            # roundtrip into either provider.
            assistant_blocks: list[dict] = []
            if resp.text:
                assistant_blocks.append({"type": "text", "text": resp.text})
            for tc in resp.tool_calls:
                assistant_blocks.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                )
            messages.append({"role": "assistant", "content": assistant_blocks})

            if resp.text:
                result.log.append({"step": step, "kind": "thought", "text": resp.text})

            if resp.stop_reason != "tool_use" or not resp.tool_calls:
                result.stopped_reason = "model_stopped"
                break

            tool_results: list[dict] = []
            finish_called = False
            for tc in resp.tool_calls:
                output, finish, summary = self._execute_tool(tc, result)
                result.log.append(
                    {
                        "step": step,
                        "kind": "tool",
                        "name": tc.name,
                        "input": tc.arguments,
                        "output_preview": (output or "")[:300],
                    }
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": output or "",
                    }
                )
                if finish:
                    finish_called = True
                    result.summary = summary

            messages.append({"role": "user", "content": tool_results})

            if finish_called:
                result.stopped_reason = "finish_called"
                break
            if len(result.candidates) >= self.max_candidates:
                result.stopped_reason = "max_candidates"
                break
        else:
            result.stopped_reason = "max_steps"

        return result

    # ------------------------------------------------------------------
    # Tool execution

    def _execute_tool(
        self,
        tc: ToolCall,
        result: HarnessResult,
    ) -> tuple[str, bool, str]:
        name = tc.name
        args = tc.arguments or {}
        try:
            if name == "list_files":
                return self._tool_list_files(args), False, ""
            if name == "read_file":
                return self._tool_read_file(args), False, ""
            if name == "mark_module":
                return self._tool_mark_module(args, result), False, ""
            if name == "finish":
                summary = str(args.get("summary", "")).strip()
                return ("OK. Stopping exploration.", True, summary)
        except _ToolError as e:
            return (f"ERROR: {e}", False, "")
        return (f"ERROR: unknown tool {name!r}", False, "")

    def _tool_list_files(self, args: dict[str, Any]) -> str:
        prefix = str(args.get("path_prefix") or "")
        limit = int(args.get("limit") or 60)
        files = [
            f for f in self.buffer.files_with_nn_modules() if f.path.startswith(prefix)
        ]
        files.sort(key=lambda f: -f.n_nn_modules)
        files = files[:limit]
        if not files:
            return f"No nn.Module files match prefix {prefix!r}."
        label = repr(prefix) if prefix else "<all>"
        lines = [f"{len(files)} files matching {label}:"]
        for f in files:
            cls_names = [c.name for c in f.classes if c.is_nn_module]
            joined = ", ".join(cls_names[:6])
            if len(cls_names) > 6:
                joined += f", … (+{len(cls_names) - 6})"
            lines.append(f"  {f.path} ({f.n_nn_modules})  -> {joined}")
        return "\n".join(lines)

    def _tool_read_file(self, args: dict[str, Any]) -> str:
        path = str(args.get("path") or "").strip()
        if not path:
            raise _ToolError("missing 'path'")
        file = self.buffer.by_path(path)
        if file is None:
            # Try matching by suffix for resilience.
            matches = [f for f in self.buffer.files if f.path.endswith(path)]
            if not matches:
                raise _ToolError(f"file not in buffer: {path}")
            file = matches[0]
        start = args.get("start_line")
        end = args.get("end_line")
        if start or end:
            s = max(int(start or 1), 1)
            e = int(end or s + 200)
            text = slice_source(file, s, e)
            header = f"# {file.path}  lines {s}-{e}\n"
        else:
            text = file.read()
            header = f"# {file.path}  (full, {file.n_lines} lines)\n"
        if len(text) > self.max_read_bytes:
            text = text[: self.max_read_bytes] + "\n# … truncated\n"
        return header + text

    def _tool_mark_module(
        self, args: dict[str, Any], result: HarnessResult
    ) -> str:
        path = str(args.get("file_path") or "").strip()
        file = self.buffer.by_path(path)
        if file is None:
            matches = [f for f in self.buffer.files if f.path.endswith(path)]
            if not matches:
                raise _ToolError(f"file not in buffer: {path}")
            file = matches[0]
            path = file.path

        cls_name = str(args.get("class_name") or "").strip()
        if not cls_name:
            raise _ToolError("missing 'class_name'")
        # Snap to the real AST range if the agent's lines drift.
        cls_entry = next(
            (c for c in file.classes if c.name == cls_name and c.is_nn_module),
            None,
        )
        if cls_entry is None:
            raise _ToolError(
                f"class {cls_name!r} (nn.Module) not found in {path}"
            )
        start_line = cls_entry.start_line
        end_line = cls_entry.end_line

        src = slice_source(file, start_line, end_line)
        cand = ModuleCandidate(
            file_path=path,
            class_name=cls_name,
            start_line=start_line,
            end_line=end_line,
            reason=str(args.get("reason") or "").strip() or "(no reason given)",
            novelty_score=float(args.get("novelty_score") or 0.5),
            tags=[str(t) for t in (args.get("tags") or [])][:6],
            source_code=src,
        )
        result.candidates.append(cand)
        return f"Marked {cls_name} in {path} (lines {start_line}-{end_line})."


class _ToolError(RuntimeError):
    pass
