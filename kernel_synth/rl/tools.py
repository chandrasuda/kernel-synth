"""Sandboxed tools the kernel-engineering agent uses inside one env folder.

Each tool is a thin Python function bound to a single ``env_dir``. All path
arguments are resolved against the env directory and `..` traversal is
rejected. Writes are restricted to an allowlist so the agent can't clobber
``reference.py`` / ``benchmark.py`` / ``inputs.py`` / metadata.

The class also exposes ``TOOL_SCHEMAS`` in the same OpenAI/Anthropic-shaped
format as ``kernel_synth.harness.TOOLS``, so we can hand it straight to
``LLMClient.chat(..., tools=KernelAgentTools.TOOL_SCHEMAS)``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAX_WRITE_BYTES = 200_000
MAX_READ_BYTES = 200_000
DEFAULT_BENCH_TIMEOUT_S = 120

# Files the agent is explicitly NOT allowed to write.
PROTECTED_FILES = frozenset(
    {
        "reference.py",
        "inputs.py",
        "benchmark.py",
        "harness.py",
        "env.json",
        "README.md",
    }
)

# Top-level filenames the agent CAN write (anything under workspace/ is also OK).
WRITABLE_FILES = frozenset({"solution.py", "triton_kernels.py", "notes.md"})


@dataclass
class ToolError(Exception):
    """Raised by individual tools when the request is malformed or rejected.

    The agent loop catches this and turns it into a string ``ERROR: ...``
    observation so the model can recover.
    """

    message: str

    def __str__(self) -> str:
        return self.message


class KernelAgentTools:
    """Bundle of stateful tool callables bound to one env directory.

    Usage::

        tools = KernelAgentTools(env_dir, python=sys.executable)
        out = tools.list_files()
        out = tools.read_file("reference.py")
        out = tools.write_file("triton_kernels.py", "...")
        out = tools.run_benchmark(runs=10)
        out = tools.finish(notes="...")

    Or by name::

        out = tools.dispatch("read_file", {"path": "reference.py"})
    """

    TOOL_SCHEMAS: list[dict[str, Any]] = [
        {
            "name": "list_files",
            "description": (
                "List every file in this env folder with its size in bytes. "
                "Use this to see the workspace tree."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "read_file",
            "description": (
                "Read the contents of a file inside the env folder. "
                "Paths are relative to the env root; `..` is rejected."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the env folder.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "write_file",
            "description": (
                "Write or overwrite a file inside the env folder. "
                "Allowed targets: solution.py, triton_kernels.py, notes.md, "
                "or any path under workspace/. Caps at 200 KB. "
                "Use this to update solution.py or save Triton kernels."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the env folder.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full new file contents (overwrites the file).",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
        {
            "name": "run_benchmark",
            "description": (
                "Run `python benchmark.py --json --runs <N>` in this env. "
                "Returns a dict with eager_ms, compile_ms, solution_ms, correct, "
                "max_diff, eager_speedup, compile_ratio (or an error key)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "runs": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200,
                        "description": "Timing samples per module (default 10).",
                    }
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "finish",
            "description": (
                "Terminate the rollout. Optionally include a short summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "notes": {
                        "type": "string",
                        "description": "Optional 1-3 sentence summary of what you tried.",
                    }
                },
                "additionalProperties": False,
            },
        },
    ]

    def __init__(
        self,
        env_dir: Path | str,
        *,
        python: str | None = None,
        benchmark_timeout_s: float = DEFAULT_BENCH_TIMEOUT_S,
    ) -> None:
        self.env_dir = Path(env_dir).resolve()
        if not self.env_dir.is_dir():
            raise ValueError(f"env_dir does not exist: {self.env_dir}")
        self.python = python or sys.executable
        self.benchmark_timeout_s = float(benchmark_timeout_s)
        self.finished = False
        self.finish_notes = ""

    # ------------------------------------------------------------------
    # Dispatch

    def dispatch(self, name: str, args: dict[str, Any] | None) -> Any:
        """Invoke a tool by name. Returns whatever the tool returns."""
        args = args or {}
        try:
            if name == "list_files":
                return self.list_files()
            if name == "read_file":
                return self.read_file(str(args.get("path", "")))
            if name == "write_file":
                return self.write_file(
                    str(args.get("path", "")),
                    str(args.get("content", "")),
                )
            if name == "run_benchmark":
                runs = int(args.get("runs", 10)) if args.get("runs") is not None else 10
                return self.run_benchmark(runs=runs)
            if name == "finish":
                return self.finish(notes=str(args.get("notes", "")))
        except ToolError as e:
            return f"ERROR: {e}"
        return f"ERROR: unknown tool {name!r}"

    # ------------------------------------------------------------------
    # Tools

    def list_files(self) -> str:
        rows: list[tuple[str, int]] = []
        for p in sorted(self.env_dir.rglob("*")):
            if not p.is_file():
                continue
            if p.name == "__pycache__" or "__pycache__" in p.parts:
                continue
            try:
                rel = str(p.relative_to(self.env_dir))
            except ValueError:
                continue
            rows.append((rel, p.stat().st_size))
        if not rows:
            return "(empty env)"
        width = max(len(r) for r, _ in rows)
        return "\n".join(f"{r:<{width}}  {s:>9} B" for r, s in rows)

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise ToolError(f"file not found: {path}")
        try:
            data = target.read_bytes()
        except OSError as e:
            raise ToolError(f"read failed: {e}") from e
        if len(data) > MAX_READ_BYTES:
            text = data[:MAX_READ_BYTES].decode("utf-8", errors="replace")
            text += f"\n# ... truncated at {MAX_READ_BYTES} bytes ..."
            return text
        return data.decode("utf-8", errors="replace")

    def write_file(self, path: str, content: str) -> str:
        target = self._resolve(path)
        rel = target.relative_to(self.env_dir)
        self._assert_writable(rel)

        if not isinstance(content, str):
            raise ToolError("content must be a string")
        encoded = content.encode("utf-8")
        if len(encoded) > MAX_WRITE_BYTES:
            raise ToolError(
                f"content too large: {len(encoded)} > {MAX_WRITE_BYTES} bytes"
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded)
        return f"OK: wrote {len(encoded)} bytes to {rel.as_posix()}"

    def run_benchmark(self, runs: int = 10) -> dict[str, Any]:
        bench = self.env_dir / "benchmark.py"
        if not bench.is_file():
            return {"error": "benchmark_missing", "detail": str(bench)}
        runs = max(1, min(int(runs), 200))
        cmd = [self.python, str(bench), "--json", "--runs", str(runs)]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.env_dir),
                capture_output=True,
                text=True,
                timeout=self.benchmark_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {
                "error": "timeout",
                "detail": f"benchmark exceeded {self.benchmark_timeout_s:.0f}s",
            }
        except OSError as e:
            return {"error": "spawn_failed", "detail": repr(e)}

        stdout = (proc.stdout or "").strip()
        stderr_tail = (proc.stderr or "").strip()[-2000:]
        parsed: dict[str, Any] | None = None
        if stdout:
            # The harness sometimes prints lines before the JSON object on
            # CPU; take the last `{...}` blob.
            blob = _last_json_object(stdout)
            if blob is not None:
                try:
                    parsed = json.loads(blob)
                except json.JSONDecodeError:
                    parsed = None
        if parsed is None:
            return {
                "error": "no_json",
                "returncode": proc.returncode,
                "stdout_tail": stdout[-2000:],
                "stderr_tail": stderr_tail,
            }
        if proc.returncode != 0 and "error" not in parsed:
            parsed.setdefault("returncode", proc.returncode)
            if stderr_tail:
                parsed.setdefault("stderr_tail", stderr_tail)
        return parsed

    def finish(self, notes: str = "") -> str:
        self.finished = True
        self.finish_notes = notes or ""
        return "OK: finishing rollout."

    # ------------------------------------------------------------------
    # Internals

    def _resolve(self, path: str) -> Path:
        if not path:
            raise ToolError("missing path")
        p = (self.env_dir / path).resolve()
        try:
            p.relative_to(self.env_dir)
        except ValueError as e:
            raise ToolError(f"path escapes env folder: {path}") from e
        return p

    def _assert_writable(self, rel: Path) -> None:
        parts = rel.parts
        if not parts:
            raise ToolError("cannot write the env folder itself")
        if rel.as_posix() in PROTECTED_FILES or parts[0] in PROTECTED_FILES:
            raise ToolError(f"refusing to write protected file: {rel.as_posix()}")
        # Writable: top-level file in WRITABLE_FILES, or anything under workspace/.
        if parts[0] == "workspace":
            return
        if len(parts) == 1 and parts[0] in WRITABLE_FILES:
            return
        raise ToolError(
            f"path not in writable allowlist: {rel.as_posix()} "
            f"(allowed: {sorted(WRITABLE_FILES)} or workspace/...)"
        )


def _last_json_object(text: str) -> str | None:
    """Return the last balanced ``{...}`` blob in ``text`` (or None)."""
    end = text.rfind("}")
    if end < 0:
        return None
    depth = 0
    for i in range(end, -1, -1):
        ch = text[i]
        if ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                return text[i : end + 1]
    return None


__all__ = ["KernelAgentTools", "ToolError", "WRITABLE_FILES", "PROTECTED_FILES"]
