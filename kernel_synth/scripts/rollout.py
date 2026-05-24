"""Run a kernel-engineering rollout against one env folder.

Usage:
    python -m kernel_synth.scripts.rollout <env_name_or_path> \\
        [--mode baseline|torch_compile|agent] \\
        [--runs N] [--max-steps N] [--envs-root PATH]

Default mode is ``baseline`` (no LLM, just proves the trace format).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import cast

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kernel_synth.rl import (
    KernelEnv,
    RolloutMode,
    rollout,
    validate,
)

console = Console()


def resolve_env_dir(target: str, envs_root: Path) -> Path:
    p = Path(target)
    if p.is_dir():
        return p.resolve()
    candidate = envs_root / target
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(
        f"could not find env: tried {p} and {candidate}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env", help="Env folder name (under envs/) or absolute path.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "torch_compile", "agent"],
        default="baseline",
    )
    parser.add_argument("--runs", type=int, default=20, help="Final benchmark runs.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="LLM-turn budget for agent mode.",
    )
    parser.add_argument(
        "--envs-root",
        default="./envs",
        help="Where the env folders live (default ./envs).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model label override (agent mode).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed for torch.manual_seed inside this process and the benchmark "
            "subprocess (via KERNEL_SYNTH_SEED). Defaults to 0 inside the "
            "harness."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit the rollout result as a single JSON object on stdout "
            "instead of the rich Panel/Table view. Suitable for piping into "
            "jq / scripts."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Sampling temperature passed to the LLM (agent mode). "
            "Unset uses the provider default."
        ),
    )
    args = parser.parse_args(argv)

    # When --json is on we suppress all decorative output so the JSON is
    # the only thing on stdout. Errors still print to stderr via the
    # console.print(file=sys.stderr) below.
    if args.json:
        global console
        console = Console(stderr=True)

    envs_root = Path(args.envs_root).resolve()
    try:
        env_dir = resolve_env_dir(args.env, envs_root)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return 2

    if not (env_dir / "benchmark.py").is_file():
        console.print(
            f"[red]{env_dir} is missing benchmark.py — rebuild envs with "
            f"`python -m kernel_synth.scripts.build_envs`.[/red]"
        )
        return 3

    if not args.json:
        console.print(
            Panel.fit(
                f"[bold]{env_dir.name}[/bold]  ·  mode=[cyan]{args.mode}[/cyan]"
                f"  ·  runs={args.runs}",
                border_style="magenta",
            )
        )

    if args.seed is not None:
        # Seed our own process for any in-process Python randomness, and
        # propagate so the benchmark subprocess (which re-imports torch and
        # builds the module) picks the same seed via inputs.py / benchmark.py.
        try:
            import torch

            torch.manual_seed(int(args.seed))
        except Exception:  # noqa: BLE001
            pass
        os.environ["KERNEL_SYNTH_SEED"] = str(int(args.seed))
        console.print(f"[dim]seed:[/dim] [magenta]{args.seed}[/magenta]")

    llm = None
    if args.mode == "agent":
        try:
            from kernel_synth.llm import LLMClient
            llm = LLMClient()
        except Exception as e:  # noqa: BLE001
            console.print(
                f"[red]agent mode requires a configured LLM ({e}). "
                f"Set ANTHROPIC_API_KEY or OPENAI_API_KEY.[/red]"
            )
            return 4

    result = rollout(
        env_dir,
        mode=cast(RolloutMode, args.mode),
        llm=llm,
        max_steps=args.max_steps,
        model_label=args.model,
        final_runs=args.runs,
        temperature=args.temperature,
    )

    ok, errs = validate(result.trace_path)

    if args.json:
        payload = {
            "env": env_dir.name,
            "mode": result.mode,
            "reward": result.reward,
            "components": result.components,
            "trace_path": str(result.trace_path),
            "atif_ok": ok,
            "atif_errors": errs,
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0 if ok else 1

    table = Table(title="reward components", show_header=False, expand=False)
    table.add_column("k", style="cyan")
    table.add_column("v", style="bold")
    table.add_row("mode", result.mode)
    table.add_row("reward", f"{result.reward:.3f}")
    for k in (
        "correct", "eager_ms", "compile_ms", "solution_ms",
        "progress", "eager_speedup", "compile_ratio",
    ):
        v = result.components.get(k)
        table.add_row(k, _fmt(v))

    console.print(table)
    console.print(
        f"[dim]trace:[/dim] [green]{result.trace_path}[/green]"
    )

    if ok:
        console.print("[green]ATIF validate: OK[/green]")
    else:
        console.print("[red]ATIF validate: FAILED[/red]")
        for e in errs[:8]:
            console.print(f"  · {e}")
    return 0 if ok else 1


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


if __name__ == "__main__":
    sys.exit(main())
