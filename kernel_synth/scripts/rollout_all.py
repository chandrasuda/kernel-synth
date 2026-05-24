"""Run ``baseline`` rollouts across every env folder and print a leaderboard.

Usage:
    python -m kernel_synth.scripts.rollout_all
    python -m kernel_synth.scripts.rollout_all --envs-root ./envs --runs 10

This is useful for sanity-checking that every env can at least build the
reference, run the eager + torch.compile baselines, and produce a valid
trajectory file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from kernel_synth.rl import rollout, validate

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--envs-root",
        default="./envs",
        help="Where the env folders live (default ./envs).",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "torch_compile"],
        default="baseline",
        help="Use baseline (default) or the torch.compile shim.",
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only run the first N envs (0 = all).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Per-env benchmark subprocess timeout (s).",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help=(
            "Optional regex; only env folder names matching it are run. "
            "Useful for re-trying a single failing env without disturbing "
            "the leaderboard view."
        ),
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help=(
            "Optional path to write the full leaderboard rows as JSON "
            "(in addition to printing the rich table)."
        ),
    )
    args = parser.parse_args(argv)

    envs_root = Path(args.envs_root).resolve()
    if not envs_root.is_dir():
        console.print(f"[red]envs root not found: {envs_root}[/red]")
        return 2

    envs = sorted(
        d for d in envs_root.iterdir()
        if d.is_dir() and (d / "benchmark.py").is_file()
    )
    if args.filter:
        try:
            pattern = re.compile(args.filter)
        except re.error as e:
            console.print(f"[red]invalid --filter regex: {e}[/red]")
            return 4
        envs = [d for d in envs if pattern.search(d.name)]
    if args.limit > 0:
        envs = envs[: args.limit]

    if not envs:
        console.print(
            f"[red]no envs with benchmark.py under {envs_root}. "
            f"Rebuild envs first.[/red]"
        )
        return 3

    rows: list[dict] = []
    for env_dir in envs:
        t0 = time.perf_counter()
        try:
            result = rollout(
                env_dir,
                mode=args.mode,
                final_runs=args.runs,
                benchmark_timeout_s=args.timeout,
            )
            ok, _ = validate(result.trace_path)
            rows.append(
                {
                    "env": env_dir.name,
                    "reward": result.reward,
                    "correct": result.components.get("correct"),
                    "eager_ms": result.components.get("eager_ms"),
                    "compile_ms": result.components.get("compile_ms"),
                    "solution_ms": result.components.get("solution_ms"),
                    "trace": result.trace_path,
                    "ok": ok,
                    "error": None,
                    "elapsed_s": time.perf_counter() - t0,
                }
            )
        except Exception as e:  # noqa: BLE001
            rows.append(
                {
                    "env": env_dir.name,
                    "reward": float("nan"),
                    "correct": None,
                    "eager_ms": None,
                    "compile_ms": None,
                    "solution_ms": None,
                    "trace": None,
                    "ok": False,
                    "error": repr(e),
                    "elapsed_s": time.perf_counter() - t0,
                }
            )
        console.print(
            f"  [{'green' if rows[-1]['ok'] else 'red'}]·[/] "
            f"{env_dir.name}  reward={_fmt(rows[-1]['reward'])}"
        )

    rows.sort(key=lambda r: (-(r["reward"] if r["reward"] == r["reward"] else -1e9)))

    table = Table(title=f"kernel-synth · {args.mode} leaderboard")
    table.add_column("env", style="bold")
    table.add_column("reward", justify="right")
    table.add_column("correct", justify="center")
    table.add_column("eager ms", justify="right")
    table.add_column("compile ms", justify="right")
    table.add_column("solution ms", justify="right")
    table.add_column("trace ok", justify="center")
    table.add_column("err", style="red")

    for r in rows:
        table.add_row(
            r["env"],
            _fmt(r["reward"]),
            _bool(r["correct"]),
            _fmt(r["eager_ms"]),
            _fmt(r["compile_ms"]),
            _fmt(r["solution_ms"]),
            _bool(r["ok"]),
            (r["error"] or "")[:80],
        )

    console.print(table)
    n_ok = sum(1 for r in rows if r["ok"])
    console.print(
        f"\n[green]{n_ok}[/green]/{len(rows)} envs produced a valid trajectory."
    )

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        console.print(f"[dim]wrote leaderboard to[/dim] [green]{out_path}[/green]")
    return 0 if n_ok == len(rows) else 1


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:  # NaN
            return "—"
        return f"{v:.3f}"
    return str(v)


def _bool(v) -> str:
    if v is None:
        return "—"
    return "✓" if v else "✗"


if __name__ == "__main__":
    sys.exit(main())
