"""Build RL env folders from everything currently in ``data/extracted``.

Usage:
    python -m kernel_synth.scripts.build_envs              # build all
    python -m kernel_synth.scripts.build_envs --out ./envs # custom output dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from kernel_synth.env_builder import build_all, env_name
from kernel_synth.extractor import load_all

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extracted-dir",
        default="./data/extracted",
        help="Where extracted manifests live (default ./data/extracted).",
    )
    parser.add_argument(
        "--out",
        default="./envs",
        help="Output root for RL env folders (default ./envs).",
    )
    args = parser.parse_args(argv)

    extracted_root = Path(args.extracted_dir).resolve()
    out_root = Path(args.out).resolve()

    records = load_all(extracted_root)
    if not records:
        console.print(
            f"[red]No manifests found under {extracted_root}. "
            f"Run `python -m kernel_synth.scripts.seed_repos` first.[/red]"
        )
        return 2

    console.print(
        f"Building envs from [bold]{len(records)}[/bold] repos into "
        f"[bold]{out_root}[/bold]…"
    )
    paths = build_all(records, envs_root=out_root)

    table = Table(title="kernel-synth RL envs", expand=True)
    table.add_column("env name", style="bold")
    table.add_column("repo")
    table.add_column("novelty", justify="right")
    table.add_column("tags", style="dim")

    n_env_built = 0
    for record in records:
        for cand in record.candidates:
            n_env_built += 1
            table.add_row(
                env_name(record, cand),
                record.name,
                f"{cand.novelty_score:.2f}",
                ", ".join(cand.tags[:4]),
            )

    console.print(table)
    console.print(
        f"\n[green]✓[/green] {n_env_built} envs written to [bold]{out_root}[/bold]"
        f"\nIndex: [bold]{out_root / 'README.md'}[/bold]"
        f"\nTry one:  [bold]cd {out_root.relative_to(Path.cwd()) if out_root.is_relative_to(Path.cwd()) else out_root}/{env_name(records[0], records[0].candidates[0])} && python harness.py[/bold]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
