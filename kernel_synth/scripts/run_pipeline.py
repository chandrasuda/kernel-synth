"""Run the pipeline against one or more GitHub URLs.

Usage:
    python -m kernel_synth.scripts.run_pipeline <url> [<url> ...] [--no-llm]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from kernel_synth.pipeline import Pipeline, PipelineConfig

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urls", nargs="+", help="One or more GitHub URLs.")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the agent harness and use the AST heuristic instead.",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Root directory for clones + extracted output (default ./data).",
    )
    parser.add_argument(
        "--force-reclone",
        action="store_true",
        help="Remove and re-clone repos even if a copy already exists.",
    )
    args = parser.parse_args(argv)

    config = PipelineConfig(
        data_dir=Path(args.data_dir).resolve(),
        use_llm=not args.no_llm,
    )
    pipeline = Pipeline(config=config)

    table = Table(title="kernel-synth pipeline", expand=True)
    table.add_column("repo", style="bold")
    table.add_column("mode")
    table.add_column("files", justify="right")
    table.add_column("modules", justify="right")
    table.add_column("avg novelty", justify="right")
    table.add_column("notes", style="dim")

    rc = 0
    for url in args.urls:
        console.rule(f"[bold]{url}")
        try:
            rec = pipeline.run(url, force_reclone=args.force_reclone)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]✗ {url}: {e}[/red]")
            rc = 1
            table.add_row(url, "error", "—", "—", "—", str(e)[:64])
            continue
        avg = (
            sum(c.novelty_score for c in rec.candidates) / len(rec.candidates)
            if rec.candidates
            else 0.0
        )
        console.print(
            f"[green]✓[/green] {rec.name}  mode=[cyan]{rec.selection_mode}[/cyan]  "
            f"files={rec.n_python_files}  modules={len(rec.candidates)}  "
            f"avg_novelty={avg:.2f}"
        )
        for c in rec.candidates:
            console.print(
                f"   · [bold]{c.class_name}[/bold]  ({c.file_path})  "
                f"novelty=[magenta]{c.novelty_score:.2f}[/magenta]  "
                f"tags={','.join(c.tags) or '-'}"
            )
        table.add_row(
            rec.name,
            rec.selection_mode,
            str(rec.n_python_files),
            str(len(rec.candidates)),
            f"{avg:.2f}",
            (rec.notes or "")[:64],
        )

    console.print(table)
    console.print(
        f"\nExtracted output: [bold]{config.extracted_dir}[/bold]\n"
        f"Open the viewer:    [bold]uvicorn kernel_synth.app:app --reload[/bold]"
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
