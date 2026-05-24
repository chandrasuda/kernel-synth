# CS 224R Milestone Report

One-page milestone report for CS 224R (Deep Reinforcement Learning, Stanford).

- `milestone.tex` — source.
- `milestone.pdf` — compiled output (1 page).

## Build

Recommended:

```bash
tectonic milestone.tex
```

Falls back to:

```bash
latexmk -pdf milestone.tex          # uses pdflatex
# or
pdflatex -interaction=nonstopmode milestone.tex && pdflatex -interaction=nonstopmode milestone.tex
```

The preamble matches the original proposal (`10pt`, `geometry` 0.72in margins, `times`,
`hyperref`). Build artifacts (`*.log`, `*.aux`, `*.out`, `*.xdv`) are gitignored.
