# Continuum — SymSys 161 final presentation

A single-page, animated research-pitch site for **Continuum** (test-time
asynchronous RL for deployed agents) and **SDPO** (our staleness-aware
off-policy update), with the LiveCodeBench results and the kernel-synth
verifiable testbed.

By Chandra Suda & Aaditya Nalawade.

## Present it

No build step, no dependencies, works offline.

```bash
# simplest — just open the file
open index.html

# or serve it (nicer for some browsers)
python3 -m http.server 8791   # then visit http://127.0.0.1:8791/
```

Navigate with **← / →** (or space, or scroll). The right-edge dots jump to
any section.

## Talk flow (~6 min)

1. **Problem** (~1 min) — agents ship frozen; knowledge work needs continual
   adaptation.
2. **Continuum** — the test-time async-RL loop (Modal sandboxes, self-graded
   vs. human-graded rollouts).
3. **SDPO** — the staleness problem and our entropy-scaled per-token clip vs.
   KL / ratio / advantage clipping. Interactive entropy plot + variance band.
4. **Results** — LiveCodeBench: accuracy held at higher staleness.
5. **Kernels** — kernel-synth as the fully-verifiable testbed.
6. **The bet** — agents that improve the more they're used.

## Editing the data

All talk numbers live at the top of `app.js` in the `RESULTS` object
(accuracy-vs-K curves, the results table). **Replace the LiveCodeBench
placeholders with the measured numbers before presenting.**

## Files

- `index.html` — narrative / structure
- `styles.css` — design system (Connectionism/Tufte paper aesthetic)
- `app.js` — nav, animations, and the interactive figures

`?cap=<section-id>` (or `?cap=all`) is a dev aid that reveals everything and
freezes animations for exporting static figures.
