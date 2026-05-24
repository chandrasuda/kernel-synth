# kernel-synth

A synthesizer that turns real GitHub repositories into a buffer of candidate
**RL environments for custom-kernel engineering**.

The premise: it is now easy to manufacture RL environments. Standard kernels
(plain attention, FFNs, Conv2d) are already well-served. The interesting
problem is *diversity* — finding the long tail of bespoke PyTorch modules
where a new kernel would actually matter.

This project does the first leg of that synthesis:

1. **Clone** a public GitHub repository (read-only).
2. **Ingest** every `.py` file into an in-memory `CodeBuffer`.
3. **Run an agent harness** (Claude tool-use loop) over the buffer. The agent
   navigates the repo with `list_files` / `read_file` and emits `mark_module`
   calls for unique `nn.Module` subclasses worth turning into RL tasks.
4. **Extract** the chosen modules to `data/extracted/<repo>/`, alongside a
   `manifest.json` that records *why* each module was picked and a
   per-module novelty score.
5. **Visualize** the buffer locally with a FastAPI app
   (`http://127.0.0.1:8000`) — dark, glassy, and made to be browsed.

## Constraints honored

- The cloner only ever **clones**. It never pushes, opens PRs, or otherwise
  mutates remote state.
- Nothing is shipped to Docker / sandboxes yet — extraction is local-only,
  as requested.

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e .

# (optional) put a key in .env for full agent mode
cp .env.example .env

# Run the pipeline on the 5 seeded semi-popular repos
python -m kernel_synth.scripts.seed_repos

# Browse the result
uvicorn kernel_synth.app:app --reload
```

Or one-off:

```bash
python -m kernel_synth.scripts.run_pipeline https://github.com/state-spaces/mamba
```

## Layout

```
kernel_synth/
  cloner.py        # shallow git clone into data/clones/
  code_buffer.py   # walk + index .py files
  llm.py           # Anthropic / OpenAI client wrapper
  harness.py       # tool-use agent loop
  heuristics.py    # AST fallback when no API key is set
  extractor.py     # write selected modules + manifest
  pipeline.py      # glue
  app.py           # FastAPI viewer
  static/          # frontend
scripts/
  run_pipeline.py
  seed_repos.py
data/
  clones/          # cloned repos (gitignored)
  extracted/       # selected modules + manifests
```
