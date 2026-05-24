# `DSSA`  ·  RL kernel-engineering env

> forward uses einsum; ~15 arithmetic ops in forward; file imports einops, einops, einops, einops, einops, einops; non-trivial parameter init

**Source:** [`lucidrains/vit-pytorch`](https://github.com/lucidrains/vit-pytorch) · `vit_pytorch/sep_vit.py`
(lines 65–206)
**Novelty:** 0.98
**Tags:** `custom-init`, `einsum`, `imports:einops`, `math-heavy`, `self-contained`
**Selection mode:** `heuristic`

## Goal

Write an implementation of `DSSA` in `solution.py` that is
numerically equivalent to `reference.py` but **faster**. The harness
rewards correctness × clipped speedup.

## Signatures (inferred)

```python
DSSA.__init__(dim, heads=8, dim_head=32, dropout=0.0, window_size=7)
DSSA.forward(x)
```

## Files

- `reference.py` — frozen original implementation (do not edit)
- `inputs.py` — input generators (**edit if defaults don't fit**)
- `solution.py` — your implementation
- `harness.py` — eval loop
- `env.json` — machine-readable metadata

## Run

```bash
# one-off
python harness.py

# machine-readable
python harness.py --json
```

## Reward

```
reward = float(correct) * clamp(speedup, 0, 10) / 10
```

Where:
- `correct = allclose(ref_out, cand_out, rtol=1e-3, atol=1e-4)`
- `speedup = ref_latency / cand_latency`

## Notes

- The reference imports were lifted verbatim from the source file. If
  your environment is missing a source-repo dep (e.g. `mamba_ssm`,
  `causal_conv1d`), `pip install` it or stub the missing names.
- Default shapes are deliberately small so the harness runs on CPU.
  Increase `BATCH` / `SEQ_LEN` / `HIDDEN` in `inputs.py` for a
  meaningful benchmark on GPU.
