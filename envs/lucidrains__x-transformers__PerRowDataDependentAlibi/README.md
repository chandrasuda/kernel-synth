# `PerRowDataDependentAlibi`  ·  RL kernel-engineering env

> forward uses cumsum, einsum, triu; file imports einops, einops, einops, einops, einops, einops, einops, einops; name 'PerRowDataDependentAlibi' suggests a non-standard mechanism

**Source:** [`lucidrains/x-transformers`](https://github.com/lucidrains/x-transformers) · `x_transformers/x_transformers.py`
(lines 696–740)
**Novelty:** 1.00
**Tags:** `creative-name`, `cumsum`, `einsum`, `imports:einops`, `self-contained`, `triu`
**Selection mode:** `heuristic`

## Goal

Write an implementation of `PerRowDataDependentAlibi` in `solution.py` that is
numerically equivalent to `reference.py` but **faster**. The harness
rewards correctness × clipped speedup.

## Signatures (inferred)

```python
PerRowDataDependentAlibi.__init__(dim, heads, causal=True, dim_head=8, post_log_scale=1.0)
PerRowDataDependentAlibi.forward(x)
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
