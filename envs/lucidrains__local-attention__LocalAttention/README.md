# `LocalAttention`  ·  RL kernel-engineering env

> forward uses einsum, softmax; ~18 arithmetic ops in forward; file imports einops

**Source:** [`lucidrains/local-attention`](https://github.com/lucidrains/local-attention) · `local_attention/local_attention.py`
(lines 52–242)
**Novelty:** 0.46
**Tags:** `einsum`, `imports:einops`, `looks-generic`, `math-heavy`, `self-contained`, `softmax`
**Selection mode:** `heuristic`

## Goal

Write an implementation of `LocalAttention` in `solution.py` that is
numerically equivalent to `reference.py` but **faster**. The harness
rewards correctness × clipped speedup.

## Signatures (inferred)

```python
LocalAttention.__init__(window_size, causal=False, look_backward=1, look_forward=None, dropout=0.0, shared_qk=False, rel_pos_emb_config=None, dim=None, autopad=False, exact_windowsize=False, scale=None, use_rotary_pos_emb=True, use_xpos=False, xpos_scale_base=None)
LocalAttention.forward(q, k, v, mask=None, input_mask=None, attn_bias=None, window_size=None)
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
