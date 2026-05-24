# `LocalTransformer`  ·  RL kernel-engineering env

> file imports einops

**Source:** [`lucidrains/local-attention`](https://github.com/lucidrains/local-attention) · `local_attention/transformer.py`
(lines 242–443)
**Novelty:** 0.15
**Tags:** `imports:einops`, `looks-generic`, `self-contained`
**Selection mode:** `heuristic`

## Goal

Write an implementation of `LocalTransformer` in `solution.py` that is
numerically equivalent to `reference.py` but **faster**. The harness
rewards correctness × clipped speedup.

## Signatures (inferred)

```python
LocalTransformer.__init__(num_tokens, max_seq_len, dim, depth, causal=True, local_attn_window_size=512, dim_head=64, heads=8, ff_mult=4, attn_dropout=0.0, ff_dropout=0.0, ignore_index=-1, use_xpos=False, xpos_scale_base=None, use_dynamic_pos_bias=False, global_attn_layer=None, layers_insert_global_attn=None, num_residual_streams=4, **kwargs)
LocalTransformer.forward(x, mask=None, cache=None, return_loss=False, return_cache=False)
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
