# `AutoregressiveWrapper`  ·  RL kernel-engineering env

> forward uses cumsum, scatter, topk; ~12 arithmetic ops in forward; file imports einops, einops, einops, einops, einops

**Source:** [`lucidrains/x-transformers`](https://github.com/lucidrains/x-transformers) · `x_transformers/autoregressive_wrapper.py`
(lines 158–687)
**Novelty:** 0.89
**Tags:** `cumsum`, `imports:einops`, `math-heavy`, `scatter`, `topk`
**Selection mode:** `heuristic`

## Goal

Write an implementation of `AutoregressiveWrapper` in `solution.py` that is
numerically equivalent to `reference.py` but **faster**. The harness
rewards correctness × clipped speedup.

## Signatures (inferred)

```python
AutoregressiveWrapper.__init__(net, ignore_index=-100, pad_value=0, mask_prob=0.0, add_attn_z_loss=False, next_embed_loss_weight=0.1, looped_loss_threshold_exit=0.05, looped_loss_slope=50, looped_exit_loss_weight=1.0)
AutoregressiveWrapper.forward(x, return_outputs=False, prepend_embeds=None, lens=None, **kwargs)
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
