# `VAAT`  ·  RL kernel-engineering env

> ~8 arithmetic ops in forward; file imports einops, einops, einops, einops, einops, einops, einops, einops

**Source:** [`lucidrains/vit-pytorch`](https://github.com/lucidrains/vit-pytorch) · `vit_pytorch/vaat.py`
(lines 421–745)
**Novelty:** 0.95
**Tags:** `imports:einops`, `math-heavy`, `self-contained`, `uses-buffers`
**Selection mode:** `heuristic`

## Goal

Write an implementation of `VAAT` in `solution.py` that is
numerically equivalent to `reference.py` but **faster**. The harness
rewards correctness × clipped speedup.

## Signatures (inferred)

```python
VAAT.__init__(vit, ast, dim, depth, heads, dim_head, dim_action, mlp_dim, num_image_views=None, num_audio_views=None, num_tasks=None, dim_extra_token=None, num_register_tokens=4, action_chunk_len=7, time_seq_len=1, dropout=0.0, add_self_attn=True, self_attn_heads=4, self_attn_dim_head=32, ast_layer_indices=None, vit_layer_indices=None, num_advantage_bins=0)
VAAT.forward(video_or_image, audio_or_spec, extra=None, tasks=None, advantages=None, actions=None, return_hiddens=False, freeze_vit=False, freeze_ast=False)
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
