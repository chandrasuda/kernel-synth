# `FineTransformer`  ·  RL kernel-engineering env

> forward uses einsum; ~26 arithmetic ops in forward; file imports einops, einops, einops, einops, einops, einops; non-trivial parameter init

**Source:** [`lucidrains/audiolm-pytorch`](https://github.com/lucidrains/audiolm-pytorch) · `audiolm_pytorch/audiolm_pytorch.py`
(lines 992–1368)
**Novelty:** 0.78
**Tags:** `custom-init`, `einsum`, `imports:einops`, `looks-generic`, `math-heavy`, `self-contained`
**Selection mode:** `heuristic`

## Goal

Write an implementation of `FineTransformer` in `solution.py` that is
numerically equivalent to `reference.py` but **faster**. The harness
rewards correctness × clipped speedup.

## Signatures (inferred)

```python
FineTransformer.__init__(num_coarse_quantizers, num_fine_quantizers, codebook_size, dim, depth, heads=8, attn_dropout=0.0, ff_dropout=0.0, t5_name=DEFAULT_T5_NAME, has_condition=False, cond_dim=None, audio_text_condition=False, cond_as_self_attn_prefix=False, cond_drop_prob=0.5, grad_shrink_alpha=0.1, project_coarse_logits=True, pad_id=-1, rel_pos_bias=True, flash_attn=False, **kwargs)
FineTransformer.forward(coarse_token_ids, fine_token_ids, text=None, text_embeds=None, cond_drop_prob=None, self_attn_mask=None, kv_cache=None, embed_cache=None, return_cache=False, return_only_fine_logits=False)
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
