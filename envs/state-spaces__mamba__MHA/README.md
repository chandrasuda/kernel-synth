# `MHA`  ·  RL kernel-engineering env

> forward uses roll, silu; ~12 arithmetic ops in forward; file imports causal_conv1d, causal_conv1d, causal_conv1d, einops, einops, flash_attn, flash_attn, flash_attn, flash_attn

**Source:** [`state-spaces/mamba`](https://github.com/state-spaces/mamba) · `mamba_ssm/modules/mha.py`
(lines 44–294)
**Novelty:** 0.96
**Tags:** `imports:causal_conv1d`, `imports:einops`, `imports:flash_attn`, `math-heavy`, `roll`, `self-contained`, `silu`
**Selection mode:** `heuristic`

## Goal

Write an implementation of `MHA` in `solution.py` that is
numerically equivalent to `reference.py` but **faster**. The harness
rewards correctness × clipped speedup.

## Signatures (inferred)

```python
MHA.__init__(embed_dim, num_heads, num_heads_kv=None, head_dim=None, mlp_dim=0, qkv_proj_bias=True, out_proj_bias=True, softmax_scale=None, causal=False, layer_idx=None, d_conv=0, rotary_emb_dim=0, rotary_emb_base=10000.0, rotary_emb_interleaved=False, device=None, dtype=None)
MHA.forward(x, inference_params=None)
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
