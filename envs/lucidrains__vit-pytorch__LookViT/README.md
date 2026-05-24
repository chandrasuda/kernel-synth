# `LookViT`  ·  RL kernel-engineering env

> ~8 arithmetic ops in forward; file imports einops, einops, einops, einops, einops, einops, einops; non-trivial parameter init

**Source:** [`lucidrains/vit-pytorch`](https://github.com/lucidrains/vit-pytorch) · `vit_pytorch/look_vit.py`
(lines 140–255)
**Novelty:** 0.90
**Tags:** `custom-init`, `imports:einops`, `math-heavy`, `self-contained`
**Selection mode:** `heuristic`

## Goal

Write an implementation of `LookViT` in `solution.py` that is
numerically equivalent to `reference.py` but **faster**. The harness
rewards correctness × clipped speedup.

## Signatures (inferred)

```python
LookViT.__init__(dim, image_size, num_classes, depth=3, patch_size=16, heads=8, mlp_factor=4, dim_head=64, highres_patch_size=12, highres_mlp_factor=4, cross_attn_heads=8, cross_attn_dim_head=64, patch_conv_kernel_size=7, dropout=0.1, channels=3)
LookViT.forward(img)
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
