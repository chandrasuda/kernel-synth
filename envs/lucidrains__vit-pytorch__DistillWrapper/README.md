# `DistillWrapper`  ·  RL kernel-engineering env

> forward uses softmax; ~8 arithmetic ops in forward; file imports einops, einops, einops; non-trivial parameter init

**Source:** [`lucidrains/vit-pytorch`](https://github.com/lucidrains/vit-pytorch) · `vit_pytorch/distill.py`
(lines 105–159)
**Novelty:** 0.98
**Tags:** `custom-init`, `imports:einops`, `math-heavy`, `self-contained`, `softmax`
**Selection mode:** `heuristic`

## Goal

Write an implementation of `DistillWrapper` in `solution.py` that is
numerically equivalent to `reference.py` but **faster**. The harness
rewards correctness × clipped speedup.

## Signatures (inferred)

```python
DistillWrapper.__init__(teacher, student, temperature=1.0, alpha=0.5, hard=False, mlp_layernorm=False)
DistillWrapper.forward(img, labels, temperature=None, alpha=None, **kwargs)
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
