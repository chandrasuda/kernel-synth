# `Mamba3`  ·  RL kernel-engineering env

> forward uses einsum; file imports einops, einops, einops, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm; name 'Mamba3' suggests a non-standard mechanism

**Source:** [`state-spaces/mamba`](https://github.com/state-spaces/mamba) · `mamba_ssm/modules/mamba3.py`
(lines 26–498)
**Novelty:** 0.88
**Tags:** `creative-name`, `einsum`, `imports:einops`, `imports:mamba_ssm`
**Selection mode:** `heuristic`

## Goal

Write an implementation of `Mamba3` in `solution.py` that is
numerically equivalent to `reference.py` but **faster**. The harness
rewards correctness × clipped speedup.

## Signatures (inferred)

```python
Mamba3.__init__(d_model, d_state=128, expand=2, headdim=64, ngroups=1, rope_fraction=0.5, dt_min=0.001, dt_max=0.1, dt_init_floor=0.0001, A_floor=0.0001, is_outproj_norm=False, is_mimo=False, mimo_rank=4, chunk_size=64, dropout=0.0, layer_idx=None, n_layer=None, device=None, dtype=None, **kwargs)
Mamba3.forward(u, seq_idx=None, cu_seqlens=None, inference_params=None)
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
