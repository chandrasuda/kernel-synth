# `Mamba2Simple`  ·  RL kernel-engineering env

> file imports causal_conv1d, causal_conv1d, einops, einops, einops, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm; name 'Mamba2Simple' suggests a non-standard mechanism

**Source:** [`state-spaces/mamba`](https://github.com/state-spaces/mamba) · `mamba_ssm/modules/mamba2_simple.py`
(lines 24–200)
**Novelty:** 0.95
**Tags:** `creative-name`, `imports:causal_conv1d`, `imports:einops`, `imports:mamba_ssm`, `self-contained`
**Selection mode:** `heuristic`

## Goal

Write an implementation of `Mamba2Simple` in `solution.py` that is
numerically equivalent to `reference.py` but **faster**. The harness
rewards correctness × clipped speedup.

## Signatures (inferred)

```python
Mamba2Simple.__init__(d_model, d_state=64, d_conv=4, conv_init=None, expand=2, headdim=128, ngroups=1, A_init_range=(1, 16), dt_min=0.001, dt_max=0.1, dt_init_floor=0.0001, dt_limit=(0.0, float('inf')), learnable_init_states=False, activation='swish', bias=False, conv_bias=True, chunk_size=256, use_mem_eff_path=True, layer_idx=None, device=None, dtype=None)
Mamba2Simple.forward(u, seq_idx=None)
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
