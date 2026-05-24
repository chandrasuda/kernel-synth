# `Mamba`  ·  RL kernel-engineering env

> file imports causal_conv1d, causal_conv1d, causal_conv1d, einops, einops, einops, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm; name 'Mamba' suggests a non-standard mechanism

**Source:** [`state-spaces/mamba`](https://github.com/state-spaces/mamba) · `mamba_ssm/modules/mamba_simple.py`
(lines 31–294)
**Novelty:** 0.90
**Tags:** `creative-name`, `imports:causal_conv1d`, `imports:einops`, `imports:mamba_ssm`, `self-contained`
**Selection mode:** `heuristic`

## Goal

Write an implementation of `Mamba` in `solution.py` that is
numerically equivalent to `reference.py` but **faster**. The harness
rewards correctness × clipped speedup.

## Signatures (inferred)

```python
Mamba.__init__(d_model, d_state=16, d_conv=4, expand=2, dt_rank='auto', dt_min=0.001, dt_max=0.1, dt_init='random', dt_scale=1.0, dt_init_floor=0.0001, conv_bias=True, bias=False, use_fast_path=True, layer_idx=None, device=None, dtype=None)
Mamba.forward(hidden_states, inference_params=None)
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
