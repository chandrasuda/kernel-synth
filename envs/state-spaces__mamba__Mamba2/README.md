# `Mamba2`  ·  RL kernel-engineering env

> forward uses silu; ~17 arithmetic ops in forward; file imports causal_conv1d, causal_conv1d, causal_conv1d, causal_conv1d, causal_conv1d, einops, einops, einops, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm; name 'Mamba2' suggests a non-standard mechanism

**Source:** [`state-spaces/mamba`](https://github.com/state-spaces/mamba) · `mamba_ssm/modules/mamba2.py`
(lines 37–383)
**Novelty:** 1.00
**Tags:** `creative-name`, `imports:causal_conv1d`, `imports:einops`, `imports:mamba_ssm`, `math-heavy`, `self-contained`, `silu`
**Selection mode:** `heuristic`

## Goal

Write an implementation of `Mamba2` in `solution.py` that is
numerically equivalent to `reference.py` but **faster**. The harness
rewards correctness × clipped speedup.

## Signatures (inferred)

```python
Mamba2.__init__(d_model, d_state=128, d_conv=4, conv_init=None, expand=2, headdim=64, d_ssm=None, ngroups=1, A_init_range=(1, 16), D_has_hdim=False, rmsnorm=True, norm_before_gate=False, dt_min=0.001, dt_max=0.1, dt_init_floor=0.0001, dt_limit=(0.0, float('inf')), bias=False, conv_bias=True, chunk_size=256, use_mem_eff_path=True, layer_idx=None, process_group=None, sequence_parallel=True, device=None, dtype=None)
Mamba2.forward(u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None)
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
