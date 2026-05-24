# `MixerModel`  ·  RL kernel-engineering env

> file imports mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm, mamba_ssm

**Source:** [`state-spaces/mamba`](https://github.com/state-spaces/mamba) · `mamba_ssm/models/mixer_seq_simple.py`
(lines 118–212)
**Novelty:** 0.65
**Tags:** `imports:mamba_ssm`, `self-contained`
**Selection mode:** `heuristic`

## Goal

Write an implementation of `MixerModel` in `solution.py` that is
numerically equivalent to `reference.py` but **faster**. The harness
rewards correctness × clipped speedup.

## Signatures (inferred)

```python
MixerModel.__init__(d_model, n_layer, d_intermediate, vocab_size, ssm_cfg=None, attn_layer_idx=None, attn_cfg=None, norm_epsilon=1e-05, rms_norm=False, initializer_cfg=None, fused_add_norm=False, residual_in_fp32=False, device=None, dtype=None)
MixerModel.forward(input_ids, inference_params=None, **kwargs)
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
