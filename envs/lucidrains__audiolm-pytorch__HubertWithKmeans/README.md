# `HubertWithKmeans`  ·  RL kernel-engineering env

> file imports einops, einops, einops, einops, einops

**Source:** [`lucidrains/audiolm-pytorch`](https://github.com/lucidrains/audiolm-pytorch) · `audiolm_pytorch/hubert_kmeans.py`
(lines 37–121)
**Novelty:** 0.75
**Tags:** `imports:einops`, `self-contained`, `uses-buffers`
**Selection mode:** `heuristic`

## Goal

Write an implementation of `HubertWithKmeans` in `solution.py` that is
numerically equivalent to `reference.py` but **faster**. The harness
rewards correctness × clipped speedup.

## Signatures (inferred)

```python
HubertWithKmeans.__init__(checkpoint_path, kmeans_path, target_sample_hz=16000, seq_len_multiple_of=None, output_layer=9)
HubertWithKmeans.forward(wav_input, flatten=True, input_sample_hz=None)
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
