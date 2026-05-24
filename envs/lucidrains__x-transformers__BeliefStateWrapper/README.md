# `BeliefStateWrapper`  ·  RL kernel-engineering env

> ~13 arithmetic ops in forward; file imports einops, einops, einops, einops, einops, einops, einops

**Source:** [`lucidrains/x-transformers`](https://github.com/lucidrains/x-transformers) · `x_transformers/belief_state_wrapper.py`
(lines 79–432)
**Novelty:** 0.90
**Tags:** `imports:einops`, `math-heavy`, `self-contained`, `uses-buffers`
**Selection mode:** `heuristic`

## Goal

Write an implementation of `BeliefStateWrapper` in `solution.py` that is
numerically equivalent to `reference.py` but **faster**. The harness
rewards correctness × clipped speedup.

## Signatures (inferred)

```python
BeliefStateWrapper.__init__(forward_decoder, backward_decoder=None, train_frac_forward_backward_pairs=1.0, text_head=None, backward_ar_loss_weight=1.0, pred_distance=False, pred_distance_loss_weight=1.0, cond_on_distance=False, cond_on_distance_prob=0.5, max_pred_distance=None)
BeliefStateWrapper.forward(seq, lens=None, loss_weight_by_fb_indices=None)
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
