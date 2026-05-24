# `SoundStream`  ·  RL kernel-engineering env

> ~16 arithmetic ops in forward; file imports einops, einops, einops, einops, einops

**Source:** [`lucidrains/audiolm-pytorch`](https://github.com/lucidrains/audiolm-pytorch) · `audiolm_pytorch/soundstream.py`
(lines 451–995)
**Novelty:** 0.75
**Tags:** `imports:einops`, `math-heavy`, `uses-buffers`
**Selection mode:** `heuristic`

## Goal

Write an implementation of `SoundStream` in `solution.py` that is
numerically equivalent to `reference.py` but **faster**. The harness
rewards correctness × clipped speedup.

## Signatures (inferred)

```python
SoundStream.__init__(channels=32, strides=(2, 4, 5, 8), channel_mults=(2, 4, 8, 16), codebook_dim=512, codebook_size=None, finite_scalar_quantizer_levels=None, rq_num_quantizers=8, rq_commitment_weight=1.0, rq_ema_decay=0.95, rq_quantize_dropout_multiple_of=1, rq_groups=1, rq_stochastic_sample_codes=False, rq_rotation_trick=True, rq_kwargs={}, use_lookup_free_quantizer=False, use_finite_scalar_quantizer=False, input_channels=1, discr_multi_scales=(1, 0.5, 0.25), stft_normalized=False, enc_cycle_dilations=(1, 3, 9), dec_cycle_dilations=(1, 3, 9), multi_spectral_window_powers_of_two=tuple(range(6, 12)), multi_spectral_n_ffts=512, multi_spectral_n_mels=64, recon_loss_weight=1.0, multi_spectral_recon_loss_weight=1e-05, adversarial_loss_weight=1.0, feature_loss_weight=100, quantize_dropout_cutoff_index=1, target_sample_hz=16000, use_local_attn=True, attn_window_size=128, attn_dim_head=64, attn_heads=8, attn_depth=1, attn_xpos_scale_base=None, attn_dynamic_pos_bias=False, use_gate_loop_layers=False, squeeze_excite=False, complex_stft_discr_logits_abs=True, pad_mode='reflect', stft_discriminator=None, complex_stft_discr_kwargs=dict())
SoundStream.forward(x, target=None, is_denoising=None, return_encoded=False, return_codes_only=False, return_discr_loss=False, return_discr_losses_separately=False, return_loss_breakdown=False, return_recons_only=False, input_sample_hz=None, apply_grad_penalty=False, curtail_from_left=False)
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
