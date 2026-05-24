# kernel-synth · RL environments

Auto-generated from extracted PyTorch modules. Each subfolder is a
**self-contained kernel-engineering task**: write an optimized
implementation of the reference module and beat it on latency while
staying numerically equivalent.

- **Envs:** 41
- **Source repos:** 6

## Layout per env

```
<env_name>/
  README.md       # task spec
  env.json        # metadata
  reference.py    # original module (with imports from its source file)
  inputs.py       # input generators (EDIT for your shapes/dtypes)
  solution.py     # your starting point — edit this
  harness.py      # python harness.py --json  ->  reward JSON
```

## Reward

```
reward = is_correct * clamp(speedup, 0, 10) / 10
```

## Running

```bash
cd <env_name>
pip install -r ../../requirements-env.txt   # torch + the source repo
python harness.py --json
```

## Envs

| class | repo | novelty | tags |
|-------|------|--------:|------|
| [`Attend`](./lucidrains__audiolm-pytorch__Attend) | `lucidrains/audiolm-pytorch` | 0.99 | einsum, imports:einops, self-contained, softmax, triu, uses-buffers |
| [`CoarseTransformer`](./lucidrains__audiolm-pytorch__CoarseTransformer) | `lucidrains/audiolm-pytorch` | 0.78 | custom-init, einsum, imports:einops, looks-generic, math-heavy, self-contained |
| [`FineTransformer`](./lucidrains__audiolm-pytorch__FineTransformer) | `lucidrains/audiolm-pytorch` | 0.78 | custom-init, einsum, imports:einops, looks-generic, math-heavy, self-contained |
| [`SoundStream`](./lucidrains__audiolm-pytorch__SoundStream) | `lucidrains/audiolm-pytorch` | 0.75 | imports:einops, math-heavy, uses-buffers |
| [`HubertWithKmeans`](./lucidrains__audiolm-pytorch__HubertWithKmeans) | `lucidrains/audiolm-pytorch` | 0.75 | imports:einops, self-contained, uses-buffers |
| [`RelativePositionBias`](./lucidrains__audiolm-pytorch__RelativePositionBias) | `lucidrains/audiolm-pytorch` | 0.65 | imports:einops, self-contained |
| [`AudioLM`](./lucidrains__audiolm-pytorch__AudioLM) | `lucidrains/audiolm-pytorch` | 0.65 | imports:einops, self-contained |
| [`MultiScaleDiscriminator`](./lucidrains__audiolm-pytorch__MultiScaleDiscriminator) | `lucidrains/audiolm-pytorch` | 0.65 | imports:einops, self-contained |
| [`LocalMHA`](./lucidrains__local-attention__LocalMHA) | `lucidrains/local-attention` | 0.71 | einsum, imports:einops, math-heavy, self-contained, softmax |
| [`LocalAttention`](./lucidrains__local-attention__LocalAttention) | `lucidrains/local-attention` | 0.46 | einsum, imports:einops, looks-generic, math-heavy, self-contained, softmax |
| [`DynamicPositionBias`](./lucidrains__local-attention__DynamicPositionBias) | `lucidrains/local-attention` | 0.40 | imports:einops, self-contained |
| [`SinusoidalEmbeddings`](./lucidrains__local-attention__SinusoidalEmbeddings) | `lucidrains/local-attention` | 0.33 | einsum, imports:einops, looks-generic, self-contained, uses-buffers |
| [`LocalTransformer`](./lucidrains__local-attention__LocalTransformer) | `lucidrains/local-attention` | 0.15 | imports:einops, looks-generic, self-contained |
| [`GEGLU`](./lucidrains__local-attention__GEGLU) | `lucidrains/local-attention` | 0.13 | gelu, imports:einops |
| [`DSSA`](./lucidrains__vit-pytorch__DSSA) | `lucidrains/vit-pytorch` | 0.98 | custom-init, einsum, imports:einops, math-heavy, self-contained |
| [`NaViT`](./lucidrains__vit-pytorch__NaViT) | `lucidrains/vit-pytorch` | 0.98 | custom-init, imports:einops, math-heavy, self-contained, topk |
| [`DistillWrapper`](./lucidrains__vit-pytorch__DistillWrapper) | `lucidrains/vit-pytorch` | 0.98 | custom-init, imports:einops, math-heavy, self-contained, softmax |
| [`VAAT`](./lucidrains__vit-pytorch__VAAT) | `lucidrains/vit-pytorch` | 0.95 | imports:einops, math-heavy, self-contained, uses-buffers |
| [`LookViT`](./lucidrains__vit-pytorch__LookViT) | `lucidrains/vit-pytorch` | 0.90 | custom-init, imports:einops, math-heavy, self-contained |
| [`NaViT`](./lucidrains__vit-pytorch__NaViT) | `lucidrains/vit-pytorch` | 0.88 | custom-init, imports:einops, self-contained, topk |
| [`Adapter`](./lucidrains__vit-pytorch__Adapter) | `lucidrains/vit-pytorch` | 0.85 | custom-init, imports:einops, self-contained, uses-buffers |
| [`ViT`](./lucidrains__vit-pytorch__ViT) | `lucidrains/vit-pytorch` | 0.85 | custom-init, imports:einops, self-contained, uses-buffers |
| [`CoPE`](./lucidrains__x-transformers__CoPE) | `lucidrains/x-transformers` | 1.00 | cumsum, einsum, gather, imports:einops, math-heavy, self-contained, softmax, uses-buffers |
| [`DataDependentAlibi`](./lucidrains__x-transformers__DataDependentAlibi) | `lucidrains/x-transformers` | 1.00 | creative-name, cumsum, imports:einops, self-contained, tril, triu |
| [`PerRowDataDependentAlibi`](./lucidrains__x-transformers__PerRowDataDependentAlibi) | `lucidrains/x-transformers` | 1.00 | creative-name, cumsum, einsum, imports:einops, self-contained, triu |
| [`AlibiPositionalBias`](./lucidrains__x-transformers__AlibiPositionalBias) | `lucidrains/x-transformers` | 1.00 | creative-name, imports:einops, self-contained, uses-buffers |
| [`BinaryMapper`](./lucidrains__x-transformers__BinaryMapper) | `lucidrains/x-transformers` | 0.98 | einsum, imports:einops, math-heavy, self-contained, uses-buffers |
| [`BeliefStateWrapper`](./lucidrains__x-transformers__BeliefStateWrapper) | `lucidrains/x-transformers` | 0.90 | imports:einops, math-heavy, self-contained, uses-buffers |
| [`AutoregressiveWrapper`](./lucidrains__x-transformers__AutoregressiveWrapper) | `lucidrains/x-transformers` | 0.89 | cumsum, imports:einops, math-heavy, scatter, topk |
| [`HyperConnection`](./lucidrains__x-transformers__HyperConnection) | `lucidrains/x-transformers` | 0.88 | custom-init, einsum, imports:einops, self-contained |
| [`Whisper`](./openai__whisper__Whisper) | `openai/whisper` | 0.35 | self-contained, uses-buffers |
| [`AudioEncoder`](./openai__whisper__AudioEncoder) | `openai/whisper` | 0.18 | gelu, looks-generic, self-contained, uses-buffers |
| [`TextDecoder`](./openai__whisper__TextDecoder) | `openai/whisper` | 0.10 | looks-generic, self-contained, uses-buffers |
| [`Mamba2`](./state-spaces__mamba__Mamba2) | `state-spaces/mamba` | 1.00 | creative-name, imports:causal_conv1d, imports:einops, imports:mamba_ssm, math-heavy, self-contained, silu |
| [`MHA`](./state-spaces__mamba__MHA) | `state-spaces/mamba` | 0.96 | imports:causal_conv1d, imports:einops, imports:flash_attn, math-heavy, roll, self-contained, silu |
| [`Mamba2Simple`](./state-spaces__mamba__Mamba2Simple) | `state-spaces/mamba` | 0.95 | creative-name, imports:causal_conv1d, imports:einops, imports:mamba_ssm, self-contained |
| [`MambaLMHeadModel`](./state-spaces__mamba__MambaLMHeadModel) | `state-spaces/mamba` | 0.90 | creative-name, imports:mamba_ssm, self-contained |
| [`Mamba`](./state-spaces__mamba__Mamba) | `state-spaces/mamba` | 0.90 | creative-name, imports:causal_conv1d, imports:einops, imports:mamba_ssm, self-contained |
| [`Mamba3`](./state-spaces__mamba__Mamba3) | `state-spaces/mamba` | 0.88 | creative-name, einsum, imports:einops, imports:mamba_ssm |
| [`MixerModel`](./state-spaces__mamba__MixerModel) | `state-spaces/mamba` | 0.65 | imports:mamba_ssm, self-contained |
| [`ParallelEmbeddings`](./state-spaces__mamba__ParallelEmbeddings) | `state-spaces/mamba` | 0.40 | imports:einops, imports:mamba_ssm, looks-generic, self-contained |
