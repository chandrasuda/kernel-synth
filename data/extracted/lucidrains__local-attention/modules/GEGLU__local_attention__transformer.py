# Extracted by kernel-synth
# Source: local_attention/transformer.py (lines 186-189)
# Class: GEGLU
# Tags: gelu, imports:einops
# Novelty: 0.13
# Reason: forward uses gelu; file imports einops

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)
