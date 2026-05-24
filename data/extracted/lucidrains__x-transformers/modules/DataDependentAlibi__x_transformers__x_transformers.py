# Extracted by kernel-synth
# Source: x_transformers/x_transformers.py (lines 652-694)
# Class: DataDependentAlibi
# Tags: creative-name, cumsum, imports:einops, self-contained, tril, triu
# Novelty: 1.00
# Reason: forward uses cumsum, tril, triu; file imports einops, einops, einops, einops, einops, einops, einops, einops; name 'DataDependentAlibi' suggests a non-standard mechanism

class DataDependentAlibi(Module):
    """ https://openreview.net/forum?id=q2Lnyegkr8 """

    def __init__(
        self,
        dim,
        heads,
        causal = True,
        bias_init = 5.,
        post_log_scale = 1.,
    ):
        super().__init__()

        self.causal = causal

        linear = nn.Linear(dim, heads * (1 if causal else 2))

        self.to_forget_gates = nn.Sequential(
            linear,
            Rearrange('b n h -> b h n'),
            nn.LogSigmoid()
        )

        nn.init.constant_(linear.bias, bias_init)
        self.post_log_scale = post_log_scale

    def forward(self, x):
        bidirectional = not self.causal

        forget_gates = self.to_forget_gates(x) * self.post_log_scale

        forget_gates = forget_gates.cumsum(dim = -1)

        if bidirectional:
            forget_gates, forget_gates_reversed = forget_gates.chunk(2, dim = 1)

        forget_gates = einx.subtract('b h i, b h j -> b h i j', forget_gates, forget_gates)

        if bidirectional:
            forget_gates_reversed = einx.subtract('b h j, b h i -> b h i j', forget_gates_reversed, forget_gates_reversed)
            forget_gates = forget_gates.tril() + forget_gates_reversed.triu()

        return forget_gates
