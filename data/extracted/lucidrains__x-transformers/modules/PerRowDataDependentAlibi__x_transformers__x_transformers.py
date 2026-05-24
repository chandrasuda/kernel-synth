# Extracted by kernel-synth
# Source: x_transformers/x_transformers.py (lines 696-740)
# Class: PerRowDataDependentAlibi
# Tags: creative-name, cumsum, einsum, imports:einops, self-contained, triu
# Novelty: 1.00
# Reason: forward uses cumsum, einsum, triu; file imports einops, einops, einops, einops, einops, einops, einops, einops; name 'PerRowDataDependentAlibi' suggests a non-standard mechanism

class PerRowDataDependentAlibi(Module):
    """ same as data dependent alibi from forgetting transformer, but the forgetting gates are also derived by a queries and keys with a small head dimension """

    def __init__(
        self,
        dim,
        heads,
        causal = True,
        dim_head = 8,
        post_log_scale = 1.
    ):
        super().__init__()
        assert causal, 'bidirectional not supported yet'

        self.scale = dim_head ** -0.5

        linear = nn.Linear(dim, heads * dim_head * 2, bias = False)

        self.to_forget_gates = nn.Sequential(
            linear,
            Rearrange('b n (qk h d) -> qk b h n d', qk = 2, d = dim_head)
        )

        self.post_log_scale = post_log_scale

    def forward(self, x):
        q, k = self.to_forget_gates(x)
        forget_gates = einsum('... i d, ... j d -> ... i j', q, k) * self.scale

        forget_gates = F.logsigmoid(forget_gates) * self.post_log_scale

        # mask out upper triangle + diagonal

        n = x.shape[-2]
        causal_mask = torch.ones((n, n), dtype = torch.bool, device = x.device).triu()

        forget_gates = forget_gates.masked_fill(causal_mask, 0.)

        # reverse cumsum

        forget_gates = forget_gates.flip(dims = (-1,))
        forget_gates = forget_gates.cumsum(dim = -1)
        forget_gates = forget_gates.flip(dims = (-1,))

        return forget_gates
