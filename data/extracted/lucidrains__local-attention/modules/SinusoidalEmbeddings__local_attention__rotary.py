# Extracted by kernel-synth
# Source: local_attention/rotary.py (lines 11-48)
# Class: SinusoidalEmbeddings
# Tags: einsum, imports:einops, looks-generic, self-contained, uses-buffers
# Novelty: 0.33
# Reason: forward uses einsum; file imports einops

class SinusoidalEmbeddings(Module):
    def __init__(
        self,
        dim,
        scale_base = None,
        use_xpos = False,
        theta = 10000
    ):
        super().__init__()
        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # xpos related

        self.use_xpos = use_xpos
        self.scale_base = scale_base

        assert not (use_xpos and not exists(scale_base)), 'scale base must be defined if using xpos'

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale, persistent = False)

    @autocast('cuda', enabled = False)
    def forward(self, x):
        seq_len, device = x.shape[-2], x.device

        t = torch.arange(seq_len, device = x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs =  torch.cat((freqs, freqs), dim = -1)

        if not self.use_xpos:
            return freqs, torch.ones(1, device = device)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale
