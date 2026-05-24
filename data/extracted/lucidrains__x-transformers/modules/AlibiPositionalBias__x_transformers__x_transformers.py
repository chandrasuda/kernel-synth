# Extracted by kernel-synth
# Source: x_transformers/x_transformers.py (lines 581-650)
# Class: AlibiPositionalBias
# Tags: creative-name, imports:einops, self-contained, uses-buffers
# Novelty: 1.00
# Reason: file imports einops, einops, einops, einops, einops, einops, einops, einops; name 'AlibiPositionalBias' suggests a non-standard mechanism

class AlibiPositionalBias(Module):
    def __init__(
        self,
        heads,
        total_heads = None,
        slopes: list[int] | None = None,
        **kwargs
    ):
        super().__init__()
        self.heads = heads
        self.total_heads = default(total_heads, heads)

        slopes = Tensor(default(slopes, self._get_slopes(heads)))
        slopes = rearrange(slopes, 'h -> h 1 1')

        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward_custom_pos(
        self,
        pos_i: Tensor,
        pos_j: Tensor | None = None
    ):
        h, device = self.total_heads, self.device

        pos_j = default(pos_j, pos_i)
        bias = -einx.subtract('... j, ... i -> ... i j', pos_j, pos_i).abs()

        if bias.ndim == 3:
            bias = rearrange(bias, 'b i j -> b 1 i j')

        bias = bias * self.slopes
        num_heads_unalibied = h - bias.shape[-3]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim = -3)

        return bias

    def forward(self, i, j):
        h, device = self.total_heads, self.device

        if exists(self.bias) and self.bias.shape[-1] >= j and self.bias.shape[-2] >= i:
            return self.bias[..., -i:, -j:]

        seq_arange = arange(j - i, j, device = device)
        context_arange = arange(j, device = device)
        bias = -einx.subtract('j, i -> 1 i j', context_arange, seq_arange).abs()

        bias = bias * self.slopes
        num_heads_unalibied = h - bias.shape[-3]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim = -3)

        self.register_buffer('bias', bias, persistent = False)
        return self.bias
