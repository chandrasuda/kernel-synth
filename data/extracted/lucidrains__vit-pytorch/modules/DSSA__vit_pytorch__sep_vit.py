# Extracted by kernel-synth
# Source: vit_pytorch/sep_vit.py (lines 65-206)
# Class: DSSA
# Tags: custom-init, einsum, imports:einops, math-heavy, self-contained
# Novelty: 0.98
# Reason: forward uses einsum; ~15 arithmetic ops in forward; file imports einops, einops, einops, einops, einops, einops; non-trivial parameter init

class DSSA(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        inner_dim = dim_head * heads

        self.norm = ChanLayerNorm(dim)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias = False)

        # window tokens

        self.window_tokens = nn.Parameter(torch.randn(dim))

        # prenorm and non-linearity for window tokens
        # then projection to queries and keys for window tokens

        self.window_tokens_to_qk = nn.Sequential(
            nn.LayerNorm(dim_head),
            nn.GELU(),
            Rearrange('b h n c -> b (h c) n'),
            nn.Conv1d(inner_dim, inner_dim * 2, 1),
            Rearrange('b (h c) n -> b h n c', h = heads),
        )

        # window attention

        self.window_attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        einstein notation

        b - batch
        c - channels
        w1 - window size (height)
        w2 - also window size (width)
        i - sequence dimension (source)
        j - sequence dimension (target dimension to be reduced)
        h - heads
        x - height of feature map divided by window size
        y - width of feature map divided by window size
        """

        batch, height, width, heads, wsz = x.shape[0], *x.shape[-2:], self.heads, self.window_size
        assert (height % wsz) == 0 and (width % wsz) == 0, f'height {height} and width {width} must be divisible by window size {wsz}'
        num_windows = (height // wsz) * (width // wsz)

        x = self.norm(x)

        # fold in windows for "depthwise" attention - not sure why it is named depthwise when it is just "windowed" attention

        x = rearrange(x, 'b c (h w1) (w w2) -> (b h w) c (w1 w2)', w1 = wsz, w2 = wsz)

        # add windowing tokens

        w = repeat(self.window_tokens, 'c -> b c 1', b = x.shape[0])
        x = torch.cat((w, x), dim = -1)

        # project for queries, keys, value

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)

        # split out heads

        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = heads), (q, k, v))

        # scale

        q = q * self.scale

        # similarity

        dots = einsum('b h i d, b h j d -> b h i j', q, k)

        # attention

        attn = self.attend(dots)

        # aggregate values

        out = torch.matmul(attn, v)

        # split out windowed tokens

        window_tokens, windowed_fmaps = out[:, :, 0], out[:, :, 1:]

        # early return if there is only 1 window

        if num_windows == 1:
            fmap = rearrange(windowed_fmaps, '(b x y) h (w1 w2) d -> b (h d) (x w1) (y w2)', x = height // wsz, y = width // wsz, w1 = wsz, w2 = wsz)
            return self.to_out(fmap)

        # carry out the pointwise attention, the main novelty in the paper

        window_tokens = rearrange(window_tokens, '(b x y) h d -> b h (x y) d', x = height // wsz, y = width // wsz)
        windowed_fmaps = rearrange(windowed_fmaps, '(b x y) h n d -> b h (x y) n d', x = height // wsz, y = width // wsz)

        # windowed queries and keys (preceded by prenorm activation)

        w_q, w_k = self.window_tokens_to_qk(window_tokens).chunk(2, dim = -1)

        # scale

        w_q = w_q * self.scale

        # similarities

        w_dots = einsum('b h i d, b h j d -> b h i j', w_q, w_k)

        w_attn = self.window_attend(w_dots)

        # aggregate the feature maps from the "depthwise" attention step (the most interesting part of the paper, one i haven't seen before)

        aggregated_windowed_fmap = einsum('b h i j, b h j w d -> b h i w d', w_attn, windowed_fmaps)

        # fold back the windows and then combine heads for aggregation

        fmap = rearrange(aggregated_windowed_fmap, 'b h (x y) (w1 w2) d -> b (h d) (x w1) (y w2)', x = height // wsz, y = width // wsz, w1 = wsz, w2 = wsz)
        return self.to_out(fmap)
