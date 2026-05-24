# Extracted by kernel-synth
# Source: vit_pytorch/look_vit.py (lines 140-255)
# Class: LookViT
# Tags: custom-init, imports:einops, math-heavy, self-contained
# Novelty: 0.90
# Reason: ~8 arithmetic ops in forward; file imports einops, einops, einops, einops, einops, einops, einops; non-trivial parameter init

class LookViT(Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        num_classes,
        depth = 3,
        patch_size = 16,
        heads = 8,
        mlp_factor = 4,
        dim_head = 64,
        highres_patch_size = 12,
        highres_mlp_factor = 4,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        patch_conv_kernel_size = 7,
        dropout = 0.1,
        channels = 3
    ):
        super().__init__()
        assert divisible_by(image_size, highres_patch_size)
        assert divisible_by(image_size, patch_size)
        assert patch_size > highres_patch_size, 'patch size of the main vision transformer should be smaller than the highres patch sizes (that does the `lookup`)'
        assert not divisible_by(patch_conv_kernel_size, 2)

        self.dim = dim
        self.image_size = image_size
        self.patch_size = patch_size

        kernel_size = patch_conv_kernel_size
        patch_dim = (highres_patch_size * highres_patch_size) * channels

        self.to_patches = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = highres_patch_size, p2 = highres_patch_size),
            nn.Conv2d(patch_dim, dim, kernel_size, padding = kernel_size // 2),
            Rearrange('b c h w -> b h w c'),
            LayerNorm(dim),
        )

        # absolute positions

        num_patches = (image_size // highres_patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(num_patches, dim))

        # lookvit blocks

        layers = ModuleList([])

        for _ in range(depth):
            layers.append(ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = dropout),
                MLP(dim = dim, factor = mlp_factor, dropout = dropout),
                Attention(dim = dim, dim_head = cross_attn_dim_head, heads = cross_attn_heads, dropout = dropout, cross_attend = True),
                Attention(dim = dim, dim_head = cross_attn_dim_head, heads = cross_attn_heads, dropout = dropout, cross_attend = True, reuse_attention = True),
                LayerNorm(dim),
                MLP(dim = dim, factor = highres_mlp_factor, dropout = dropout)
            ]))

        self.layers = layers

        self.norm = LayerNorm(dim)
        self.highres_norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, num_classes, bias = False)

    def forward(self, img):
        assert img.shape[-2:] == (self.image_size, self.image_size)

        # to patch tokens and positions

        highres_tokens = self.to_patches(img)
        size = highres_tokens.shape[-2]

        pos_emb = posemb_sincos_2d(highres_tokens)
        highres_tokens = highres_tokens + rearrange(pos_emb, '(h w) d -> h w d', h = size)

        tokens = F.interpolate(
            rearrange(highres_tokens, 'b h w d -> b d h w'),
            img.shape[-1] // self.patch_size,
            mode = 'bilinear'
        )

        tokens = rearrange(tokens, 'b c h w -> b (h w) c')
        highres_tokens = rearrange(highres_tokens, 'b h w c -> b (h w) c')

        # attention and feedforwards

        for attn, mlp, lookup_cross_attn, highres_attn, highres_norm, highres_mlp in self.layers:

            # main tokens cross attends (lookup) on the high res tokens

            lookup_out, qk_sim = lookup_cross_attn(tokens, highres_tokens, return_qk_sim = True)  # return attention as they reuse the attention matrix
            tokens = lookup_out + tokens

            tokens = attn(tokens) + tokens
            tokens = mlp(tokens) + tokens

            # attention-reuse

            qk_sim = rearrange(qk_sim, 'b h i j -> b h j i') # transpose for reverse cross attention

            highres_tokens = highres_attn(highres_tokens, tokens, qk_sim = qk_sim) + highres_tokens
            highres_tokens = highres_norm(highres_tokens)

            highres_tokens = highres_mlp(highres_tokens) + highres_tokens

        # to logits

        tokens = self.norm(tokens)
        highres_tokens = self.highres_norm(highres_tokens)

        tokens = reduce(tokens, 'b n d -> b d', 'mean')
        highres_tokens = reduce(highres_tokens, 'b n d -> b d', 'mean')

        return self.to_logits(tokens + highres_tokens)
