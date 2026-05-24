# Extracted by kernel-synth
# Source: vit_pytorch/vit_with_decorr.py (lines 154-220)
# Class: ViT
# Tags: custom-init, imports:einops, self-contained, uses-buffers
# Novelty: 0.85
# Reason: file imports einops, einops, einops, einops, einops, einops, einops, einops, einops; non-trivial parameter init

class ViT(Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., decorr_sample_frac = 1.):
        super().__init__()
        image_height, image_width = pair(image_size)
        self.patch_size = patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

        # decorrelation loss related

        self.has_decorr_loss = decorr_sample_frac > 0.

        if self.has_decorr_loss:
            self.decorr_loss = DecorrelationLoss(decorr_sample_frac)

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    def forward(
        self,
        img,
        return_decorr_aux_loss = None
    ):
        return_decorr_aux_loss = default(return_decorr_aux_loss, self.training) and self.has_decorr_loss

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x, normed_layer_inputs = self.transformer(x)

        # maybe return decor loss

        decorr_aux_loss = self.zero

        if return_decorr_aux_loss:
            decorr_aux_loss = self.decorr_loss(normed_layer_inputs)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x), decorr_aux_loss
