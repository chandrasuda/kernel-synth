# Extracted by kernel-synth
# Source: vit_pytorch/learnable_memory_vit.py (lines 157-218)
# Class: Adapter
# Tags: custom-init, imports:einops, self-contained, uses-buffers
# Novelty: 0.85
# Reason: file imports einops, einops, einops, einops, einops; non-trivial parameter init

class Adapter(nn.Module):
    def __init__(
        self,
        *,
        vit,
        num_memories_per_layer = 10,
        num_classes = 2,
    ):
        super().__init__()
        assert isinstance(vit, ViT)

        # extract some model variables needed

        dim = vit.cls_token.shape[-1]
        layers = len(vit.transformer.layers)
        num_patches = vit.pos_embedding.shape[-2]

        self.vit = vit

        # freeze ViT backbone - only memories will be finetuned

        freeze_all_layers_(vit)

        # learnable parameters

        self.memory_cls_token = nn.Parameter(torch.randn(dim))
        self.memories_per_layer = nn.Parameter(torch.randn(layers, num_memories_per_layer, dim))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        # specialized attention mask to preserve the output of the original ViT
        # it allows the memory CLS token to attend to all other tokens (and the learnable memory layer tokens), but not vice versa

        attn_mask = torch.ones((num_patches, num_patches), dtype = torch.bool)
        attn_mask = F.pad(attn_mask, (1, num_memories_per_layer), value = False)  # main tokens cannot attend to learnable memories per layer
        attn_mask = F.pad(attn_mask, (0, 0, 1, 0), value = True)                  # memory CLS token can attend to everything
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, img):
        b = img.shape[0]

        tokens = self.vit.img_to_tokens(img)

        # add task specific memory tokens

        memory_cls_tokens = repeat(self.memory_cls_token, 'd -> b 1 d', b = b)
        tokens = torch.cat((memory_cls_tokens, tokens), dim = 1)

        # pass memories along with image tokens through transformer for attending

        out = self.vit.transformer(tokens, memories = self.memories_per_layer, attn_mask = self.attn_mask)

        # extract memory CLS tokens

        memory_cls_tokens = out[:, 0]

        # pass through task specific adapter head

        return self.mlp_head(memory_cls_tokens)
