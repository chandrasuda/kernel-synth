# Extracted by kernel-synth
# Source: vit_pytorch/na_vit_nested_tensor_3d.py (lines 135-323)
# Class: NaViT
# Tags: custom-init, imports:einops, self-contained, topk
# Novelty: 0.88
# Reason: forward uses topk; file imports einops, einops, einops, einops; non-trivial parameter init

class NaViT(Module):
    def __init__(
        self,
        *,
        image_size,
        max_frames,
        patch_size,
        frame_patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        num_registers = 4,
        qk_rmsnorm = True,
        token_dropout_prob: float | None = None
    ):
        super().__init__()
        image_height, image_width = pair(image_size)

        if pkg_version.parse(torch.__version__) < pkg_version.parse('2.5'):
            print('nested tensor NaViT was tested on pytorch 2.5')

        # what percent of tokens to dropout
        # if int or float given, then assume constant dropout prob
        # otherwise accept a callback that in turn calculates dropout prob from height and width

        self.token_dropout_prob = token_dropout_prob

        # calculate patching related stuff

        assert divisible_by(image_height, patch_size) and divisible_by(image_width, patch_size), 'Image dimensions must be divisible by the patch size.'
        assert divisible_by(max_frames, frame_patch_size)

        patch_frame_dim, patch_height_dim, patch_width_dim = (max_frames // frame_patch_size), (image_height // patch_size), (image_width // patch_size)

        patch_dim = channels * (patch_size ** 2) * frame_patch_size

        self.channels = channels
        self.patch_size = patch_size
        self.to_patches = Rearrange('c (f pf) (h p1) (w p2) -> f h w (c pf p1 p2)', p1 = patch_size, p2 = patch_size, pf = frame_patch_size)

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embed_frame = nn.Parameter(torch.zeros(patch_frame_dim, dim))
        self.pos_embed_height = nn.Parameter(torch.zeros(patch_height_dim, dim))
        self.pos_embed_width = nn.Parameter(torch.zeros(patch_width_dim, dim))

        # register tokens

        self.register_tokens = nn.Parameter(torch.zeros(num_registers, dim))

        nn.init.normal_(self.pos_embed_frame, std = 0.02)
        nn.init.normal_(self.pos_embed_height, std = 0.02)
        nn.init.normal_(self.pos_embed_width, std = 0.02)
        nn.init.normal_(self.register_tokens, std = 0.02)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, qk_rmsnorm)

        # final attention pooling queries

        self.attn_pool_queries = nn.Parameter(torch.randn(dim))
        self.attn_pool = Attention(dim = dim, dim_head = dim_head, heads = heads)

        # output to logits

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim, bias = False),
            nn.Linear(dim, num_classes, bias = False)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        volumes: List[Tensor], # different resolution images / CT scans
    ):
        batch, device = len(volumes), self.device
        arange = partial(torch.arange, device = device)

        assert all([volume.ndim == 4 and volume.shape[0] == self.channels for volume in volumes]), f'all volumes must have {self.channels} channels and number of dimensions of {self.channels} (channels, frame, height, width)'

        all_patches = [self.to_patches(volume) for volume in volumes]

        # prepare factorized positional embedding height width indices

        positions = []

        for patches in all_patches:
            patch_frame, patch_height, patch_width = patches.shape[:3]
            fhw_indices = torch.stack(torch.meshgrid((arange(patch_frame), arange(patch_height), arange(patch_width)), indexing = 'ij'), dim = -1)
            fhw_indices = rearrange(fhw_indices, 'f h w c -> (f h w) c')

            positions.append(fhw_indices)

        # need the sizes to compute token dropout + positional embedding

        tokens = [rearrange(patches, 'f h w d -> (f h w) d') for patches in all_patches]

        # handle token dropout

        seq_lens = torch.tensor([i.shape[0] for i in tokens], device = device)

        if self.training and self.token_dropout_prob > 0:

            keep_seq_lens = ((1. - self.token_dropout_prob) * seq_lens).int().clamp(min = 1)

            kept_tokens = []
            kept_positions = []

            for one_image_tokens, one_image_positions, seq_len, num_keep in zip(tokens, positions, seq_lens, keep_seq_lens):
                keep_indices = torch.randn((seq_len,), device = device).topk(num_keep, dim = -1).indices

                one_image_kept_tokens = one_image_tokens[keep_indices]
                one_image_kept_positions = one_image_positions[keep_indices]

                kept_tokens.append(one_image_kept_tokens)
                kept_positions.append(one_image_kept_positions)

            tokens, positions, seq_lens = kept_tokens, kept_positions, keep_seq_lens

        # add all height and width factorized positions


        frame_indices, height_indices, width_indices = torch.cat(positions).unbind(dim = -1)
        frame_embed, height_embed, width_embed = self.pos_embed_frame[frame_indices], self.pos_embed_height[height_indices], self.pos_embed_width[width_indices]

        pos_embed = frame_embed + height_embed + width_embed

        tokens = torch.cat(tokens)

        # linear projection to patch embeddings

        tokens = self.to_patch_embedding(tokens)

        # absolute positions

        tokens = tokens + pos_embed

        # add register tokens

        tokens = tokens.split(seq_lens.tolist())

        tokens = [torch.cat((self.register_tokens, one_tokens)) for one_tokens in tokens]

        # use nested tensor for transformers and save on padding computation

        tokens = nested_tensor(tokens, layout = torch.jagged, device = device)

        # embedding dropout

        tokens = self.dropout(tokens)

        # transformer

        tokens = self.transformer(tokens)

        # attention pooling
        # will use a jagged tensor for queries, as SDPA requires all inputs to be jagged, or not

        attn_pool_queries = [rearrange(self.attn_pool_queries, '... -> 1 ...')] * batch

        attn_pool_queries = nested_tensor(attn_pool_queries, layout = torch.jagged)

        pooled = self.attn_pool(attn_pool_queries, tokens)

        # back to unjagged

        logits = torch.stack(pooled.unbind())

        logits = rearrange(logits, 'b 1 d -> b d')

        logits = self.to_latent(logits)

        return self.mlp_head(logits)
