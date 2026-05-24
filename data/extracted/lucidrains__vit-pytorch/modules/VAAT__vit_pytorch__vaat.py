# Extracted by kernel-synth
# Source: vit_pytorch/vaat.py (lines 421-745)
# Class: VAAT
# Tags: imports:einops, math-heavy, self-contained, uses-buffers
# Novelty: 0.95
# Reason: ~8 arithmetic ops in forward; file imports einops, einops, einops, einops, einops, einops, einops, einops

class VAAT(Module):
    def __init__(
        self,
        vit: ViT | dict,
        ast: AST | dict,
        *,
        dim,
        depth,
        heads,
        dim_head,
        dim_action,
        mlp_dim,
        num_image_views = None,
        num_audio_views = None,
        num_tasks = None,
        dim_extra_token = None,
        num_register_tokens = 4,
        action_chunk_len = 7,
        time_seq_len = 1,
        dropout = 0.,
        add_self_attn = True,  # in the paper, they didn't have any ways for the action token to exchange information with the extra token, so we'll just add it as an option
        self_attn_heads = 4,
        self_attn_dim_head = 32,
        ast_layer_indices: tuple[int, ...] | None = None,
        vit_layer_indices: tuple[int, ...] | None = None,
        num_advantage_bins = 0
    ):
        super().__init__()

        # vit

        if isinstance(vit, dict):
            vit = ViT(**vit)

        self.vit = vit

        vit_dim = vit.dim

        assert vit.depth == depth or exists(vit_layer_indices), f'if the VAAT depth is not equal to the ViT depth, you must pass in the indices from the ViT to be layered to the VAAT in order from bottom to top'

        vit_layer_indices = default(vit_layer_indices, tuple(range(depth)))

        assert len(vit_layer_indices) == depth, f'number of vit layer indices {len(vit_layer_indices)} does not much the VAT depth {depth}'

        self.register_buffer('vit_layer_indices', tensor(vit_layer_indices), persistent = False)

        # ast

        if isinstance(ast, dict):
            ast = AST(**ast)

        self.ast = ast

        ast_dim = ast.dim

        self.ast_accept_spec = ast.accept_spec

        assert ast.depth == depth or exists(ast_layer_indices), f'if the VAAT depth is not equal to the AST depth, you must pass in the indices from the AST to be layered to the VAAT in order from bottom to top'

        ast_layer_indices = default(ast_layer_indices, tuple(range(depth)))

        assert len(ast_layer_indices) == depth, f'number of ast layer indices {len(ast_layer_indices)} does not much the VAAT depth {depth}'

        self.register_buffer('ast_layer_indices', tensor(ast_layer_indices), persistent = False)

        # handle maybe multiple frames

        is_video = time_seq_len > 1

        self.is_video = is_video
        self.time_seq_len = time_seq_len
        self.time_pos_emb = nn.Parameter(torch.randn(time_seq_len, vit_dim) * 1e-2) if is_video else None

        # maybe view embeddings

        self.image_view_emb = nn.Parameter(torch.randn(num_image_views, vit_dim) * 1e-2) if exists(num_image_views) and num_image_views > 1 else None

        self.audio_view_emb = nn.Parameter(torch.randn(num_audio_views, ast_dim) * 1e-2) if exists(num_audio_views) and num_audio_views > 1 else None

        # handle maybe task conditioning

        self.has_tasks = exists(num_tasks)

        if self.has_tasks:
            self.task_emb = nn.Parameter(torch.randn(num_tasks, dim) * 1e-2)

        # register tokens from Darcet et al.

        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim) * 1e-2)

        # to action tokens

        self.action_pos_emb = nn.Parameter(torch.randn(action_chunk_len, dim) * 1e-2)

        # handle maybe advantage conditioning

        self.has_advantages = num_advantage_bins > 0
        self.num_advantage_bins = num_advantage_bins

        if self.has_advantages:
            self.advantage_emb = nn.Embedding(num_advantage_bins + 1, dim)

        self.layers = ModuleList([])

        for _ in range(depth):
            maybe_film = FiLM(dim = dim) if self.has_tasks else None
            maybe_self_attn = Attention(dim = dim, heads = self_attn_heads, dim_head = self_attn_dim_head, dropout = dropout) if add_self_attn else None

            self.layers.append(ModuleList([
                maybe_film,
                maybe_self_attn,
                Attention(dim = dim, dim_context = vit_dim, heads = heads, dim_head = dim_head, dropout = dropout, cross_attend = True),
                Attention(dim = dim, dim_context = ast_dim, heads = heads, dim_head = dim_head, dropout = dropout, cross_attend = True),
                FeedForward(dim = dim, hidden_dim = mlp_dim, dropout = dropout)
            ]))

        self.final_norm = nn.LayerNorm(dim)
        self.to_pred_action = nn.Linear(dim, dim_action, bias = False)

        # handle the extra token

        self.accept_extra_token = exists(dim_extra_token)

        if exists(dim_extra_token):
            self.to_extra_token = nn.Linear(dim_extra_token, dim)

    def forward(
        self,
        video_or_image,   # (b v? c t? h w)      - batch, views [wrist + third person or more], channels, maybe time, height, width
        audio_or_spec,    # (b v? t) | (b v?f t) - batch, audio len | batch, spec freq, time
        *,
        extra = None,     # (b d)                - batch, dim extra
        tasks = None,     # (b)
        advantages = None,# (b)
        actions = None,   # (b k d)              - batch, action chunk length, action dimension
        return_hiddens = False,
        freeze_vit = False,
        freeze_ast = False
    ):
        batch, device = video_or_image.shape[0], video_or_image.device
        return_loss = exists(actions)

        # handle some various input dimensions

        if video_or_image.ndim == 4:
            video_or_image = rearrange(video_or_image, 'b 1 c h w')

        assert (
            (video_or_image.ndim == 5 and not self.is_video) or
            (video_or_image.ndim == 6 and self.is_video)
        )

        if video_or_image.ndim == 5:
            video_or_image = rearrange(video_or_image, 'b v c h w -> b v c 1 h w')

        assert video_or_image.shape[3] == self.time_seq_len

        # audio shapes - adding view if impliciy to be 1

        if audio_or_spec.ndim == 2 and not self.ast_accept_spec:
            audio_or_spec = rearrange(audio_or_spec, 'b t -> b 1 t')

        elif audio_or_spec.ndim == 3 and self.ast_accept_spec:
            audio_or_spec = rearrange(audio_or_spec, 'b f t -> b 1 f t')

        # to images

        images = rearrange(video_or_image, 'b v c t h w -> b v t c h w')

        images, image_packed_shape = pack([images], '* c h w')

        # to audio

        if self.ast_accept_spec:
            audio_or_spec, audio_packed_shape = pack([audio_or_spec], '* f t')
        else:
            audio_or_spec, audio_packed_shape = pack([audio_or_spec], '* t')

        # get representation trajectory from vit

        vit_forward_context = torch.no_grad if freeze_vit else nullcontext

        with vit_forward_context():
            embed, hiddens = self.vit(images, return_hiddens = True)

        hiddens = cat((hiddens, embed[None, ...]))

        # extract the hiddens needed for the action cross attention

        hiddens = hiddens[self.vit_layer_indices]

        # unpack temporarily for embedding

        hiddens, = unpack(hiddens, image_packed_shape, 'l * n d') # l for layers

        # maybe add time embeddings

        if self.is_video:
            time_pos_emb = rearrange(self.time_pos_emb, 't d -> t 1 d')
            hiddens = hiddens + time_pos_emb

        # maybe view embeddings

        if exists(self.image_view_emb):
            assert self.image_view_emb.shape[0] == hiddens.shape[2]

            image_view_emb = rearrange(self.image_view_emb, 'v d -> v 1 1 d')
            hiddens = hiddens + image_view_emb

        # get representation trajectory from ast

        ast_forward_context = torch.no_grad if freeze_ast else nullcontext

        with ast_forward_context():
            audio_embed, audio_hiddens = self.ast(audio_or_spec, return_hiddens = True)

        audio_hiddens = cat((audio_hiddens, audio_embed[None, ...]))

        # extract the hiddens needed for the action cross attention

        audio_hiddens = audio_hiddens[self.ast_layer_indices]

        # unpack audio temporarily for embedding

        audio_hiddens, = unpack(audio_hiddens, audio_packed_shape, 'l * n d') # l for layers

        # maybe audio view embeddings

        if exists(self.audio_view_emb):
            assert self.audio_view_emb.shape[0] == audio_hiddens.shape[2]

            audio_view_emb = rearrange(self.audio_view_emb, 'v d -> v 1 1 d')
            audio_hiddens = audio_hiddens + audio_view_emb

        # maybe tasks

        if exists(tasks):
            assert self.has_tasks, f'`num_tasks` must be set on `VAT` for task conditioning'

            task_emb = self.task_emb[tasks]

        # cross from actions to representation trajectory

        image_context = rearrange(hiddens, 'l b v t n d -> l b (v t n) d')

        audio_context = rearrange(audio_hiddens, 'l b v n d -> l b (v n) d')

        # main action tokens

        action_tokens = repeat(self.action_pos_emb, 'n d -> b n d', b = batch)

        # maybe advantage tokens

        empty_token = action_tokens[:, 0:0]

        maybe_advantage_embed = empty_token

        if self.has_advantages and exists(advantages):
            if isinstance(advantages, int):
                advantages = torch.full((batch,), advantages, device = device, dtype = torch.long)

            maybe_advantage_embed = self.advantage_emb(advantages + 1)

        # register tokens

        register_tokens = empty_token

        if exists(self.register_tokens):
            register_tokens = repeat(self.register_tokens, 'n d -> b n d', b = batch)

        # extra

        maybe_extra_embed = empty_token

        has_extra = exists(extra)
        if has_extra:
            assert self.accept_extra_token

            maybe_extra_embed = self.to_extra_token(extra)

        # pack all tokens for attention

        tokens, ps = pack((register_tokens, maybe_advantage_embed, action_tokens, maybe_extra_embed), 'b * d')

        # transformer

        hiddens = [tokens]

        for (maybe_film, maybe_self_attn, image_cross_attn, audio_cross_attn, ff), image_layer_context, audio_layer_context in zip(self.layers, image_context, audio_context):

            if exists(maybe_film) and exists(tasks):
                tokens = maybe_film(tokens, task_emb)

            tokens = image_cross_attn(tokens, image_layer_context) + tokens

            tokens = audio_cross_attn(tokens, audio_layer_context) + tokens

            if exists(maybe_self_attn):
                tokens = maybe_self_attn(tokens) + tokens

            tokens = ff(tokens) + tokens

            hiddens.append(tokens)

        # unpack register, advantage, action, and extra tokens

        maybe_register_embed, maybe_advantage_embed, action_tokens, maybe_extra_embed = unpack(tokens, ps, 'b * d')

        # norm and prediction

        action_tokens = self.final_norm(action_tokens)

        pred_action = self.to_pred_action(action_tokens)

        if not return_loss:
            if not return_hiddens:
                return pred_action

            return pred_action, stack(hiddens)

        assert pred_action.shape[1] == actions.shape[1]

        # they found l1 loss suffices

        return F.l1_loss(pred_action, actions)
