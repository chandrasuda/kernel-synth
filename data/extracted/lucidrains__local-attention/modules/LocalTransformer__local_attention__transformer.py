# Extracted by kernel-synth
# Source: local_attention/transformer.py (lines 242-443)
# Class: LocalTransformer
# Tags: imports:einops, looks-generic, self-contained
# Novelty: 0.15
# Reason: file imports einops

class LocalTransformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        dim,
        depth,
        causal = True,
        local_attn_window_size = 512,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ignore_index = -1,
        use_xpos = False,
        xpos_scale_base = None,
        use_dynamic_pos_bias = False,
        global_attn_layer: Module | None = None,
        layers_insert_global_attn: tuple[int, ...] | None = None,
        num_residual_streams = 4,
        **kwargs
    ):
        super().__init__()

        self.has_embed_unembed = exists(num_tokens)

        if self.has_embed_unembed:
            self.token_emb = nn.Embedding(num_tokens, dim)
            self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.max_seq_len = max_seq_len
        self.layers = ModuleList([])

        self.local_attn_window_size = local_attn_window_size
        self.dynamic_pos_bias = None
        if use_dynamic_pos_bias:
            self.dynamic_pos_bias = DynamicPositionBias(dim = dim // 2, heads = heads)

        init_hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        # allow for inserting global attention or memory layers

        layers_insert_global_attn = default(layers_insert_global_attn, tuple(range(1, depth + 1)))
        assert all([0 < layer <= depth for layer in layers_insert_global_attn])

        global_attn_layers = set(layers_insert_global_attn)

        self.global_layers = ModuleList([])

        # define modules throughout layers

        for index in range(depth):
            layer = index + 1

            self.global_layers.append(init_hyper_conn(dim = dim, branch = deepcopy(global_attn_layer)) if exists(global_attn_layer) and layer in global_attn_layers else None)

            self.layers.append(nn.ModuleList([
                init_hyper_conn(dim = dim, branch = LocalMHA(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, causal = causal, window_size = local_attn_window_size, use_xpos = use_xpos, xpos_scale_base = xpos_scale_base, use_rotary_pos_emb = not use_dynamic_pos_bias, prenorm = True, **kwargs)),
                init_hyper_conn(dim = dim, branch = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout))
            ]))

        self.ignore_index = ignore_index

        if self.has_embed_unembed:
            self.to_logits = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_tokens, bias = False)
            )

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prime,
        seq_len,
        temperature = 1.,
        filter_thres = 0.9,
        use_kv_cache = True,
        **kwargs
    ):
        assert self.has_embed_unembed
        assert temperature >= 0.

        n, device = prime.shape[1], prime.device

        out = prime

        cache = None

        for _ in range(seq_len):

            logits, new_cache = self.forward(
                out[:, -self.max_seq_len:],
                cache = cache,
                return_cache = True,
                **kwargs
            )

            if use_kv_cache:
                cache = new_cache

            filtered_logits = top_k(logits[:, -1], thres = filter_thres)

            if temperature == 0.:
                sampled = filtered_logits.argmax(dim = -1, keepdim = True)
            else:
                probs = F.softmax(filtered_logits / temperature, dim = -1)
                sampled = torch.multinomial(probs, 1)

            out = torch.cat((out, sampled), dim = -1)

        return out[:, n:]

    def forward(
        self,
        x,
        mask = None,
        cache = None,
        return_loss = False,
        return_cache = False
    ):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        n, device = x.shape[1], x.device

        if self.has_embed_unembed:
            x = self.token_emb(x)

            assert n <= self.max_seq_len
            x = x + self.pos_emb(torch.arange(n, device = device))

        # handle old and new cache

        has_cache = exists(cache)
        cached_kv = cached_attn_bias = None

        if has_cache:
            cached_kv, cached_attn_bias = cache

        new_cached_kv = []
        iter_cached_kv = iter(default(cached_kv, []))

        if has_cache:
            x = x[:, -1:]

        # dynamic pos bias

        attn_bias = cached_attn_bias

        if not exists(attn_bias) and exists(self.dynamic_pos_bias):
            w = self.local_attn_window_size
            attn_bias = self.dynamic_pos_bias(w, w * 2)

        # go through layers

        x = self.expand_streams(x)

        for (attn, ff), global_layer in zip(self.layers, self.global_layers):

            if exists(global_layer):
                x = global_layer(x)

            x, layer_cached_kv = attn(
                x,
                mask = mask,
                attn_bias = attn_bias,
                return_cache = True,
                cache = next(iter_cached_kv, None)
            )

            new_cached_kv.append(layer_cached_kv)

            x = ff(x)

        x = self.reduce_streams(x)

        if not self.has_embed_unembed:
            return x

        # to logits

        logits = self.to_logits(x)

        if not return_loss:

            if not return_cache:
                return logits

            return logits, Cache(new_cached_kv, attn_bias)

        # cross entropy loss

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.ignore_index
        )

        return loss
