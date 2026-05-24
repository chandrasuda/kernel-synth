# Extracted by kernel-synth
# Source: audiolm_pytorch/audiolm_pytorch.py (lines 992-1368)
# Class: FineTransformer
# Tags: custom-init, einsum, imports:einops, looks-generic, math-heavy, self-contained
# Novelty: 0.78
# Reason: forward uses einsum; ~26 arithmetic ops in forward; file imports einops, einops, einops, einops, einops, einops; non-trivial parameter init

class FineTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_coarse_quantizers,
        num_fine_quantizers,
        codebook_size,
        dim,
        depth,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        t5_name = DEFAULT_T5_NAME,
        has_condition = False,
        cond_dim = None,
        audio_text_condition = False,
        cond_as_self_attn_prefix = False,
        cond_drop_prob = 0.5,
        grad_shrink_alpha = 0.1,
        project_coarse_logits = True,
        pad_id = -1,
        rel_pos_bias = True,
        flash_attn = False,
        **kwargs
    ):
        super().__init__()
        rel_pos_bias = rel_pos_bias and not flash_attn

        if audio_text_condition:
            has_condition = True
            cond_dim = default(cond_dim, dim)

        self.has_condition = has_condition
        self.embed_text = partial(t5_encode_text, name = t5_name)
        self.cond_drop_prob = cond_drop_prob

        self.num_coarse_quantizers = num_coarse_quantizers

        self.coarse_start_token = nn.Parameter(torch.randn(dim))
        self.fine_start_token = nn.Parameter(torch.randn(dim))

        self.coarse_embedding = nn.Embedding(num_coarse_quantizers * codebook_size, dim)
        self.fine_embedding = nn.Embedding(num_fine_quantizers * codebook_size, dim)

        self.coarse_quantize_embedding = nn.Embedding(num_coarse_quantizers, dim)
        self.fine_quantize_embedding = nn.Embedding(num_fine_quantizers, dim)

        self.pad_id = pad_id
        self.eos_id = codebook_size

        text_dim = default(cond_dim, get_encoded_dim(t5_name))
        self.proj_text_embed = nn.Linear(text_dim, dim, bias = False) if text_dim != dim else nn.Identity()

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            cross_attend = has_condition and not cond_as_self_attn_prefix,
            cond_as_self_attn_prefix = cond_as_self_attn_prefix,
            rel_pos_bias = False,
            grad_shrink_alpha = grad_shrink_alpha,
            flash_attn = flash_attn,
            **kwargs
        )

        # doing a specialized attn bias so that corresponding time steps at fine and coarse sequences attend to each other better

        self.null_pos_bias = nn.Parameter(torch.randn(heads, 1, 1)) if rel_pos_bias else None

        pos_bias_mlp_dim = dim // 2

        self.pos_bias_mlp = nn.Sequential(
            nn.Linear(2, pos_bias_mlp_dim),
            nn.SiLU(),
            nn.Linear(pos_bias_mlp_dim, pos_bias_mlp_dim),
            nn.SiLU(),
            nn.Linear(pos_bias_mlp_dim, heads)
        ) if rel_pos_bias else None

        self.codebook_size = codebook_size
        self.num_coarse_quantizers = num_coarse_quantizers
        self.num_fine_quantizers = num_fine_quantizers

        self.coarse_logit_weights = nn.Parameter(torch.randn(num_coarse_quantizers, codebook_size, dim)) if project_coarse_logits else None
        self.fine_logit_weights = nn.Parameter(torch.randn(num_fine_quantizers, codebook_size, dim))

    @property
    def device(self):
        return next(self.parameters()).device

    def load(self, path):
        # Return pkg so that if this function gets called from within a Trainer function call,
        # the trainer can also access the package loaded from the checkpoint.
        device = self.device
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = device)
        # check version
        if 'version' in pkg and version.parse(pkg['version']) < version.parse(__version__):
            print(f'model was trained on older version {pkg["version"]} of audiolm-pytorch')
        self.load_state_dict(pkg['model'])
        return pkg

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3,
        return_kv_cache = False,
        kv_cache = None,
        embed_cache = None,
        **kwargs
    ):
        iter_kv_cache = iter(default(kv_cache, []))
        iter_embed_cache = iter(default(embed_cache, []))
        new_kv_caches = []
        new_embed_caches = []

        (semantic_logits, coarse_logits), (new_kv_cache, new_embed_cache) = self.forward(*args, cond_drop_prob = 0., return_cache = True, kv_cache = next(iter_kv_cache, None), embed_cache = next(iter_embed_cache, None), **kwargs)
        new_kv_caches.append(new_kv_cache)
        new_embed_caches.append(new_embed_cache)

        if cond_scale == 1 or not self.has_condition:
            if not return_kv_cache:
                return semantic_logits, coarse_logits

            return (semantic_logits, coarse_logits), (torch.stack(new_kv_caches), torch.stack(new_embed_caches))

        (null_semantic_logits, null_coarse_logits), (null_new_kv_cache, null_new_embed_cache) = self.forward(*args, cond_drop_prob = 1., return_cache = True, kv_cache = next(iter_kv_cache, None), embed_cache = next(iter_embed_cache, None), **kwargs)
        new_kv_caches.append(null_new_kv_cache)
        new_embed_caches.append(null_new_embed_cache)

        scaled_semantic_logits = None
        if exists(null_semantic_logits):
            scaled_semantic_logits = null_semantic_logits + (semantic_logits - null_semantic_logits) * cond_scale

        scaled_coarse_logits = null_coarse_logits + (coarse_logits - null_coarse_logits) * cond_scale

        if not return_kv_cache:
            return scaled_semantic_logits, scaled_coarse_logits

        return (scaled_semantic_logits, scaled_coarse_logits), (torch.stack(new_kv_caches), torch.stack(new_embed_caches))

    def forward(
        self,
        coarse_token_ids,
        fine_token_ids,
        text: list[str] | None = None,
        text_embeds = None,
        cond_drop_prob = None,
        self_attn_mask = None,
        kv_cache = None,
        embed_cache = None,
        return_cache = False,
        return_only_fine_logits = False
    ):
        b, device = coarse_token_ids.shape[0], coarse_token_ids.device

        # handle text conditioning

        has_text = exists(text) or exists(text_embeds)
        assert not (self.has_condition ^ has_text)

        text_mask = None
        if not exists(text_embeds) and exists(text):
            with torch.inference_mode():
                text_embeds = self.embed_text(text, output_device = device)
                text_mask = torch.any(text_embeds != 0, dim = -1)

        if exists(text_embeds):
            text_embeds = self.proj_text_embed(text_embeds)

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        if exists(text_mask) and cond_drop_prob > 0:
            keep_mask = prob_mask_like((b,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        coarse_token_ids, fine_token_ids = map(lambda t: rearrange(t, 'b ... -> b (...)'), (coarse_token_ids, fine_token_ids))

        # do not attend to any of the coarse padding tokens or coarse end token either

        coarse_self_attn_mask = (coarse_token_ids != self.pad_id) & (coarse_token_ids != self.eos_id)
        coarse_token_ids = coarse_token_ids.masked_fill(~coarse_self_attn_mask, 0)

        fine_token_seq_len = fine_token_ids.shape[-1]
        coarse_self_attn_mask = F.pad(coarse_self_attn_mask, (1, fine_token_seq_len + 1), value = True)

        if exists(self_attn_mask):
            self_attn_mask &= coarse_self_attn_mask
        else:
            self_attn_mask = coarse_self_attn_mask

        # prepare coarse and fine token embeddings

        b, n = coarse_token_ids.shape

        coarse_length = coarse_token_ids.shape[-1]
        coarse_offsets = torch.arange(self.num_coarse_quantizers, device = device)
        coarse_seq_length = ceil_div(coarse_token_ids.shape[-1], self.num_coarse_quantizers)
        coarse_offsets = repeat(coarse_offsets, 'q -> (n q)', n = coarse_seq_length)
        coarse_offsets = coarse_offsets[:coarse_length]
        coarse_token_ids = coarse_token_ids + rearrange(coarse_offsets, '... -> 1 ...') * self.codebook_size

        fine_length = fine_token_ids.shape[-1]
        fine_offsets = torch.arange(self.num_fine_quantizers, device = device)
        fine_seq_length = ceil_div(fine_token_ids.shape[-1], self.num_fine_quantizers)
        fine_offsets = repeat(fine_offsets, 'q -> (n q)', n = fine_seq_length)
        fine_offsets = fine_offsets[:fine_length]
        fine_token_ids = fine_token_ids + rearrange(fine_offsets, '... -> 1 ...') * self.codebook_size

        coarse_tokens = self.coarse_embedding(coarse_token_ids)
        fine_tokens = self.fine_embedding(fine_token_ids)

        coarse_quantize_tokens = repeat(self.coarse_quantize_embedding.weight, 'q d -> (n q) d', n = ceil_div(coarse_token_ids.shape[-1], self.num_coarse_quantizers))
        coarse_quantize_tokens = coarse_quantize_tokens[:coarse_token_ids.shape[-1], ...]
        coarse_tokens = coarse_tokens + coarse_quantize_tokens

        fine_quantize_tokens = repeat(self.fine_quantize_embedding.weight, 'q d -> (n q) d', n = ceil_div(fine_token_ids.shape[-1], self.num_fine_quantizers))
        fine_quantize_tokens = fine_quantize_tokens[:fine_token_ids.shape[-1], ...]
        fine_tokens = fine_tokens + fine_quantize_tokens

        coarse_start_tokens = repeat(self.coarse_start_token, 'd -> b 1 d', b = b)
        fine_start_tokens = repeat(self.fine_start_token, 'd -> b 1 d', b = b)

        tokens = torch.cat((
            coarse_start_tokens,
            coarse_tokens,
            fine_start_tokens,
            fine_tokens
        ), dim = 1)

        # an engineered attention bias so coarse and fine sequences attend to each other better

        attn_bias = None

        if exists(self.pos_bias_mlp):
            max_seq_len = max(coarse_seq_length, fine_seq_length)

            coarse_pos = torch.arange(coarse_seq_length, device = device)
            fine_pos = torch.arange(fine_seq_length, device = device)

            coarse_pos = repeat(coarse_pos, 'n -> (n q)', q = self.num_coarse_quantizers)[:coarse_length]
            fine_pos = repeat(fine_pos, 'n -> (n q)', q = self.num_fine_quantizers)[:fine_length]

            coarse_pos = F.pad(coarse_pos, (1, 0), value = -1)
            fine_pos = F.pad(fine_pos, (1, 0), value = -1)

            seq_positions = torch.cat((coarse_pos, fine_pos), dim = -1)

            coarse_offsets = F.pad(coarse_offsets, (1, 0), value = 0)
            fine_offsets = fine_offsets + self.num_coarse_quantizers
            fine_offsets = F.pad(fine_offsets, (1, 0), value = 0)

            seq_offsets = torch.cat((coarse_offsets, fine_offsets), dim = -1)

            pos_mlp_input = torch.stack((seq_positions.clamp(min = 0), seq_offsets), dim = -1)

            num_offsets = self.num_fine_quantizers + self.num_coarse_quantizers

            # relative positions are always (2 * N - 1), where N is the length of the dimension

            rel_seq_len, rel_offsets = map(lambda n: 2 * n - 1, (max_seq_len, num_offsets))

            # get all relative distances

            rel_dist = (rearrange(pos_mlp_input, 'i c -> i 1 c') - rearrange(pos_mlp_input, 'j c -> 1 j c'))

            # get all possible relative distances for the attention bias to be computed from the mlp
            # which would be - (2 * N - 1) * (2 * Q - 1) - where N = sequence length and Q = total quantizers

            rel_seq_len_range = repeat(torch.arange(rel_seq_len, device = device), 'n -> (n q)', q = rel_offsets)
            rel_offset_range = repeat(torch.arange(rel_offsets, device = device), 'q -> (n q)', n = rel_seq_len)

            mlp_inputs = torch.stack((rel_seq_len_range, rel_offset_range), dim = -1)

            # implicitly parameterized relative distances, by sequence and quantizer positions

            attn_bias = self.pos_bias_mlp(mlp_inputs.float())

            # translate coordinates of (rel_seq_pos, rel_quantizer_offset) -> positive index to select from attn bias

            rel_dist_seq_pos, rel_dist_seq_offset = rel_dist.unbind(dim = -1)

            rel_dist_seq_pos += max_seq_len - 1
            rel_dist_seq_offset += num_offsets - 1

            rel_dist_indices = rel_dist_seq_pos * rel_offsets + rel_dist_seq_offset

            # select the relative positional attention bias outputted by the MLP
            # savings go from (N * Q) ^ 2 -> ~ (4 * N * Q)

            attn_bias = attn_bias[rel_dist_indices]

            attn_bias = rearrange(attn_bias, '... h -> h ...')

            # need to make sure start token has a custom positional bias

            is_start_token_seq = seq_positions == -1
            start_token_mask = rearrange(is_start_token_seq, 'i -> i 1') | rearrange(is_start_token_seq, 'j -> 1 j')

            attn_bias = torch.where(
                start_token_mask,
                self.null_pos_bias,
                attn_bias,
            )

        # attention

        tokens, next_kv_cache = self.transformer(
            tokens,
            context = text_embeds,
            self_attn_mask = self_attn_mask,
            context_mask = text_mask,
            attn_bias = attn_bias,
            kv_cache = kv_cache,
            return_kv_cache = True
        )

        if exists(embed_cache):
            tokens = torch.cat((embed_cache, tokens), dim = -2)

        new_embed_cache = tokens

        # figure out which tokens are coarse vs fine for logit projection

        pred_coarse_tokens, pred_fine_tokens = tokens[:, :n], tokens[:, (n + 1):]

        # get coarse logits

        pred_coarse_seq_len = pred_coarse_tokens.shape[1]

        padding = remainder_needed_until_multiple(pred_coarse_seq_len, self.num_coarse_quantizers)

        if padding != 0:
            pred_coarse_tokens = F.pad(pred_coarse_tokens, (0, 0, 0, padding), value = 0.)

        pred_coarse_tokens = rearrange(pred_coarse_tokens, 'b (n q) d -> b n q d', q = self.num_coarse_quantizers)

        coarse_logits = None

        if not return_only_fine_logits and exists(self.coarse_logit_weights):
            coarse_logits = einsum('q c d, b n q d -> b n q c', self.coarse_logit_weights, pred_coarse_tokens)

            coarse_logits = rearrange(coarse_logits, 'b n q c -> b (n q) c')

            coarse_logits = coarse_logits[:, :pred_coarse_seq_len]

        # get fine logits

        pred_fine_seq_len = pred_fine_tokens.shape[1]
        nq = round_down_nearest_multiple(pred_fine_seq_len, self.num_fine_quantizers)

        pred_fine_tokens_groupable, pred_fine_tokens_remainder = pred_fine_tokens[:, :nq], pred_fine_tokens[:, nq:]

        pred_fine_tokens_groupable = rearrange(pred_fine_tokens_groupable, 'b (n q) d -> b n q d', q = self.num_fine_quantizers)

        fine_logits_groupable = einsum('q c d, b n q d -> b n q c', self.fine_logit_weights, pred_fine_tokens_groupable)

        fine_logits_groupable = rearrange(fine_logits_groupable, 'b n q c -> b (n q) c')

        remainder_num_quantizers = pred_fine_tokens_remainder.shape[1]

        if remainder_num_quantizers > 0:
            fine_logits_remainder = einsum('q c d, b q d -> b q c', self.fine_logit_weights[:remainder_num_quantizers], pred_fine_tokens_remainder)

            fine_logits = torch.cat((fine_logits_groupable, fine_logits_remainder), dim = 1)
        else:
            fine_logits = fine_logits_groupable

        logits = (coarse_logits, fine_logits)

        if not return_cache:
            return logits

        return logits, (next_kv_cache, new_embed_cache)
