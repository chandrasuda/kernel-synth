# Extracted by kernel-synth
# Source: audiolm_pytorch/soundstream.py (lines 451-995)
# Class: SoundStream
# Tags: imports:einops, math-heavy, uses-buffers
# Novelty: 0.75
# Reason: ~16 arithmetic ops in forward; file imports einops, einops, einops, einops, einops

class SoundStream(Module):
    def __init__(
        self,
        *,
        channels = 32,
        strides = (2, 4, 5, 8),
        channel_mults = (2, 4, 8, 16),
        codebook_dim = 512,
        codebook_size: int | None = None,
        finite_scalar_quantizer_levels: list[int] | None = None,
        rq_num_quantizers = 8,
        rq_commitment_weight = 1.,
        rq_ema_decay = 0.95,
        rq_quantize_dropout_multiple_of = 1,
        rq_groups = 1,
        rq_stochastic_sample_codes = False,
        rq_rotation_trick = True,
        rq_kwargs: dict = {},
        use_lookup_free_quantizer = False,              # proposed in https://arxiv.org/abs/2310.05737, adapted for residual quantization
        use_finite_scalar_quantizer = False,            # proposed in https://arxiv.org/abs/2309.15505, adapted for residual quantization
        input_channels = 1,
        discr_multi_scales = (1, 0.5, 0.25),
        stft_normalized = False,
        enc_cycle_dilations = (1, 3, 9),
        dec_cycle_dilations = (1, 3, 9),
        multi_spectral_window_powers_of_two = tuple(range(6, 12)),
        multi_spectral_n_ffts = 512,
        multi_spectral_n_mels = 64,
        recon_loss_weight = 1.,
        multi_spectral_recon_loss_weight = 1e-5,
        adversarial_loss_weight = 1.,
        feature_loss_weight = 100,
        quantize_dropout_cutoff_index = 1,
        target_sample_hz = 16000,
        use_local_attn = True,
        attn_window_size = 128,
        attn_dim_head = 64,
        attn_heads = 8,
        attn_depth = 1,
        attn_xpos_scale_base = None,
        attn_dynamic_pos_bias = False,
        use_gate_loop_layers = False,
        squeeze_excite = False,
        complex_stft_discr_logits_abs = True,
        pad_mode = 'reflect',
        stft_discriminator: Module | None = None,  # can pass in own stft discriminator
        complex_stft_discr_kwargs: dict = dict()
    ):
        super().__init__()

        # for autosaving the config

        _locals = locals()
        _locals.pop('self', None)
        _locals.pop('__class__', None)
        self._configs = pickle.dumps(_locals)

        # rest of the class

        self.target_sample_hz = target_sample_hz # for resampling on the fly

        self.single_channel = input_channels == 1
        self.strides = strides

        layer_channels = tuple(map(lambda t: t * channels, channel_mults))
        layer_channels = (channels, *layer_channels)
        chan_in_out_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        encoder_blocks = []

        for ((chan_in, chan_out), layer_stride) in zip(chan_in_out_pairs, strides):
            encoder_blocks.append(EncoderBlock(chan_in, chan_out, layer_stride, enc_cycle_dilations, squeeze_excite, pad_mode))

            if use_gate_loop_layers:
                encoder_blocks.append(Residual(ChannelTranspose(GateLoop(chan_out, use_heinsen = False))))

        self.encoder = nn.Sequential(
            CausalConv1d(input_channels, channels, 7, pad_mode = pad_mode),
            *encoder_blocks,
            CausalConv1d(layer_channels[-1], codebook_dim, 3, pad_mode = pad_mode)
        )

        attn_kwargs = dict(
            dim = codebook_dim,
            dim_head = attn_dim_head,
            heads = attn_heads,
            depth = attn_depth,
            window_size = attn_window_size,
            xpos_scale_base = attn_xpos_scale_base,
            dynamic_pos_bias = attn_dynamic_pos_bias,
            prenorm = True,
            causal = True
        )

        self.encoder_attn = LocalTransformer(**attn_kwargs) if use_local_attn else None

        self.encoder_film = FiLM(codebook_dim, dim_cond = 2)

        self.num_quantizers = rq_num_quantizers

        self.codebook_dim = codebook_dim

        self.rq_groups = rq_groups

        assert not (use_lookup_free_quantizer and use_finite_scalar_quantizer)

        self.use_lookup_free_quantizer = use_lookup_free_quantizer
        self.use_finite_scalar_quantizer = use_finite_scalar_quantizer

        if use_lookup_free_quantizer:
            assert exists(codebook_size) and not exists(finite_scalar_quantizer_levels), 'if use_finite_scalar_quantizer is set to False, `codebook_size` must be set (and not `finite_scalar_quantizer_levels`)'

            self.rq = GroupedResidualLFQ(
                dim = codebook_dim,
                num_quantizers = rq_num_quantizers,
                codebook_size = codebook_size,
                groups = rq_groups,
                quantize_dropout = True,
                quantize_dropout_cutoff_index = quantize_dropout_cutoff_index,
                **rq_kwargs
            )

            self.codebook_size = codebook_size

        elif use_finite_scalar_quantizer:
            assert not exists(codebook_size) and exists(finite_scalar_quantizer_levels), 'if use_finite_scalar_quantizer is set to True, `finite_scalar_quantizer_levels` must be set (and not `codebook_size`). the effective codebook size is the cumulative product of all the FSQ levels'

            self.rq = GroupedResidualFSQ(
                dim = codebook_dim,
                levels = finite_scalar_quantizer_levels,
                num_quantizers = rq_num_quantizers,
                groups = rq_groups,
                quantize_dropout = True,
                quantize_dropout_cutoff_index = quantize_dropout_cutoff_index,
                **rq_kwargs
            )

            self.codebook_size = self.rq.codebook_size

        else:
            assert exists(codebook_size) and not exists(finite_scalar_quantizer_levels), 'if use_finite_scalar_quantizer is set to False, `codebook_size` must be set (and not `finite_scalar_quantizer_levels`)'
            self.rq = GroupedResidualVQ(
                dim = codebook_dim,
                num_quantizers = rq_num_quantizers,
                codebook_size = codebook_size,
                groups = rq_groups,
                decay = rq_ema_decay,
                commitment_weight = rq_commitment_weight,
                quantize_dropout_multiple_of = rq_quantize_dropout_multiple_of,
                kmeans_init = True,
                threshold_ema_dead_code = 2,
                quantize_dropout = True,
                quantize_dropout_cutoff_index = quantize_dropout_cutoff_index,
                stochastic_sample_codes = rq_stochastic_sample_codes,
                rotation_trick = rq_rotation_trick,
                **rq_kwargs
            )

            self.codebook_size = codebook_size

        self.decoder_film = FiLM(codebook_dim, dim_cond = 2)

        self.decoder_attn = LocalTransformer(**attn_kwargs) if use_local_attn else None

        decoder_blocks = []

        for ((chan_in, chan_out), layer_stride) in zip(reversed(chan_in_out_pairs), reversed(strides)):
            decoder_blocks.append(DecoderBlock(chan_out, chan_in, layer_stride, dec_cycle_dilations, squeeze_excite, pad_mode))

            if use_gate_loop_layers:
                decoder_blocks.append(Residual(ChannelTranspose(GateLoop(chan_in))))

        self.decoder = nn.Sequential(
            CausalConv1d(codebook_dim, layer_channels[-1], 7, pad_mode = pad_mode),
            *decoder_blocks,
            CausalConv1d(channels, input_channels, 7, pad_mode = pad_mode)
        )

        # discriminators

        self.discr_multi_scales = discr_multi_scales
        self.discriminators = ModuleList([MultiScaleDiscriminator() for _ in range(len(discr_multi_scales))])
        discr_rel_factors = [int(s1 / s2) for s1, s2 in zip(discr_multi_scales[:-1], discr_multi_scales[1:])]
        self.downsamples = ModuleList([nn.Identity()] + [nn.AvgPool1d(2 * factor, stride = factor, padding = factor) for factor in discr_rel_factors])

        self.stft_discriminator = stft_discriminator

        if not exists(self.stft_discriminator):
            self.stft_discriminator = ComplexSTFTDiscriminator(
                stft_normalized = stft_normalized,
                logits_abs = complex_stft_discr_logits_abs,  # whether to output as abs() or use view_as_real
                **complex_stft_discr_kwargs
            )

        # multi spectral reconstruction

        self.mel_spec_transforms = ModuleList([])
        self.mel_spec_recon_alphas = []

        num_transforms = len(multi_spectral_window_powers_of_two)
        multi_spectral_n_ffts = cast_tuple(multi_spectral_n_ffts, num_transforms)
        multi_spectral_n_mels = cast_tuple(multi_spectral_n_mels, num_transforms)

        for powers, n_fft, n_mels in zip_longest(multi_spectral_window_powers_of_two, multi_spectral_n_ffts, multi_spectral_n_mels):
            win_length = 2 ** powers
            alpha = (win_length / 2) ** 0.5

            calculated_n_fft = default(max(n_fft, win_length), win_length)  # @AndreyBocharnikov said this is usually win length, but overridable

            # if any audio experts have an opinion about these settings, please submit a PR

            melspec_transform = T.MelSpectrogram(
                sample_rate = target_sample_hz,
                n_fft = calculated_n_fft,
                win_length = win_length,
                hop_length = win_length // 4,
                n_mels = n_mels,
                normalized = stft_normalized
            )

            self.mel_spec_transforms.append(melspec_transform)
            self.mel_spec_recon_alphas.append(alpha)

        # loss weights

        self.recon_loss_weight = recon_loss_weight
        self.multi_spectral_recon_loss_weight = multi_spectral_recon_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.feature_loss_weight = feature_loss_weight

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def configs(self):
        return pickle.loads(self._configs)

    def decode_from_codebook_indices(self, quantized_indices):
        assert quantized_indices.dtype in (torch.long, torch.int32)

        if quantized_indices.ndim == 3:
            quantized_indices = rearrange(quantized_indices, 'b n (g q) -> g b n q', g = self.rq_groups)

        x = self.rq.get_output_from_indices(quantized_indices)

        return self.decode(x)

    def decode(self, x, quantize = False):
        if quantize:
            x, *_ = self.rq(x)

        if exists(self.decoder_attn):
            x = self.decoder_attn(x)

        x = rearrange(x, 'b n c -> b c n')
        return self.decoder(x)

    def save(self, path):
        path = Path(path)
        pkg = dict(
            model = self.state_dict(),
            config = self._configs,
            version = __version__
        )

        torch.save(pkg, str(path))

    @classmethod
    def init_and_load_from(cls, path, strict = True):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')

        assert 'config' in pkg, 'model configs were not found in this saved checkpoint'

        config = pickle.loads(pkg['config'])
        soundstream = cls(**config)
        soundstream.load(path, strict = strict)
        soundstream.eval()
        return soundstream

    def load(self, path, strict = True):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')

        # check version

        if 'version' in pkg and version.parse(pkg['version']) < parsed_version:
            print(f'soundstream model being loaded was trained on an older version of audiolm-pytorch ({pkg["version"]})')

        has_ema = 'ema_model' in pkg
        model_pkg = pkg['ema_model'] if has_ema else pkg['model']

        if has_ema:
            model_pkg = filter_by_keys(lambda k: k.startswith('ema_model.'), model_pkg)
            model_pkg = map_keys(lambda k: k[len('ema_model.'):], model_pkg)

        self.load_state_dict(model_pkg, strict = strict)

    def load_from_trainer_saved_obj(self, path):
        path = Path(path)
        assert path.exists()
        obj = torch.load(str(path))
        self.load_state_dict(obj['model'])

    def non_discr_parameters(self):
        return [
            *self.encoder.parameters(),
            *self.decoder.parameters(),
            *(self.encoder_attn.parameters() if exists(self.encoder_attn) else []),
            *(self.decoder_attn.parameters() if exists(self.decoder_attn) else []),
            *self.encoder_film.parameters(),
            *self.decoder_film.parameters(),
            *self.rq.parameters()
        ]

    @property
    def seq_len_multiple_of(self):
        return functools.reduce(lambda x, y: x * y, self.strides)

    @property
    def downsample_factor(self):
        return self.seq_len_multiple_of

    def process_input(
        self,
        x,
        input_sample_hz = None,
        curtail_from_left = False
    ):
        x, ps = pack([x], '* n')

        if exists(input_sample_hz):
            x = resample(x, input_sample_hz, self.target_sample_hz)

        x = curtail_to_multiple(x, self.seq_len_multiple_of, from_left = curtail_from_left)

        if x.ndim == 2:
            x = rearrange(x, 'b n -> b 1 n')

        return x, ps

    @torch.no_grad()
    def tokenize(self, audio):
        self.eval()
        return self.forward(audio, return_codes_only = True)

    def forward(
        self,
        x,
        target = None,
        is_denoising = None, # if you want to learn film conditioners that teach the soundstream to denoise - target would need to be passed in above
        return_encoded = False,
        return_codes_only = False,
        return_discr_loss = False,
        return_discr_losses_separately = False,
        return_loss_breakdown = False,
        return_recons_only = False,
        input_sample_hz = None,
        apply_grad_penalty = False,
        curtail_from_left = False
    ):
        assert not (exists(is_denoising) and not exists(target))

        process_input = partial(self.process_input, input_sample_hz = input_sample_hz, curtail_from_left = curtail_from_left)

        x, ps = process_input(x)

        if exists(target):
            target, _ = process_input(target)

        orig_x = x.clone()

        x = self.encoder(x)

        x = rearrange(x, 'b c n -> b n c')

        if exists(self.encoder_attn):
            x = self.encoder_attn(x)

        if exists(is_denoising):
            denoise_input = torch.tensor([is_denoising, not is_denoising], dtype = x.dtype, device = self.device) # [1, 0] for denoise, [0, 1] for not denoising
            x = self.encoder_film(x, denoise_input)

        if not self.use_finite_scalar_quantizer:
            x, indices, commit_loss = self.rq(x)
        else:
            # finite scalar quantizer does not have any aux loss

            x, indices = self.rq(x)
            commit_loss = self.zero

        if return_codes_only:
            return indices

        if return_encoded:
            indices = rearrange(indices, 'g b n q -> b n (g q)')
            return x, indices, commit_loss

        if exists(is_denoising):
            x = self.decoder_film(x, denoise_input)

        if exists(self.decoder_attn):
            x = self.decoder_attn(x)

        x = rearrange(x, 'b n c -> b c n')

        recon_x = self.decoder(x)

        if return_recons_only:
            recon_x, = unpack(recon_x, ps, '* c n')
            return recon_x

        # multi-scale discriminator loss

        if return_discr_loss:
            real, fake = orig_x, recon_x.detach()

            stft_discr_loss = None
            stft_grad_penalty = None
            discr_losses = []
            discr_grad_penalties = []

            if self.single_channel:
                real, fake = orig_x.clone(), recon_x.detach()
                stft_real_logits, stft_fake_logits = map(self.stft_discriminator, (real.requires_grad_(), fake.requires_grad_()))
                stft_discr_loss = hinge_discr_loss(stft_fake_logits, stft_real_logits)

                if apply_grad_penalty:
                    stft_grad_penalty = gradient_penalty(real, stft_discr_loss) + gradient_penalty(fake, stft_discr_loss)

            scaled_real, scaled_fake = real, fake
            for discr, downsample in zip(self.discriminators, self.downsamples):
                scaled_real, scaled_fake = map(downsample, (scaled_real, scaled_fake))

                real_logits, fake_logits = map(discr, (scaled_real.requires_grad_(), scaled_fake.requires_grad_()))
                one_discr_loss = hinge_discr_loss(fake_logits, real_logits)

                discr_losses.append(one_discr_loss)
                if apply_grad_penalty:
                    discr_grad_penalties.extend([
                        gradient_penalty(scaled_real, one_discr_loss),
                        gradient_penalty(scaled_fake, one_discr_loss)
                    ])

            if not return_discr_losses_separately:
                all_discr_losses = torch.stack(discr_losses).mean()

                if exists(stft_discr_loss):
                    all_discr_losses = all_discr_losses + stft_discr_loss

                if exists(stft_grad_penalty):
                    all_discr_losses = all_discr_losses + stft_grad_penalty

                return all_discr_losses

            # return a list of discriminator losses with List[Tuple[str, Tensor]]

            discr_losses_pkg = []

            discr_losses_pkg.extend([(f'scale:{scale}', multi_scale_loss) for scale, multi_scale_loss in zip(self.discr_multi_scales, discr_losses)])

            discr_losses_pkg.extend([(f'scale_grad_penalty:{scale}', discr_grad_penalty) for scale, discr_grad_penalty in zip(self.discr_multi_scales, discr_grad_penalties)])

            if exists(stft_discr_loss):
                discr_losses_pkg.append(('stft', stft_discr_loss))

            if exists(stft_grad_penalty):
                discr_losses_pkg.append(('stft_grad_penalty', stft_grad_penalty))

            return discr_losses_pkg

        # recon loss

        target = default(target, orig_x)  # target can also be passed in, in the case of denoising

        recon_loss = F.mse_loss(target, recon_x)

        # multispectral recon loss - eq (4) and (5) in https://arxiv.org/abs/2107.03312

        multi_spectral_recon_loss = self.zero

        if self.multi_spectral_recon_loss_weight > 0:
            for mel_transform, alpha in zip(self.mel_spec_transforms, self.mel_spec_recon_alphas):
                orig_mel, recon_mel = map(mel_transform, (orig_x, recon_x))
                log_orig_mel, log_recon_mel = map(log, (orig_mel, recon_mel))

                l1_mel_loss = (orig_mel - recon_mel).abs().sum(dim = -2).mean()
                l2_log_mel_loss = alpha * vector_norm(log_orig_mel - log_recon_mel, dim = -2).mean()

                multi_spectral_recon_loss = multi_spectral_recon_loss + l1_mel_loss + l2_log_mel_loss

        # adversarial loss

        adversarial_losses = []

        discr_intermediates = []

        # adversarial loss for multi-scale discriminators

        real, fake = orig_x, recon_x

        # features from stft

        (stft_real_logits, stft_real_intermediates), (stft_fake_logits, stft_fake_intermediates) = map(partial(self.stft_discriminator, return_intermediates=True), (real, fake))
        discr_intermediates.append((stft_real_intermediates, stft_fake_intermediates))

        scaled_real, scaled_fake = real, fake
        for discr, downsample in zip(self.discriminators, self.downsamples):
            scaled_real, scaled_fake = map(downsample, (scaled_real, scaled_fake))

            (real_logits, real_intermediates), (fake_logits, fake_intermediates) = map(partial(discr, return_intermediates = True), (scaled_real, scaled_fake))

            discr_intermediates.append((real_intermediates, fake_intermediates))

            one_adversarial_loss = hinge_gen_loss(fake_logits)
            adversarial_losses.append(one_adversarial_loss)

        feature_losses = []

        for real_intermediates, fake_intermediates in discr_intermediates:
            losses = [F.l1_loss(real_intermediate, fake_intermediate) for real_intermediate, fake_intermediate in zip(real_intermediates, fake_intermediates)]
            feature_losses.extend(losses)

        feature_loss = torch.stack(feature_losses).mean()

        # adversarial loss for stft discriminator

        adversarial_losses.append(hinge_gen_loss(stft_fake_logits))
        adversarial_loss = torch.stack(adversarial_losses).mean()

        # sum commitment loss

        all_commitment_loss = commit_loss.sum()

        total_loss = recon_loss * self.recon_loss_weight + multi_spectral_recon_loss * self.multi_spectral_recon_loss_weight + adversarial_loss * self.adversarial_loss_weight + feature_loss * self.feature_loss_weight + all_commitment_loss

        if return_loss_breakdown:
            return total_loss, (recon_loss, multi_spectral_recon_loss, adversarial_loss, feature_loss, all_commitment_loss)

        return total_loss
