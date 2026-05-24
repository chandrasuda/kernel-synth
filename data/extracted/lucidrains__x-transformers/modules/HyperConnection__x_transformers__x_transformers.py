# Extracted by kernel-synth
# Source: x_transformers/x_transformers.py (lines 1277-1363)
# Class: HyperConnection
# Tags: custom-init, einsum, imports:einops, self-contained
# Novelty: 0.88
# Reason: forward uses einsum; file imports einops, einops, einops, einops, einops, einops, einops, einops; non-trivial parameter init

class HyperConnection(Module):
    def __init__(
        self,
        dim,
        *,
        layer_index,
        num_residual_streams,
        num_input_views = 1,
        sinkhorn_iters = 5,
        **kwargs
    ):
        """
        https://arxiv.org/abs/2409.19606
        Appendix J - Algorithm 2, Dynamic only

        https://arxiv.org/abs/2512.24880
        "Manifold constrained" mixing matrices
        """
        super().__init__()

        self.norm = nn.LayerNorm(dim, bias = False)

        self.num_residual_streams = num_residual_streams
        self.layer_index = layer_index

        self.static_beta = nn.Parameter(torch.ones(num_residual_streams))

        init_alpha0 = torch.zeros((num_residual_streams, num_input_views))
        init_alpha0[layer_index % num_residual_streams, :] = 1.

        self.static_alpha = nn.Parameter(cat([init_alpha0, torch.eye(num_residual_streams)], dim = 1))

        self.dynamic_alpha_fn = nn.Parameter(torch.zeros(dim, num_residual_streams + num_input_views))
        self.dynamic_alpha_scale = nn.Parameter(torch.ones(()) * 1e-2)

        self.num_input_views = num_input_views

        self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim))
        self.dynamic_beta_scale = nn.Parameter(torch.ones(()) * 1e-2)

        self.sinkhorn_iters = sinkhorn_iters

    def prepare(self, residuals):
        views = self.num_input_views
        streams = self.num_residual_streams

        residuals = rearrange(residuals, '(b s) n d -> b n s d', s = self.num_residual_streams)

        normed = self.norm(residuals)

        wc_weight = normed @ self.dynamic_alpha_fn
        dynamic_alpha = wc_weight * self.dynamic_alpha_scale
        alpha = dynamic_alpha + self.static_alpha

        alpha_input, alpha_residual = alpha[..., :views], alpha[..., views:]

        alpha_input = alpha_input.sigmoid() # constraint Hpre

        # the sinkhorn knopps constraint for the residual mixing

        alpha_residual = rearrange(alpha_residual, '... (s1 s2) -> ... s1 s2', s2 = streams)
        alpha_residual = sinkhorn(alpha_residual, self.sinkhorn_iters)
        alpha_residual = rearrange(alpha_residual, '... s1 s2 -> ... (s1 s2)')

        alpha = cat((alpha_input, alpha_residual), dim = -1)

        dc_weight = (normed @ self.dynamic_beta_fn).sigmoid() * 2
        dynamic_beta = dc_weight * self.dynamic_beta_scale
        beta = dynamic_beta + self.static_beta

        beta = beta.sigmoid() * 2 # constraint Hpost

        # width connection

        mix_h = einsum('... s t, ... s d -> ... t d', alpha, residuals)

        if views == 1:
            branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]
        else:
            branch_input, residuals = mix_h[..., :views, :], mix_h[..., views:, :]
            branch_input = rearrange(branch_input, '... v d -> v ... d')

        return branch_input, residuals, dict(beta = beta)

    def forward(self, x, residuals, *, beta, **kwargs):
        residuals = einsum('b n d, b n s -> b n s d', x, beta) + residuals
        return rearrange(residuals, 'b n s d -> (b s) n d')
