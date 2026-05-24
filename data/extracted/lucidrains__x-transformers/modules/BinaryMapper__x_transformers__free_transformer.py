# Extracted by kernel-synth
# Source: x_transformers/free_transformer.py (lines 59-128)
# Class: BinaryMapper
# Tags: einsum, imports:einops, math-heavy, self-contained, uses-buffers
# Novelty: 0.98
# Reason: forward uses einsum; ~8 arithmetic ops in forward; file imports einops, einops, einops, einops, einops, einops, einops, einops, einops, einops

class BinaryMapper(Module):
    def __init__(
        self,
        bits = 1,
        kl_loss_threshold = NAT # 1 bit
    ):
        super().__init__()

        self.bits = bits
        self.num_codes = 2 ** bits

        power_two = 2 ** arange(bits)
        codes = (arange(self.num_codes)[:, None].bitwise_and(power_two) != 0).byte().bool()

        self.register_buffer('power_two', power_two, persistent = False)
        self.register_buffer('codes', codes, persistent = False)

        # aux loss

        self.kl_loss_threshold = kl_loss_threshold
        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        logits,
        temperature = 1.,
        straight_through = None,
        calc_aux_loss = None
    ):
        straight_through = default(straight_through, self.training)
        calc_aux_loss = default(calc_aux_loss, self.training)

        assert logits.shape[-1] == self.bits, f'logits must have a last dimension of {self.bits}'

        # temperature and prob for sampling

        prob_for_sample = (logits / temperature).sigmoid()

        # sampling

        sampled_bits = (torch.rand_like(logits) <= prob_for_sample).long()
        indices = (self.power_two * sampled_bits).sum(dim = -1)

        one_hot = F.one_hot(indices, self.num_codes).float()

        # maybe calculate aux loss

        aux_kl_loss = self.zero

        if calc_aux_loss:
            # calculate negative entropy

            kl_div = self.bits * NAT - binary_entropy(logits)
            aux_kl_loss = F.relu(kl_div - self.kl_loss_threshold).mean()

        # maybe straight through

        if straight_through:
            # get the soft G for the gradients and do a straight through

            soft_G = (
                einsum(F.logsigmoid(logits), self.codes.float(), '... bits, codes bits -> ... codes') +
                einsum(F.logsigmoid(-logits), (~self.codes).float(), '... bits, codes bits -> ... codes')
            ).exp()

            # straight through

            one_hot = one_hot + soft_G - soft_G.detach()

        return one_hot, aux_kl_loss
