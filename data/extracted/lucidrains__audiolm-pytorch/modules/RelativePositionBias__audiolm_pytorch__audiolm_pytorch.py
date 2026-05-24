# Extracted by kernel-synth
# Source: audiolm_pytorch/audiolm_pytorch.py (lines 202-242)
# Class: RelativePositionBias
# Tags: imports:einops, self-contained
# Novelty: 0.65
# Reason: file imports einops, einops, einops, einops, einops, einops

class RelativePositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(
        self,
        *,
        dim,
        heads,
        layers = 3
    ):
        super().__init__()
        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(1, dim), nn.SiLU()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.SiLU()))

        self.net.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        assert j >= i
        device = self.device

        i_pos = torch.arange(i, device = device) + (j - i)
        j_pos = torch.arange(j, device = device)

        rel_pos = (rearrange(i_pos, 'i -> i 1') - rearrange(j_pos, 'j -> 1 j'))
        rel_pos += (j - 1)

        x = torch.arange(-j + 1, j, device = device).float()
        x = rearrange(x, '... -> ... 1')

        for layer in self.net:
            x = layer(x)

        x = x[rel_pos]
        return rearrange(x, 'i j h -> h i j')
