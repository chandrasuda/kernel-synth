# Extracted by kernel-synth
# Source: audiolm_pytorch/soundstream.py (lines 92-140)
# Class: MultiScaleDiscriminator
# Tags: imports:einops, self-contained
# Novelty: 0.65
# Reason: file imports einops, einops, einops, einops, einops

class MultiScaleDiscriminator(Module):
    def __init__(
        self,
        channels = 16,
        layers = 4,
        groups = (4, 16, 64, 256),
        chan_max = 1024,
        input_channels = 1
    ):
        super().__init__()
        self.init_conv = nn.Conv1d(input_channels, channels, 15, padding = 7)
        self.conv_layers = ModuleList([])

        curr_channels = channels

        for _, group in zip(range(layers), groups):
            chan_out = min(curr_channels * 4, chan_max)

            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(curr_channels, chan_out, 41, stride = 4, padding = 20, groups = group),
                leaky_relu()
            ))

            curr_channels = chan_out

        self.final_conv = nn.Sequential(
            nn.Conv1d(curr_channels, curr_channels, 5, padding = 2),
            leaky_relu(),
            nn.Conv1d(curr_channels, 1, 3, padding = 1),
        )

    def forward(
        self,
        x,
        return_intermediates = False
    ):
        x = self.init_conv(x)
        intermediates = []

        for layer in self.conv_layers:
            x = layer(x)
            intermediates.append(x)

        out = self.final_conv(x)

        if not return_intermediates:
            return out

        return out, intermediates
