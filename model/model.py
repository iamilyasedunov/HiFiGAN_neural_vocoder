from utils.utils import *
from torch.nn.utils import weight_norm


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlockCore(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(channels, channels, kernel_size, (1,), dilation=dilations[0],
                                           padding=get_padding(kernel_size, dilations[0][0])))
        self.conv2 = weight_norm(nn.Conv1d(channels, channels, kernel_size, (1,), dilation=dilations[1],
                                           padding=get_padding(kernel_size, dilations[1][0])))

    def forward(self, x):
        x_lrelu_conv1 = nn.LeakyReLU(LRELU_NEGATIVE_SLOPE)(x)
        x_lrelu_conv1 = self.conv1(x_lrelu_conv1) + x

        x_lrelu_conv2 = nn.LeakyReLU(LRELU_NEGATIVE_SLOPE)(x_lrelu_conv1)
        x_lrelu_conv2 = self.conv2(x_lrelu_conv2) + x_lrelu_conv1

        return x_lrelu_conv2


class MRF(nn.Module):
    def __init__(self, config, channels):
        super(MRF, self).__init__()
        Dr = config.resblock_dilation_sizes
        kr = config.resblock_kernel_sizes

        self.mrf_layers = nn.ModuleList()

        for l, (k_r, d_r) in enumerate(zip(kr, Dr)):
            self.mrf_layers.append(
                ResBlockCore(channels, k_r, d_r)
            )

    def forward(self, x):
        accum_x = None
        for idx in range(len(self.mrf_layers)):
            accum_x = self.mrf_layers[idx](x) if accum_x is None else accum_x + self.mrf_layers[idx](x)
        x = accum_x / len(self.mrf_layers)
        return x


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_conv = weight_norm(nn.Conv1d(config.input_channels, config.upsample_initial_channel,
                                                kernel_size=(7,), padding=3))
        hu = config.upsample_initial_channel
        generator_layers = []
        out_channels = None

        for l, (ku, ur) in enumerate(zip(config.upsample_kernel_sizes, config.upsample_rates)):
            in_channels = hu // (2 ** l)
            out_channels = hu // (2 ** (l + 1))
            generator_layers += [
                nn.LeakyReLU(LRELU_NEGATIVE_SLOPE),
                weight_norm(nn.ConvTranspose1d(in_channels, out_channels, ku, ur, padding=(ku - ur) // 2)),
                MRF(config, out_channels),
            ]
        assert out_channels is not None
        self.generator = nn.Sequential(*generator_layers)
        self.output_layers = nn.Sequential(
            nn.LeakyReLU(LRELU_NEGATIVE_SLOPE),
            weight_norm(nn.Conv1d(out_channels, 1, kernel_size=(7,), stride=(1,), padding=3)),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.generator(x)
        x = self.output_layers(x)
        return x.squeeze()


class HiFiGAN:
    def __init__(self, config):
        super().__init__()

        self.generator = Generator(config).to(config.device)
        # self.msd = MultiScaleDiscriminator(config).to(config.device)
        # self.mpd = MultiPeriodDiscriminator().to(config.device)

    def to_train(self):
        self.generator.train()
        # self.msd.train()
        # self.mpd.train()
