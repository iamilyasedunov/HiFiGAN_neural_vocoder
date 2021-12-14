from utils.utils import *
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class PeriodDiscriminator(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(PeriodDiscriminator, self).__init__()
        self.period = period
        # norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        # period discriminator config

        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(2, 0)),
            nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0)),
            nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(2, 0)),
            nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0)),
            nn.Conv2d(1024, 1024, (kernel_size, 1), (1,), padding=(2, 0)),
        ])

        d_p_config = {
            "in_channels": [1, 32, 128, 512, 1024],
            "out_channels": [32, 128, 512, 1024, 1024],
            "kernel_sizes": [(kernel_size, 1)] * 5,
            "stride": [(stride, 1)] * 4 + [(1,)],
            "padding": [(2, 0)] * 5,
        }
        num_convs = len(d_p_config["in_channels"])
        self.conv_leaky_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=d_p_config["d_s_config"][idx],
                          out_channels=d_p_config["out_channels"][idx],
                          kernel_size=d_p_config["kernel_size"][idx],
                          stride=d_p_config["stride"][idx],
                          padding=d_p_config["padding"][idx]),
                nn.LeakyReLU(LRELU_NEGATIVE_SLOPE),
            )
            for idx in range(num_convs)
        ])

        self.conv_post = nn.Conv2d(1024, 1, (3, 1), (1,), padding=(1, 0))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.conv_leaky_layers:
            x = l(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ScaleDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(ScaleDiscriminator, self).__init__()
        # norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        # scale discriminator config
        d_s_config = {
            "in_channels": [1, 128, 128, 256, 512, 1024, 1024],
            "out_channels": [128, 128, 256, 512, 1024, 1024, 1024],
            "kernel_sizes": [15, 41, 41, 41, 41, 41, 5],
            "stride": [1, 2, 2, 4, 4, 1, 1],
            "padding": [7, 20, 20, 20, 20, 20, 2],
            "groups": [1, 4, 16, 16, 16, 16, 1],
        }
        num_convs = len(d_s_config["in_channels"])

        self.conv_leaky_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=d_s_config["in_channels"][idx],
                          out_channels=d_s_config["out_channels"][idx],
                          kernel_size=(d_s_config["kernel_size"][idx],),
                          stride=(d_s_config["stride"][idx],),
                          groups=d_s_config["groups"][idx],
                          padding=d_s_config["padding"][idx]),
                nn.LeakyReLU(LRELU_NEGATIVE_SLOPE),
            )
            for idx in range(num_convs)
        ])
        self.conv_output = nn.Conv1d(1024, 1, (3,), (1,), padding=1)

    def forward(self, x):
        fmap = []
        for layer in self.conv_leaky_layers:
            x = layer(x)
            fmap.append(x)
        x = self.conv_output(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, ):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


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
        for idx in range(len(self.mrf_layers)):
            x = x + self.mrf_layers[idx](x)
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
        # print(f"x(input): {x.shape}")
        x = self.input_conv(x)
        # print(f"x(input_conv): {x.shape}")
        x = self.generator(x)
        # print(f"x(generator): {x.shape}")
        x = self.output_layers(x)
        # print(f"x(output): {x.shape}")
        return x.squeeze()


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0][0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1][0])))
        ])
        #self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_NEGATIVE_SLOPE)
            xt = c(xt)
            x = xt + x
        return x


class Generator_(torch.nn.Module):
    def __init__(self, h):
        super(Generator_, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        #self.ups.apply(init_weights)
        #self.conv_post.apply(init_weights)

    def forward(self, x):
        # print(f"x(start)   : {x.shape}")
        x = self.conv_pre(x)
        # print(f"x(conv_pre): {x.shape}")
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_NEGATIVE_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
            # print(f"x(upsample): {x.shape}")
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        # print(f"x(conv_post): {x.shape}")

        x = torch.tanh(x)

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
