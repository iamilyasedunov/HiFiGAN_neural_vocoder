from utils.utils import *
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import itertools
from preprocessing.log_mel_spec import MelSpectrogram
from datetime import datetime


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


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
                nn.Conv1d(in_channels=d_p_config["in_channels"][idx],
                          out_channels=d_p_config["out_channels"][idx],
                          kernel_size=d_p_config["kernel_sizes"][idx],
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
        # print(f"PeriodDiscriminator: x shape: {x.shape}")
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
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
                          kernel_size=(d_s_config["kernel_sizes"][idx],),
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
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
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
        return x


class HiFiGAN:
    def __init__(self, config, writer):
        super().__init__()
        self.steps = 0
        self.metrics = {
            "loss_gen_all": [],
            "mel_error": [],
            "loss_mel": [],
        }
        self.featurizer = MelSpectrogram(config)
        self.config = config
        self.writer = writer
        self.generator = Generator(config)
        self.msd = MultiScaleDiscriminator()
        self.mpd = MultiPeriodDiscriminator()

        self.generator = self.generator.to(config.device)
        self.msd = self.msd.to(config.device)
        self.mpd = self.mpd.to(config.device)

        self.optim_g = torch.optim.AdamW(self.generator.parameters(), config.learning_rate,
                                         betas=[config.adam_b1, config.adam_b2])
        self.optim_d = torch.optim.AdamW(itertools.chain(self.msd.parameters(), self.mpd.parameters()),
                                         config.learning_rate, betas=[config.adam_b1, config.adam_b2])

        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=config.lr_decay)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=config.lr_decay)

    def train_step(self, batch):
        self.steps += 1
        batch = batch.to(self.config.device)
        x, y = batch.mel, batch.waveform

        y_g_hat = self.generator(x)

        y_g_hat_mel = self.featurizer(y_g_hat.squeeze(1).cpu()).to(self.config.device)
        y = y.unsqueeze(1)
        self.optim_d.zero_grad()

        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_g_hat.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = self.discriminator_loss(y_df_hat_r, y_df_hat_g)

        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_g_hat.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = self.discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_disc_all = loss_disc_s + loss_disc_f
        loss_disc_all.backward()
        self.optim_d.step()

        self.optim_g.zero_grad()
        loss_mel = F.l1_loss(x, y_g_hat_mel) * 45
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
        loss_fm_f = self.feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = self.feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = self.generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = self.generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        loss_gen_all.backward()

        self.optim_g.step()
        self.metrics["loss_mel"].append(loss_mel.item())
        self.metrics["loss_gen_all"].append(loss_gen_all.item())
        self.metrics["y_g_hat_mel"] = y_g_hat_mel
        self.metrics["true_mel"] = x

    def train_logging(self):
        if self.steps % self.config.log_step == 0:
            to_log = {'Step' : self.steps,
                      'loss_mel': np.mean(self.metrics["loss_mel"]),
                      'loss_gen_all': np.mean(self.metrics["loss_gen_all"])}
            self.writer.set_step(self.steps)
            self.writer.add_spectrogram("true_mel", self.metrics["true_mel"])
            self.writer.add_spectrogram("gen_mel", self.metrics["y_g_hat_mel"].detach())

            self.writer.add_scalars("train", to_log)

            self.metrics["loss_mel"] = []
            self.metrics["loss_gen_all"] = []

            print(to_log)

    def validation(self, val_dataloader):
        self.generator.eval()
        torch.cuda.empty_cache()
        val_err_tot = 0
        with torch.no_grad():
            for j, batch in tqdm(enumerate(val_dataloader), desc="val",
                                 total=len(val_dataloader)):
                batch = batch.to(self.config.device)
                x, y = batch.mel, batch.waveform
                y_g_hat = self.generator(x.to(self.config.device))
                y_g_hat_mel = self.featurizer(y_g_hat.squeeze(1).cpu()).to(self.config.device)
                val_err_tot += F.l1_loss(x, y_g_hat_mel).item()

                if j % self.config.val_log_step == 0:
                    self.writer.set_step(self.steps)
                    # if self.steps == 0:
                    self.writer.add_audio('gt/y_{}'.format(j), y[0], self.config.sampling_rate)
                    self.writer.add_spectrogram('gt/y_spec_{}'.format(j), x)

                    self.writer.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], self.config.sampling_rate)
                    y_hat_spec = self.featurizer(y_g_hat.squeeze(1).cpu())
                    self.writer.add_spectrogram('generated/y_hat_spec_{}'.format(j), y_hat_spec)
            self.writer.set_step(self.steps)
            val_err = val_err_tot / (j + 1)
            self.writer.add_scalar("validation/mel_spec_error", val_err)
        self.generator.train()

    def sched_step(self):
        self.scheduler_g.step()
        self.scheduler_d.step()

    @staticmethod
    def feature_loss(fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss * 2

    @staticmethod
    def generator_loss(disc_outputs):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean((1 - dg) ** 2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses

    @staticmethod
    def discriminator_loss(disc_real_outputs, disc_generated_outputs):
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg ** 2)
            loss += (r_loss + g_loss)
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses

    def to_train(self):
        self.generator.train()
        self.msd.train()
        self.mpd.train()
