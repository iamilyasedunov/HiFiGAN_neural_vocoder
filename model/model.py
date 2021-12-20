from utils.utils import *
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import itertools, torchaudio
from preprocessing.log_mel_spec import MelSpectrogram
from datetime import datetime
from utils.config import MAX_LEN


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.01)


def get_padding(k, d=1):
    return int((d * k - d) / 2)


class PeriodDiscriminator(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(PeriodDiscriminator, self).__init__()
        self.period = period

        # period discriminator config
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
                weight_norm(nn.Conv1d(in_channels=d_p_config["in_channels"][idx],
                                      out_channels=d_p_config["out_channels"][idx],
                                      kernel_size=d_p_config["kernel_sizes"][idx],
                                      stride=d_p_config["stride"][idx],
                                      padding=d_p_config["padding"][idx])),
                nn.LeakyReLU(LRELU_NEGATIVE_SLOPE),
            )
            for idx in range(num_convs)
        ])

        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), (1,), padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
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
    def __init__(self):
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
                weight_norm(nn.Conv1d(in_channels=d_s_config["in_channels"][idx],
                                      out_channels=d_s_config["out_channels"][idx],
                                      kernel_size=(d_s_config["kernel_sizes"][idx],),
                                      stride=(d_s_config["stride"][idx],),
                                      groups=d_s_config["groups"][idx],
                                      padding=d_s_config["padding"][idx])),
                nn.LeakyReLU(LRELU_NEGATIVE_SLOPE),
            )
            for idx in range(num_convs)
        ])
        self.conv_output = weight_norm(nn.Conv1d(1024, 1, (3,), (1,), padding=1))

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
            ScaleDiscriminator() for _ in range(3)
        ])
        self.meanpool = nn.AvgPool1d(4, 2, padding=2)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpool(y)
                y_hat = self.meanpool(y_hat)
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

    def remove_weight_norm(self):
        remove_weight_norm(self.conv1)
        remove_weight_norm(self.conv2)


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

    def remove_weight_norm(self):
        remove_weight_norm(self.input_conv)
        print('Removing weight norm...')
        for block in [self.generator, self.output_layers]:
            for layer in block:
                if len(layer.state_dict()) != 0:
                    try:
                        nn.utils.remove_weight_norm(layer)
                    except:
                        layer.remove_weight_norm()


class HiFiGAN:
    def __init__(self, config, writer):
        super().__init__()
        self.steps = 0
        self.metrics = {
            "loss_gen_all": [],
            "mel_error": [],
            "loss_mel": [],
        }
        self.last_val_loss = None
        self.best_loss = MAX_LEN
        self.featurizer_loss = MelSpectrogram(config, True)
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
        x, y, y_mel = batch.mel, batch.waveform, batch.mel_loss

        y_g_hat = self.generator(x)

        y_g_hat_mel = self.featurizer_loss(y_g_hat.squeeze(1).cpu()).to(self.config.device)
        y = y.unsqueeze(1)
        self.optim_d.zero_grad()

        loss_disc_all = self.get_disc_loss(y, y_g_hat.detach())
        loss_disc_all.backward()
        self.optim_d.step()

        self.optim_g.zero_grad()
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45
        loss_gen = self.get_gen_loss(y, y_g_hat)
        loss_gen_all = loss_gen + loss_mel

        loss_gen_all.backward()

        self.optim_g.step()
        self.metrics["loss_mel"].append(loss_mel.item())
        self.metrics["loss_gen_all"].append(loss_gen_all.item())
        self.metrics["y_g_hat_mel"] = y_g_hat_mel
        self.metrics["true_mel"] = x

    def train_logging(self):
        if self.steps % self.config.log_step == 0:
            to_log = {'Step': self.steps,
                      'loss_mel': np.mean(self.metrics["loss_mel"]),
                      'loss_gen_all': np.mean(self.metrics["loss_gen_all"])}
            self.writer.set_step(self.steps)
            self.writer.add_spectrogram("train/true_mel", self.metrics["true_mel"])
            self.writer.add_spectrogram("train/gen_mel", self.metrics["y_g_hat_mel"].detach())

            self.writer.add_scalars("train", to_log)

            self.metrics["loss_mel"] = []
            self.metrics["loss_gen_all"] = []

            print(to_log)

    def validation(self, val_dataloader, train=True):
        if not train:
            self.generator.remove_weight_norm()
        self.generator.eval()
        torch.cuda.empty_cache()
        val_err_tot = 0
        with torch.no_grad():
            for j, batch in tqdm(enumerate(val_dataloader), desc="val",
                                 total=len(val_dataloader), position=0, leave=True):
                batch = batch.to(self.config.device)
                x, y, y_mel = batch.mel, batch.waveform, batch.mel_loss
                y_g_hat = self.generator(x.to(self.config.device))
                y_g_hat_mel = self.featurizer_loss(y_g_hat.squeeze(1).cpu()).to(self.config.device)
                val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                if j % self.config.val_log_step == 0:
                    self.writer.set_step(self.steps)
                    # if self.steps == 0:
                    self.writer.add_audio('real/y', y[0], self.config.sampling_rate)
                    self.writer.add_spectrogram('real/y_spec', x)

                    self.writer.add_audio('generated/y_hat', y_g_hat[0], self.config.sampling_rate)
                    self.writer.add_spectrogram('generated/y_hat_spec', y_g_hat_mel.detach())
            for idx, path_file in enumerate(os.listdir(self.config.test_files_dir)):
                wav, sr = torchaudio.load(os.path.join(self.config.test_files_dir, path_file))
                x = self.featurizer(wav).to(self.config.device)
                y_g_hat = self.generator(x)
                audio = y_g_hat.squeeze()
                self.writer.add_audio(f'test/y_{idx}', audio, self.config.sampling_rate)
                self.writer.add_spectrogram(f'test/y_spec_{idx}', x)

            self.writer.set_step(self.steps)
            val_err = val_err_tot / (j + 1)
            print(f"Val mel_spec err: {val_err}")
            self.writer.add_scalar("validation/mel_spec_error", val_err)
        self.generator.train()
        self.last_val_loss = val_err
        return val_err

    def sched_step(self):
        self.scheduler_g.step()
        self.scheduler_d.step()

    def get_disc_loss(self, y, y_g_hat):
        mpd_real, mpd_gen, _, _ = self.mpd(y, y_g_hat.detach())
        msd_real, msd_gen, _, _ = self.msd(y, y_g_hat.detach())

        loss_mpd, loss_msd = 0, 0
        for (mpd_r, mpd_g, msd_r, msd_g) in zip(mpd_real, mpd_gen, msd_real, msd_gen):
            loss_mpd += (torch.mean((1 - mpd_r) ** 2) + torch.mean(mpd_g ** 2))
            loss_msd += (torch.mean((1 - msd_r) ** 2) + torch.mean(msd_g ** 2))
        return loss_mpd + loss_msd

    def get_gen_loss(self, y, y_g_hat):
        feature_loss, generator_loss = 0, 0

        mpd_real, mpd_gen, fmap_mpd_real, fmap_mpd_gen = self.mpd(y, y_g_hat.detach())
        msd_real, msd_gen, fmap_msd_real, fmap_msd_gen = self.msd(y, y_g_hat.detach())
        # feature loss
        for (f_mpd_r, f_mpd_g, f_msd_r, f_msd_g) in zip(fmap_mpd_real, fmap_mpd_gen,
                                                        fmap_msd_real, fmap_msd_gen):
            for f_mpd_r_, f_mpd_g_, f_msd_r_, f_msd_g_ in zip(f_mpd_r, f_mpd_g,
                                                              f_msd_r, f_msd_g):
                feature_loss += torch.mean(torch.abs(f_mpd_r_ - f_mpd_g_)) + \
                                torch.mean(torch.abs(f_msd_r_ - f_msd_g_))
        feature_loss = feature_loss * 2

        # generator loss
        for mpd_gen_, msd_gen_ in zip(mpd_gen, msd_gen):
            generator_loss += torch.mean((1 - mpd_gen_) ** 2) + torch.mean((1 - msd_gen_) ** 2)

        return feature_loss + generator_loss

    def to_train(self):
        self.generator.train()
        self.msd.train()
        self.mpd.train()

    def save_checkpoint(self, epoch):
        checkpoint_path = f"{self.config.save_dir}/model_{epoch}_{round(self.last_val_loss, 3)}.pth"
        print(f"Saving checkpoint: {checkpoint_path}")
        save_obj = {'generator': self.generator.state_dict(),
                    'mpd': self.mpd.state_dict(),
                    'msd': self.msd.state_dict(),
                    'optim_g': self.optim_g.state_dict(),
                    'optim_d': self.optim_d.state_dict(),
                    'steps': self.steps,
                    'epoch': epoch}
        torch.save(save_obj, checkpoint_path)

    def load_checkpoint(self):
        checkpoint_obj = torch.load(self.config.checkpont_path)
        self.generator.load_state_dict(checkpoint_obj['generator'])
        self.mpd.load_state_dict(checkpoint_obj['mpd'])
        self.msd.load_state_dict(checkpoint_obj['msd'])
        self.steps = checkpoint_obj['steps'] + 1
        self.optim_g.load_state_dict(checkpoint_obj['optim_g'])
        self.optim_d.load_state_dict(checkpoint_obj['optim_d'])
        last_epoch = checkpoint_obj['epoch']
        print(f"Checpoint loaded: {self.config.checkpont_path}")
