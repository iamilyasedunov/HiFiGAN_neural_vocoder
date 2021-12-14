from dataset_utils import DatasetDownloader, LJSpeechDataset, LJSpeechCollator
from preprocessing.log_mel_spec import MelSpectrogram, MelSpectrogramConfig
from model.model import HiFiGAN
from train_utils.utils import *
from logger import *
from itertools import islice
import warnings
from utils.config import TaskConfig
import itertools
from tqdm import tqdm
from preprocessing.log_mel_spec import mel_spectrogram

warnings.filterwarnings("ignore")

set_seed(21)


def main(config):
    writer = get_writer(config)
    _ = DatasetDownloader(config.data_path)

    # dataset = LJSpeechDataset('../')
    # print(dataset[0])
    # featurizer = MelSpectrogram(MelSpectrogramConfig())

    train_dataloader = DataLoader(LJSpeechDataset('../', config), batch_size=3, collate_fn=LJSpeechCollator())

    model = HiFiGAN(config)

    optim_g = torch.optim.AdamW(model.generator.parameters(), config.learning_rate,
                                betas=[config.adam_b1, config.adam_b2])

    model.to_train()

    for epoch in range(config.training_epoch):
        for i, batch in tqdm(enumerate(train_dataloader), desc="train", total=len(train_dataloader)):
            # print(batch.mel, batch.waveform)
            batch = batch.to(config.device)
            x, y = batch.mel, batch.waveform
            y_g_hat = model.generator(x)
            print(f"x.shape: {x.shape}")
            print(f"y.shape: {y.shape}")
            print(f"y_g_hat.shape: {y_g_hat.shape}")

            #y_g_hat_mel = featurizer(y_g_hat.cpu(), y.cpu()).to(config.device)

            y_g_hat_mel = mel_spectrogram(y_g_hat, config.n_fft, config.num_mels,
                                          config.sampling_rate, config.hop_size, config.win_size,
                                          config.fmin, config.fmax_for_loss)
            print(f"y_g_hat_mel.shape: {y_g_hat_mel.shape}")

            optim_g.zero_grad()
            if y_g_hat_mel.shape[2] > x.shape[2]:
                print(f"Incorrect shapes: y_g_hat_mel.shape: {y_g_hat_mel.shape[2]}, s.shape: {x.shape[2]}")
                raise Exception
            loss_mel = F.l1_loss(x, y_g_hat_mel)
            loss_mel.backward()
            optim_g.step()


if __name__ == "__main__":
    config_ = TaskConfig()
    # print(f"keyword: '{config.keyword}'\ndevice: {config.device}")
    main(config_)
