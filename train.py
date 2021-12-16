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

warnings.filterwarnings("ignore")

set_seed(21)


def main(config):
    writer = get_writer(config)
    _ = DatasetDownloader(config.data_path)
    metrics = {
        "num_steps": 0,
        "loss": [],
    }
    featurizer = MelSpectrogram(config)

    train_dataloader = DataLoader(LJSpeechDataset('../', config), batch_size=3, collate_fn=LJSpeechCollator())

    model = HiFiGAN(config)

    optim_g = torch.optim.AdamW(model.generator.parameters(), config.learning_rate,
                                betas=[config.adam_b1, config.adam_b2])

    model.to_train()
    for epoch in range(config.training_epoch):
        for i, batch in tqdm(enumerate(train_dataloader), desc="train", total=min(len(train_dataloader), config.len_data)):
            if i > config.len_data:
                break

            metrics["num_steps"] += 1
            batch = batch.to(config.device)
            x, y = batch.mel, batch.waveform
            y_g_hat = model.generator(x)

            y_g_hat_mel = featurizer(y_g_hat.cpu()).to(config.device)

            optim_g.zero_grad()

            loss_mel = F.l1_loss(x, y_g_hat_mel)
            loss_mel.backward()
            optim_g.step()

            metrics["loss"].append(loss_mel.item())
            if metrics["num_steps"] % config.log_step == 0:
                to_log = {'mel_loss': np.mean(metrics["loss"])}
                writer.set_step(metrics["num_steps"])
                writer.add_spectrogram("true_mel", x)
                writer.add_spectrogram("gen_mel", y_g_hat_mel.detach())

                writer.add_scalars("train", {'mel_loss': np.mean(metrics["loss"])})
                metrics["loss"] = []
                print(to_log)


if __name__ == "__main__":
    config_ = TaskConfig()
    main(config_)
