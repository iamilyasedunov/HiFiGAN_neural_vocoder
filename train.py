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

set_seed(21)


def main(config):
    writer = get_writer(config)
    _ = DatasetDownloader(config.data_path)
    metrics = {
        "num_steps": 0,
        "loss": [],
        "best_loss": 10000
    }
    # featurizer = MelSpectrogram(config)

    train_dataloader = DataLoader(LJSpeechDataset('../', config, train=True), batch_size=config.batch_size,
                                  collate_fn=LJSpeechCollator())
    val_dataloader = DataLoader(LJSpeechDataset('../', config, train=False), batch_size=config.batch_size,
                                collate_fn=LJSpeechCollator())

    model = HiFiGAN(config, writer)

    if config.checkpont_path:
        model.load_checkpoint()

    model.to_train()
    for epoch in range(config.training_epoch):
        for i, batch in tqdm(enumerate(train_dataloader), desc="train",
                             total=min(len(train_dataloader), config.len_data)):
            if i > config.len_data:
                break

            metrics["num_steps"] += 1
            model.train_step(batch)
            model.train_logging()
            if metrics["num_steps"] % config.val_step == 0 and metrics["num_steps"] != 1:
                val_loss = model.validation(val_dataloader)
                if val_loss < metrics["best_loss"]:
                    metrics["best_loss"] = val_loss
                    model.save_checkpoint(epoch)
        model.sched_step()


if __name__ == "__main__":
    config_ = TaskConfig()
    main(config_)
