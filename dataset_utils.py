import os
import pandas as pd
from tqdm import tqdm
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from typing import Tuple, Union, List, Callable, Optional, Dict
import pathlib
from itertools import islice
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import dataclasses
import random
from preprocessing.log_mel_spec import mel_spectrogram


class DatasetDownloader():

    def __init__(self, datadir="./LJSpeech-1.1/"):
        self.datadir = datadir

        if os.path.isfile('LJSpeech-1.1.tar.bz2'):
            print('Data is already downloaded.')
        else:
            print('Downloading data...')
            os.system(
                'wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -O LJSpeech-1.1.tar.bz2')
            os.system('tar -xjf LJSpeech-1.1.tar.bz2 > log')
            print("Ready!")


@dataclasses.dataclass
class Batch:
    mel: torch.Tensor
    waveform: torch.Tensor
    mel_loss: torch.Tensor

    def to(self, device: torch.device) -> 'Batch':
        mel = self.mel.to(device)
        waveform = self.waveform.to(device)
        mel_loss = self.mel_loss.to(device)
        return Batch(mel, waveform, mel_loss)


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root, config):
        super().__init__(root=root)
        self.config = config
        # self.config.fmax_loss = None
        # self.featurizer = featurizer

    def __getitem__(self, index: int):
        audio, _, _, _ = super().__getitem__(index)
        if audio.size(1) >= self.config.segment_size:
            max_audio_start = audio.size(1) - self.config.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio = audio[:, audio_start:audio_start + self.config.segment_size]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.config.segment_size - audio.size(1)), 'constant')

        mel = mel_spectrogram(audio, self.config.n_fft, self.config.num_mels,
                              self.config.sampling_rate, self.config.hop_size, self.config.win_size,
                              self.config.fmin, self.config.fmax, center=False)

        mel_loss = mel_spectrogram(audio, self.config.n_fft, self.config.num_mels,
                                   self.config.sampling_rate, self.config.hop_size,
                                   self.config.win_size, self.config.fmin, self.config.fmax_for_loss,
                                   center=False)
        # print(f"shapes mel:{mel.shape}, audio: {audio.shape}, mel_loss: {mel_loss.shape}")
        # print(f"LJSpeechDataset | shapes mel:{mel.squeeze().shape}, audio: {audio.squeeze(0).shape}, mel_loss: {mel_loss.squeeze().shape}")

        return mel, audio, mel_loss


class LJSpeechCollator:

    def __call__(self, instances: List[Tuple]) -> 'Batch':
        mel, waveform, mel_loss = list(
            zip(*instances)
        )
        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)

        mel = pad_sequence([
            mel_[0] for mel_ in mel
        ])

        mel_loss = pad_sequence([
            mel_loss_[0] for mel_loss_ in mel_loss
        ])

        mel = torch.permute(mel, (1, 0, 2))
        mel_loss = torch.permute(mel_loss, (1, 0, 2))
        print(f"collator| shapes mel:{mel.shape},"
              f"audio: {waveform.shape}, mel_loss: {mel_loss.shape}")

        # print(f"shape mel: {mel.shape}")
        # print(f"wav mel: {waveform.shape}")
        return Batch(mel, waveform, mel_loss)
