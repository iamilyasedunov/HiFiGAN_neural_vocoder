import os
import pandas as pd
from tqdm import tqdm
import torch
import csv
import torchaudio
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from typing import Tuple, Union, List, Callable, Optional, Dict
import pathlib
from itertools import islice
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import dataclasses
import random
from preprocessing.log_mel_spec import MelSpectrogram


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
    # mel_loss: torch.Tensor

    def to(self, device: torch.device) -> 'Batch':
        mel = self.mel.to(device)
        waveform = self.waveform.to(device)
        return Batch(mel, waveform)


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root, config, train=True):
        super().__init__(root=root)
        self.config = config
        # self.config.fmax_loss = None
        self.featurizer = MelSpectrogram(config)
        if train:
            self._flist = self._flist[:int(len(self._flist) * 0.90)]
            print(f"Len train manifest: {len(self._flist)}")
        else:
            self._flist = self._flist[int(len(self._flist) * 0.90):]
            print(f"Len val manifest: {len(self._flist)}")

    def __getitem__(self, index: int):
        audio, _, _, _ = super().__getitem__(index)
        if audio.size(1) >= self.config.segment_size:
            max_audio_start = audio.size(1) - self.config.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio = audio[:, audio_start:audio_start + self.config.segment_size]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.config.segment_size - audio.size(1)), 'constant')
        mel = self.featurizer(audio)
        return mel, audio


class LJSpeechCollator:
    def __call__(self, instances: List[Tuple]) -> 'Batch':
        mel, waveform = list(
            zip(*instances)
        )
        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)

        mel = pad_sequence([
            mel_[0] for mel_ in mel
        ])

        mel = torch.permute(mel, (1, 0, 2))

        return Batch(mel, waveform)
