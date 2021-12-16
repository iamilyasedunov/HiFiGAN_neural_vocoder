import sys
from utils.utils import *
from dataclasses import dataclass
import torch
from torch import nn
import torchaudio
import librosa
from librosa.filters import mel as librosa_mel_fn
from utils.config import TaskConfig
from matplotlib import pyplot as plt
sys.path.append(sys.path[0] + "/..")

@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251


class MelSpectrogram(nn.Module):

    def __init__(self, config: TaskConfig):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sampling_rate,
            win_length=config.win_size,
            hop_length=config.hop_size,
            n_fft=config.n_fft,
            f_min=config.fmin,
            f_max=config.fmax,
            n_mels=config.num_mels,
            center=False,
        )
        # self.pad_size = (config.n_fft - config.hop_size) // 2
        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis_ = librosa.filters.mel(
            sr=config.sampling_rate,
            n_fft=config.n_fft,
            n_mels=config.num_mels,
            fmin=config.fmin,
            fmax=config.fmax
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis_))
        pad = int((config.n_fft - config.hop_size) / 2)
        self.pad = (pad, pad)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """
        audio = torch.nn.functional.pad(audio.unsqueeze(1), self.pad, mode='reflect').squeeze(1)
        mel = self.mel_spectrogram(audio).clamp_(min=1e-5).log_()

        return mel
