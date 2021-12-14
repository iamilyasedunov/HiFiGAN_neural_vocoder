from dataclasses import dataclass, field
import torch, copy
from typing import Tuple, Union, List, Callable, Optional, Dict


def default_field(obj):
    return field(default_factory=lambda: copy.copy(obj))


@dataclass
class TaskConfig:
    save_dir: str = ""
    log_dir: str = ""
    data_path: str = './LJSpeech-1.1/'
    verbosity: int = 2
    name: str = "TTS"
    log_step: int = 10
    exper_name: str = f"neural_vocoder_test_start"
    device: torch.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')

    sampling_rate: int = 22050
    segment_size: int = 8192
    num_mels: int = 80
    num_freq: int = 1025
    n_fft: int = 1024
    hop_size: int = 256
    win_size: int = 1024
    fmin: int = 0
    fmax: int = 8000
    fmax_for_loss: int = None

    # model config V3
    lrelu_negative_slope: float = 0.2
    input_channels: int = 80
    upsample_rates: List[int] = default_field([8, 8, 4])
    upsample_kernel_sizes: List[int] = default_field([16, 16, 8]) # ku
    resblock_kernel_sizes: List[int] = default_field([3, 5, 7]) # kr
    resblock_dilation_sizes: List[int] = default_field([[[1], [2]], [[2], [6]], [[3], [12]]]) # Dr
    upsample_initial_channel: int = 256 # hu

    num_gpus: int = 0
    batch_size: int = 16
    learning_rate: float = 0.0002
    adam_b1: float = 0.8
    adam_b2: float = 0.99
    lr_decay: float = 0.999

    training_epoch: int = 3000
