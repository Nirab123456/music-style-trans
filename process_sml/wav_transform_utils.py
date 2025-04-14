import torch
import random
from typing import Callable, List, Tuple
import torch.nn.functional as F
import torchaudio.functional as tf 
import torchaudio
import configarations.global_initial_config as global_initial_config
from torchvision.transforms import RandomCrop


# ------------------------------
# Individual Transformation Classes
# ------------------------------


#Highly memory expensive 
class RandomPitchShift_wav:
    def __init__(
        self,
        sample_rate: int = global_initial_config.SAMPLE_RATE,
        n_steps: int = 2,            # small semitone shift
        bins_per_octave: int = 12,
        n_fft: int = 512,
    ):
        self.sample_rate = sample_rate
        self.n_steps = n_steps
        self.bins_per_octave = bins_per_octave
        self.n_fft = n_fft

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform should be of shape [channels, time]
        shifted_waveform = torchaudio.functional.pitch_shift(
            waveform,
            sample_rate=self.sample_rate,
            n_steps=self.n_steps,
            bins_per_octave=self.bins_per_octave,
            n_fft=self.n_fft
        )
        return shifted_waveform
    
#optimally optimized     
class RandomVolume_wav:
    def __init__(self, volume_range: Tuple[float, float] = (0.8, 1.2)):
        self.volume_range = volume_range
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        scale: float = random.uniform(*self.volume_range)

        self.transform = torchaudio.transforms.Vol(gain=scale, gain_type="amplitude")


        return self.transform(waveform)
