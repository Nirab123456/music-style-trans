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

def random_crop_2d(
    tensor: torch.Tensor,
    size: int
) -> torch.Tensor:
    """
    Randomly crop a 2D tensor along the time axis to the given size.

    Args:
        tensor (torch.Tensor): The input tensor to crop. Expected shape [C, T].
        size (int): The desired time length after cropping.

    Returns:
        torch.Tensor: The randomly cropped tensor.
    """
    if tensor.ndim != 2:
        raise ValueError("Expected a 2D tensor with shape [C, T]")
    
    c, t = tensor.shape
    if size > t:
        raise ValueError(f"Crop size {size} is larger than tensor time dimension {t}")

    start = random.randint(0, t - size)
    end = start + size
    cropped_tensor = tensor[:, start:end]

    return cropped_tensor


# #updated Random noise for absolute realworld noise simulation 
class RandomAbsoluteNoise_wav:
    def __init__(
        self,
        noise_std: float = 0.05,
        noise_tensor_path=f"{global_initial_config.NOISE_TENSOR_SAVE_DIR}/{global_initial_config.NOISE_TENSOR_NAME}"
    ):
        """updated Random noise for absolute realworld noise simulation"""
        self.noise_std = noise_std
        self.noise_tensor = torch.load(noise_tensor_path)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        length_of_first_dim : int = waveform.shape[1]

        noise_tensor: torch.Tensor = random_crop_2d(self.noise_tensor, length_of_first_dim)

        # Scale noise by noise_std and add to spec directly
        wav: torch.Tensor = waveform + (noise_tensor * self.noise_std)
        # sape = wav.shape

        return wav

