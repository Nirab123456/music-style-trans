
import torch
import cython 
import random
from typing import Tuple 
import torch.nn.functional as F
# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def to_stereo(waveform: torch.Tensor) -> torch.Tensor:
    """Ensure the waveform is stereo (2 channels). If mono, duplicate the channel."""
    return waveform.repeat(2, 1) if waveform.size(0) == 1 else waveform[:2]

def compute_spectrogram(
    waveform: torch.Tensor,
    n_fft: cython.int = 2048,
    hop_length: cython.int = 512
) -> torch.Tensor:
    """Compute the magnitude spectrogram using torch.stft."""
    window = torch.hann_window(n_fft, device=waveform.device)
    spec: torch.Tensor = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    return spec.abs()

def random_time_crop(spectrogram: torch.Tensor, target_time: int) -> torch.Tensor:
    """
    Randomly crop the time dimension of the spectrogram.
    
    Args:
        spectrogram: Tensor of shape (channels, freq, time) or (freq, time)
        target_time: Number of time frames to crop.
    Returns:
        Cropped spectrogram.
    """
    if spectrogram.dim() == 2:
        freq, time = spectrogram.shape
        if time > target_time:
            start = random.randint(0, time - target_time)
            return spectrogram[:, start:start+target_time]
        return spectrogram
    elif spectrogram.dim() == 3:
        channels, freq, time = spectrogram.shape
        if time > target_time:
            start = random.randint(0, time - target_time)
            return spectrogram[:, :, start:start+target_time]
        return spectrogram
    else:
        raise ValueError("Unsupported tensor shape for random_time_crop.")

def random_time_stretch(
    spectrogram: torch.Tensor,
    factor_range: Tuple[float, float] = (0.9, 1.1)
) -> torch.Tensor:
    """
    Randomly time-stretch a spectrogram along its time dimension.
    
    Args:
        spectrogram: Tensor of shape (channels, freq, time).
        factor_range: Tuple specifying the minimum and maximum stretch factors.
    Returns:
        Time-stretched spectrogram of approximately the original time dimension.
    """
    factor: float = random.uniform(*factor_range)
    channels, freq, time = spectrogram.shape
    new_time: int = int(time * factor)
    stretched = F.interpolate(
        spectrogram.unsqueeze(0),
        size=(freq, new_time),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)
    if stretched.size(2) < time:
        return F.pad(stretched, (0, time - stretched.size(2)))
    else:
        return stretched[:, :, :time]

def random_pitch_shift(
    spectrogram: torch.Tensor,
    shift_range: Tuple[float, float] = (-1.0, 1.0)
) -> torch.Tensor:
    """
    Randomly pitch shift the spectrogram by scaling the frequency dimension.
    
    Args:
        spectrogram: Tensor of shape (channels, freq, time).
        shift_range: Tuple specifying the minimum and maximum pitch shift (in semitones).
    Returns:
        Pitch-shifted spectrogram.
    """
    semitones: float = random.uniform(*shift_range)
    factor: float = 2 ** (semitones / 12.0)
    channels, freq, time = spectrogram.shape
    new_freq: int = int(freq * factor)
    shifted = F.interpolate(
        spectrogram.unsqueeze(0),
        size=(new_freq, time),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)
    if shifted.size(1) < freq:
        return F.pad(shifted, (0, 0, 0, freq - shifted.size(1)))
    else:
        return shifted[:, :freq, :]

def augmentation_pipeline(spec: torch.Tensor) -> torch.Tensor:
    """
    Composite transformation to apply data augmentation:
      1. Random time crop to T frames.
      2. Random time stretch.
      3. Random pitch shift.
    """
    # Random time crop: crop to DEFAULT_AUDIO_PARAMS["T"] time frames.
    spec = random_time_crop(spec, target_time=512)
    # Random time stretch.
    spec = random_time_stretch(spec, factor_range=(0.9, 1.1))
    # Random pitch shift.
    spec = random_pitch_shift(spec, shift_range=(-1.0, 1.0))
    return spec
