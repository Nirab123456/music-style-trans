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
        spectrogram: Tensor of shape (channels, freq, time) or (freq, time).
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

# --- New Transformations ---

def random_noise(spectrogram: torch.Tensor, noise_std: float = 0.05) -> torch.Tensor:
    """
    Add random Gaussian noise to the spectrogram.

    Args:
        spectrogram: Input spectrogram.
        noise_std: Standard deviation factor for noise relative to the maximum spectrogram value.
        
    Returns:
        Noisy spectrogram.
    """
    noise_scale: float = noise_std * spectrogram.max().item()
    noise = torch.randn_like(spectrogram) * noise_scale
    return spectrogram + noise

def random_distortion(
    spectrogram: torch.Tensor,
    gamma_range: Tuple[float, float] = (0.8, 1.2)
) -> torch.Tensor:
    """
    Apply a random nonlinear (gamma) distortion to the spectrogram.

    Args:
        spectrogram: Input spectrogram (assumed to be non-negative).
        gamma_range: Range for the gamma exponent.
        
    Returns:
        Gamma-distorted spectrogram.
    """
    gamma: float = random.uniform(*gamma_range)
    # Apply a power-law (gamma) transformation.
    return torch.pow(spectrogram, gamma)

def random_volume(
    spectrogram: torch.Tensor,
    volume_range: Tuple[float, float] = (0.8, 1.2)
) -> torch.Tensor:
    """
    Randomly scale the amplitude (volume) of the spectrogram.

    Args:
        spectrogram: Input spectrogram.
        volume_range: Tuple indicating the minimum and maximum scaling factors.
        
    Returns:
        Volume-scaled spectrogram.
    """
    scale: float = random.uniform(*volume_range)
    return spectrogram * scale

# --- Composite Augmentation Pipeline ---

def augmentation_pipeline(spec: torch.Tensor) -> torch.Tensor:
    """
    Composite transformation to apply data augmentation.
    
    Sequentially applies:
      1. Random time crop (to 512 time frames)
      2. Random time stretch (with a factor between 0.9 and 1.1)
      3. Random pitch shift (with a shift between -1 and 1 semitones)
      4. Random noise addition
      5. Random nonlinear distortion (gamma correction)
      6. Random volume scaling
    
    Args:
        spec: Input spectrogram tensor.
    Returns:
        Augmented spectrogram.
    """
    spec = random_time_crop(spec, target_time=512)
    spec = random_time_stretch(spec, factor_range=(0.9, 1.1))
    spec = random_pitch_shift(spec, shift_range=(-1.0, 1.0))
    spec = random_noise(spec, noise_std=0.05)
    spec = random_distortion(spec, gamma_range=(0.8, 1.2))
    spec = random_volume(spec, volume_range=(0.8, 1.2))


    return spec
