import torch
from typing import Callable, List , Optional
import torchaudio
import configarations
import configarations.global_initial_config as GI

# ------------------------------
# Basic Utility Functions
# ------------------------------

def to_stereo(waveform: torch.Tensor) -> torch.Tensor:
    """Ensure the waveform is stereo (2 channels). If mono, duplicate the channel."""
    return waveform.repeat(2, 1) if waveform.size(0) == 1 else waveform[:2]

# we can also add additional arguments to compute_spectrogram
def compute_spectrogram(
    waveform: torch.Tensor,
    n_fft: int = GI.N_FFT,
    hop_length: int = GI.HOP_LENGTH,
    power : float = None,
    normalized : bool = False
) -> torch.Tensor:
    """Compute the magnitude spectrogram using torchaudio.functional.spectrogram."""
    window = torch.hann_window(n_fft, device=waveform.device)
    spec = torchaudio.functional.spectrogram(
        waveform=waveform,
        pad=0,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        power=power,  
        normalized=normalized,
    )

    return spec  # .abs() or .abs()**2 for power

class ComputeSpectrogram:
    def __init__(self, n_fft: int = 2048, hop_length: int = 512, power: float = None, normalized: bool = False):
        """
        Compute the magnitude spectrogram using torchaudio.functional.spectrogram.
        
        Args:
            n_fft (int): The FFT window size.
            hop_length (int): The hop length for the window.
            power (float, optional): If None, returns a complex tensor and we take the magnitude.
                                      Otherwise, returns the power spectrogram.
            normalized (bool): If True, the spectrogram is normalized.
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.normalized = normalized

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform (Tensor): Audio tensor. Expected shapes are (time), (channel, time) or (batch, channel, time).
        
        Returns:
            Tensor: The computed magnitude spectrogram with shape (channel, freq, time) (or with batch dimension if provided).
        """

        spec = compute_spectrogram(waveform,self.n_fft,self.hop_length,self.power,self.normalized)
        

        if self.power is None:
            return spec.abs()
        else:
            return spec

def compute_waveform(
    mag_spec: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
    num_iters: int = 64
) -> torch.Tensor:
    """Reconstruct waveform from magnitude spectrogram using Griffin-Lim."""
    window = torch.hann_window(n_fft).to(mag_spec.device)

    spec_power = mag_spec ** 2  # Griffin-Lim uses power spectrogram

    waveform = torchaudio.functional.griffinlim(
        specgram=spec_power,
        window=window,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        power=2.0,
        n_iter=num_iters,
        momentum=0.99,
        length=None,         # Let Griffin-Lim infer length
        rand_init=True       # Start with random phase
    )
    return waveform

def reconstruct_waveform_from_complex_spec(
    complex_spec: torch.Tensor,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window_fn: Optional[callable] = None,
    normalized: bool = False,
    center: bool = True,
    pad_mode: str = 'reflect',
    onesided: bool = True,
    length: Optional[int] = None
) -> torch.Tensor:
    """
    Reconstruct a time-domain waveform from a complex STFT tensor
    using torchaudio.transforms.InverseSpectrogram.

    Args:
        complex_spec (Tensor): Complex-valued STFT of shape (..., freq, time).
        n_fft (int): FFT size used for the original STFT.
        hop_length (int, optional): hop length used for STFT (defaults to win_length//2).
        win_length (int, optional): window length (defaults to n_fft).
        window_fn (callable, optional): window function, defaults to torch.hann_window.
        normalized (bool): whether the original STFT was normalized.
        center (bool): whether the STFT frames were centered.
        pad_mode (str): padding mode used when centering.
        onesided (bool): whether the original STFT was one-sided.
        length (int, optional): expected length of the output waveform.

    Returns:
        Tensor: real-valued waveform of shape (..., time).
    """
    if not torch.is_complex(complex_spec):
        raise ValueError("`complex_spec` must be a complex tensor (dtype=torch.cfloat/complex64 or complex128).")

    # Default hop_length and win_length
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 2

    # Default window function
    if window_fn is None:
        window_fn = torch.hann_window

    # Instantiate the inverse transform
    inv_spec = torchaudio.transforms.InverseSpectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        pad=0,
        window_fn=window_fn,
        normalized=normalized,
        wkwargs=None,
        center=center,
        pad_mode=pad_mode,
        onesided=onesided,
    )

    # Apply it
    # –– spec should have shape (..., freq, time)
    # –– output will be (..., time)
    waveform = inv_spec(complex_spec, length)

    return waveform
# ------------------------------
# Compose Class for Transformations
# ------------------------------

class Compose:
    def __init__(self, transforms: List[Callable[[torch.Tensor], torch.Tensor]]):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x

    def __iter__(self):
        return iter(self.transforms)

