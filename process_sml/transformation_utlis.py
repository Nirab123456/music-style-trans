import torch
import torch.nn.functional as F
from typing import Callable, List , Optional,Tuple
import torchaudio
import configarations.global_initial_config as GI

# ------------------------------
# Basic Utility Functions
# ------------------------------

def to_stereo(waveform: torch.Tensor) -> torch.Tensor:
    """Ensure the waveform is stereo (2 channels). If mono, duplicate the channel."""
    return waveform.repeat(2, 1) if waveform.size(0) == 1 else waveform[:2]

def reconstruction_from_four_channel(spec):
    C = spec.size(0) // 2           # 4 // 2 == 2
    mag   = spec[0:C,  :,  :]       # → [2, 1025, 157]
    phase = spec[C: ,  :,  :]       # → [2, 1025, 157]
    complex_spec = torch.polar(mag, phase)
    reconstruction = reconstruct_waveform(complex_spec)
    return reconstruction

def compute_waveform_griffinlim_B(
    mag_spec: torch.Tensor,
    n_fft: int = None,
    hop_length: int = None,
    num_iters: int = 120
) -> torch.Tensor:
    """
    Reconstruct waveform(s) from magnitude spectrogram(s) using Griffin–Lim.

    Args:
        mag_spec: Tensor of shape (B, F, T) or (B, C, F, T).
        n_fft: FFT size. If None, inferred as (F-1)*2.
        hop_length: Hop length. If None, set to n_fft//4.
        num_iters: Number of Griffin–Lim iterations.

    Returns:
        Tensor of waveforms with shape
          - (B, signal_length) if mag_spec was 3-D
          - (B, C, signal_length) if mag_spec was 4-D
    """
    # Unpack batch (and optional channel) dims
    orig_shape = mag_spec.shape
    if mag_spec.dim() == 4:
        B, C, F, T = orig_shape
        specs = mag_spec.view(B * C, F, T)
    elif mag_spec.dim() == 3:
        B, F, T = orig_shape
        C = None
        specs = mag_spec
    else:
        raise ValueError(f"mag_spec must be 3‑D or 4‑D, got {mag_spec.dim()}‑D")

    # Infer FFT/hop_length if omitted
    if n_fft is None:
        n_fft = (F - 1) * 2
    if hop_length is None:
        hop_length = n_fft // 4

    window = torch.hann_window(n_fft, device=mag_spec.device)
    power_spec = specs.pow(2)

    # Run Griffin–Lim in one batch call
    waveforms = torchaudio.functional.griffinlim(
        specgram=power_spec,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        window=window,
        power=2.0,
        n_iter=num_iters,
        momentum=0.99,
        length=None,
        rand_init=True,
    )

    # Reshape back to (B, C, T') if needed
    if C is not None:
        waveforms = waveforms.view(B, C, -1)

    return waveforms

def compute_waveform_griffinlim(
    mag_spec: torch.Tensor,
    n_fft: int = None,
    hop_length: int = None,
    num_iters: int = 120
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

def batch_reconstruct_waveform(
    complex_spec: torch.Tensor,
    n_fft: int = GI.N_FFT,
    hop_length: int = GI.HOP_LENGTH,
    win_length: int = None,
    window_fn: callable = None,
    center: bool = True,
    pad_mode: str = "reflect",
    normalized: bool = False,
    onesided: bool = True,
    length: int = None,
) -> torch.Tensor:
    """
    Reconstruct a batch of waveforms from complex spectrograms on GPU.

    Args:
        complex_spec (B×C×F×T, complex): input STFTs.
        n_fft (int): FFT size.
        hop_length (int): hop size.
        win_length (int, optional): window length (defaults to n_fft).
        window_fn (callable, optional): fn(win_length)->window Tensor.
        center, pad_mode, normalized, onesided: passed to torch.istft.
        length (int, optional): output length in samples.

    Returns:
        waveforms (B×C×L, float): reconstructed time-domain signals.
    """
    length = length if length is not None else GI.WAV_LENGTH
    B, C, F, T = complex_spec.shape
    # flatten batch & channel dims so istft sees a 3D input (N, F, T)
    spec_flat = complex_spec.reshape(B * C, F, T)

    # set defaults
    if win_length is None:
        win_length = n_fft
    if window_fn is None:
        window = torch.hann_window(win_length, device=complex_spec.device)
    else:
        # assume window_fn takes (win_length,) and returns a Tensor
        window = window_fn(win_length, device=complex_spec.device)

    # run a single istft over the flattened batch
    wave_flat = torch.istft(
        spec_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        normalized=normalized,
        onesided=onesided,
        length=length,
    )  # shape: (B*C, L)

    # reshape back to (B, C, L)
    L = wave_flat.size(-1)
    waveforms = wave_flat.view(B, C, L)
    return waveforms

def reconstruct_waveform(
    complex_spec: torch.Tensor,
    n_fft: int = GI.N_FFT,
    hop_length: int = GI.HOP_LENGTH,
    win_length: int = None,
    window_fn: callable = None,
    center: bool = True,
    pad_mode: str = "reflect",
    normalized: bool = False,
    onesided: bool = True,
    length: int = None
) -> torch.Tensor:
    """
    Perfectly invert a complex spectrogram, as long as all parameters
    match your forward STFT exactly.
    """
    if win_length is None:
        win_length = n_fft
    if window_fn is None:
        # must match your forward window
        window = torch.hann_window(win_length, device=complex_spec.device)
    else:
        window = window_fn(win_length, device=complex_spec.device)

    # torch.istft accepts both batched and channel‐first inputs:
    #   if your spec is [C, F, T], you’ll get [C, time] out
    waveform = torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        normalized=normalized,
        onesided=onesided,
        length=length
    )
    return waveform


def adjust_phase_shape(phase: torch.Tensor, target_shape:Tuple[int, int]) -> torch.Tensor:
    """
    Adjust the last two dimensions of a phase tensor to match target_shape.
    
    Args:
        phase (torch.Tensor): Phase tensor of shape [channels, n_freq, time].
        target_shape (Tuple[int, int]): Desired (n_freq, time) shape.
    
    Returns:
        torch.Tensor: Phase tensor resized to [channels, *target_shape].
    """
    current_shape = phase.shape[-2:]
    if current_shape == target_shape:
        return phase

    # Unsqueeze a batch dim so interpolate works on 4D ([1, C, F, T])
    phase_adjusted = F.interpolate(
        phase.unsqueeze(0),
        size=target_shape,
        mode='bilinear',
        align_corners=False
    )
    return phase_adjusted.squeeze(0)


def adjust_spec_shape(spec: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Adjust the last two dimensions of a spectrogram tensor.
    
    Args:
        spec (torch.Tensor): Spectrogram of shape [channels, n_freq, time].
        target_shape (Tuple[int, int]): The desired (n_freq, time) shape.
    
    Returns:
        torch.Tensor: Adjusted spectrogram.
    """
    target_shape = target_shape
    current_shape = spec.shape[-2:]
    if current_shape == target_shape:
        return spec

    # spec.unsqueeze(0) adds a batch dimension so that interpolate works on 4D data.
    spec_adjusted = F.interpolate(
        spec.unsqueeze(0),  # shape: [1, channels, n_freq, time]
        size=target_shape,
        mode='bilinear',
        align_corners=False
    )
    # Remove the added batch dimension
    return spec_adjusted.squeeze(0)

def get_shape_first_sample(waveform,hnn_window_cpu,n_fft,hop_length):
    waveform = to_stereo(waveform)
    wav_shape = waveform.shape
    #in 0th indaxing we want the size of first index [2,x]-> we want x first index 2 because we ensured stereo
    length = wav_shape[1]
    spec = torchaudio.functional.spectrogram(
                    waveform=waveform,
                    pad=0,
                    window=hnn_window_cpu,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    power=None,  
                    normalized=False,
                    win_length=n_fft,
                )
    spec = spec.abs()
    shape = spec.shape

    return shape , length


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

