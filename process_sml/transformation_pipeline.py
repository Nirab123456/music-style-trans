# my_pipeline.py
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import random

# Import helper functions and individual transform classes.
from .transformation_utlis import compute_spectrogram, to_stereo
from .wav_transform_utils import (
    RandomPitchShift_wav,
    RandomVolume_wav,
    RandomAbsoluteNoise_wav,
    RandomSpeed_wav,
    RandomFade_wav
)
from .spec_transform import (
    RandomFrequencyMasking_spec,
    RandomTimeMasking_spec,
    RandomTimeStretch_spec
)

class MyPipeline(nn.Module):
    def __init__(self,
                 wav_transforms: nn.Module = None,
                 spec_transforms: nn.Module = None):
        """
        A unified pipeline that applies both waveform- and spectrogram-level transforms.
        
        Args:
            use_complex_for_time_stretch (bool): If True, compute a complex spectrogram
                for transforms that require phase (e.g., time stretching).
            stft_params (dict): Parameters for computing the STFT (n_fft, hop_length, window).
            wav_transforms (nn.Module, optional): A module or sequential chain of waveform-level transforms.
            spec_transforms (nn.Module, optional): A module or sequential chain of spectrogram-level transforms.
        """
        super(MyPipeline, self).__init__()

        # Set default waveform transforms if none provided.
        if wav_transforms is None:
            self.wav_transforms = nn.Sequential(
                RandomVolume_wav(),        # adjust amplitude.
                RandomAbsoluteNoise_wav(), # add noise.
                RandomFade_wav()           # fade in/out.
            )
        else:
            self.wav_transforms = wav_transforms

        # Set default spectrogram transforms if none provided.
        if spec_transforms is None:
            self.spec_transforms = nn.ModuleList([
                RandomFrequencyMasking_spec(max_freq_mask_param=30, iid_masks=False),
                RandomTimeMasking_spec(max_time_mask_param=80, iid_masks=False, max_proportion=1.0),
                RandomTimeStretch_spec(n_freq=201, hop_length=256, rate_range=(0.8, 1.25))
            ])
        else:
            self.spec_transforms = spec_transforms




    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Applies the following steps:
          1. Waveform transforms (e.g., pitch shift, volume, noise).
          2. Compute a spectrogram (complex if required for time stretching).
          3. Spectrogram transforms (e.g., masking, time stretching).
        
        Args:
            waveform (torch.Tensor): Input tensor of shape (channels, time).
        
        Returns:
            torch.Tensor: The final transformed spectrogram.
        """
        # Ensure stereo format.
        waveform = to_stereo(waveform)
        
        # Apply waveform-level augmentations.
        if self.wav_transforms is not None:
            waveform = self.wav_transforms(waveform)
        
        # Decide on spectrogram type.
        if any(
            isinstance(t, RandomTimeStretch_spec) for t in self.spec_transforms
        ):
            spec = compute_spectrogram(waveform)

        else:
            spec = compute_spectrogram(waveform)
            spec = spec.abs()

        # Free waveform memory.
        del waveform

        # Apply spectrogram-level transforms.
        for transform in self.spec_transforms:
            if isinstance(transform, RandomTimeStretch_spec) and not torch.is_complex(spec):
                raise ValueError("RandomTimeStretch_spec requires a complex spectrogram. "
                                 "Set use_complex_for_time_stretch=True and provide proper STFT parameters.")
            spec = transform(spec)

        return spec

