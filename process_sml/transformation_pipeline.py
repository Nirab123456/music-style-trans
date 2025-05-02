# my_pipeline.py
import torch
import torch.nn as nn
import typing 


# Import helper functions and individual transform classes.
from .transformation_utlis import compute_spectrogram, to_stereo , adjust_phase_shape, adjust_spec_shape 

from .spec_transform import (
    RandomFrequencyMasking_spec,
    RandomTimeMasking_spec,
    RandomTimeStretch_spec
)
import torchaudio
import configarations.global_initial_config as GI


class MyPipeline(nn.Module):
    def __init__(self,
                 wav_transforms: nn.Module = None,
                 spec_transforms: nn.Module = None,
                 shape_of_untransformed_size:torch.Size = None,
                 input_name : typing.Optional[str] = None,
                 perriferal_name: typing.Optional[typing.List[str]] = None,
                 shape_of_first_wav_tensor : torch.Size = None,
                 ):
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
        self.shape_of_first_nontransformed_spec_sample = shape_of_untransformed_size
        self.input_name =input_name
        self.perriferal_name = perriferal_name

        # If user passed None, skip transforms entirely
        self.wav_transforms = wav_transforms  # keep None to skip
        self.spec_transforms = list(spec_transforms) if spec_transforms is not None else []

        # Separate transforms into complex-only and real-only lists
        self._complex_transforms = [
            t for t in self.spec_transforms if isinstance(t, RandomTimeStretch_spec)
        ]
        self._real_transforms = [
            t for t in self.spec_transforms if not isinstance(t, RandomTimeStretch_spec)
        ]
        self._use_complex = bool(self._complex_transforms)
        self.melscale_transform = torchaudio.transforms.MelScale(sample_rate=GI.SAMPLE_RATE, n_stft=GI.N_FFT // 2 + 1)

    def forward(self, waveform: torch.Tensor, component: str ) -> torch.Tensor:
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
        # Bypass if not target component
        if component != self.input_name and (
            self.perriferal_name is None or component not in self.perriferal_name
        ):
            waveform = to_stereo(waveform)
            full_spec = compute_spectrogram(waveform)
            spec = full_spec.abs()
            spec = self.melscale_transform(spec)

            return spec

        # Apply waveform transforms if provided
        if self.wav_transforms is not None:
            waveform = self.wav_transforms(waveform)  # type: ignore

        # Compute spectrogram and apply complex transforms if any
        if self._use_complex:
            complex_spec = compute_spectrogram(waveform)
            for t in self._complex_transforms:
                complex_spec = t(complex_spec)
            spec = complex_spec.abs()
        else:
            full_spec = compute_spectrogram(waveform)
            spec = full_spec.abs()

        # Apply real-only spectrogram transforms
        for t in self._real_transforms:
            spec = t(spec)

        # Adjust shapes if needed
        if self.shape_of_first_nontransformed_spec_sample is not None:
            target = self.shape_of_first_nontransformed_spec_sample[-2:]
            spec = adjust_spec_shape(spec, target)

        # Concatenate magnitude and phase
        spec = self.melscale_transform(spec)
        return spec