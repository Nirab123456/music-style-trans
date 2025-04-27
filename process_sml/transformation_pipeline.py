# my_pipeline.py
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import typing 
import torch.nn.functional as F


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


def adjust_phase_shape(phase: torch.Tensor, target_shape: typing.Tuple[int, int]) -> torch.Tensor:
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


def adjust_spec_shape(spec: torch.Tensor, target_shape: typing.Tuple[int, int]) -> torch.Tensor:
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

def get_shape_first_sample(waveform):
    waveform = to_stereo(waveform)
    spec = compute_spectrogram(waveform)
    spec = spec.abs()
    shape = spec.shape

    return shape
    

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
                RandomTimeStretch_spec(n_fft=2048, hop_length=256, rate_range=(0.8, 1.25))
            ])
        else:
            self.spec_transforms = spec_transforms

        self._complex_transforms = [t for t in self.spec_transforms if isinstance(t, RandomTimeStretch_spec)] 
        self._real_transforms    = [t for t in self.spec_transforms if not isinstance(t, RandomTimeStretch_spec)] 
        self._use_complex = len(self._complex_transforms) > 0 


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
        waveform = to_stereo(waveform)
        phase = None

        if component == self.input_name or (self.perriferal_name != None and component in self.perriferal_name):
            
            if self.wav_transforms:
                waveform = self.wav_transforms(waveform)

            if self._use_complex: 
                complex_spec = compute_spectrogram(waveform) 
            for t in self._complex_transforms:         
                complex_spec = t(complex_spec) 
                phase = complex_spec.angle() 
                spec  = complex_spec.abs() 
            else: 
                spec  = compute_spectrogram(waveform).abs() 
                phase = spec.angle()
                spec = spec.abs()
            
            for t in self._real_transforms: 
                spec = t(spec)             
            spec = adjust_spec_shape(spec, self.shape_of_first_nontransformed_spec_sample[-2:])
            phase = adjust_phase_shape(phase, self.shape_of_first_nontransformed_spec_sample[-2:])
            combined = torch.cat((spec, phase), dim=0)  # (4, H, W)
            return combined
        else:
            return waveform
