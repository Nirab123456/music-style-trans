import torch
import random
from typing import Tuple
import torchaudio.transforms as T
import torch.nn as nn 
import configarations.global_initial_config as GI

class RandomFrequencyMasking_spec(nn.Module):
    def __init__(
        self,
        max_freq_mask_param: int = 30,
        iid_masks: bool = False
    ):
        """
        Random frequency masking for spectrogram tensors.
        
        Args:
            max_freq_mask_param (int): Maximum frequency mask size.
            iid_masks (bool): Whether to apply different masks per example/channel.
        """
        self.max_freq_mask_param = max_freq_mask_param
        self.iid_masks = iid_masks
        super(RandomFrequencyMasking_spec, self).__init__()


    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency masking to a spectrogram.

        Args:
            spectrogram (torch.Tensor): Spectrogram tensor of shape (channel, freq, time) or (batch, channel, freq, time).

        Returns:
            torch.Tensor: Masked spectrogram.
        """
        # Randomly select mask param for this call
        freq_mask_param = random.randint(1, self.max_freq_mask_param)

        # Create the transform with the selected param
        transform = T.FrequencyMasking(freq_mask_param=freq_mask_param, iid_masks=self.iid_masks)

        # Apply it to the spectrogram
        return transform(spectrogram)

class RandomTimeMasking_spec(nn.Module):
    def __init__(
        self,
        max_time_mask_param: int = 80,
        iid_masks: bool = False,
        max_proportion: float = 1.0
    ):
        """
        Random time masking for spectrogram tensors.

        Args:
            max_time_mask_param (int): Maximum length of the time mask.
            iid_masks (bool): Whether to apply different masks per example/channel.
            max_proportion (float): Maximum proportion of time steps that can be masked.
        """
        self.max_time_mask_param = max_time_mask_param
        self.iid_masks = iid_masks
        self.max_proportion = max_proportion
        super(RandomTimeMasking_spec, self).__init__()


    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply time masking to a spectrogram.

        Args:
            spectrogram (torch.Tensor): Spectrogram tensor of shape (channel, freq, time) or (batch, channel, freq, time).

        Returns:
            torch.Tensor: Time-masked spectrogram.
        """
        # Randomly select a time_mask_param within bounds
        time_mask_param = random.randint(1, self.max_time_mask_param)
        p = random.uniform(0.0, self.max_proportion)

        # Create the transform
        transform = T.TimeMasking(time_mask_param=time_mask_param, iid_masks=self.iid_masks, p=p)
        t_spec = transform(spectrogram)

        # Apply it
        return t_spec

class RandomTimeStretch_spec(nn.Module):
    def __init__(
        self,
        n_fft: int = GI.N_FFT,
        hop_length: int = 512,  # Hop length for STFT (as integer).
        rate_range: Tuple[float, float] = (0.1, 0.9)
    ):
        """
        Random time-stretching for complex spectrograms.
        
        Args:
            n_freq (int): Number of frequency bins from the STFT. For a given n_fft, n_freq = n_fft // 2 + 1.
            hop_length (int): Hop length used during STFT.
            rate_range (Tuple[float, float]): Range of time-stretching rates to randomly choose from.
        """
        self.n_freq = (n_fft // 2) +1
        self.hop_length = hop_length   # Removed trailing comma so that hop_length is an integer.
        self.rate_range = rate_range
        super(RandomTimeStretch_spec, self).__init__()

        
        # Create the TimeStretch transform from torchaudio.
        # Ensure that n_freq passed here matches the number of frequency bins in your STFT.
        self.transform = T.TimeStretch(n_freq=self.n_freq, hop_length=hop_length)

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply random time-stretching to a complex spectrogram.
        
        Args:
            spec (torch.Tensor): Complex-valued spectrogram with shape [channel, n_freq, time]
                                  (note: if using torch.stft with return_complex=True, the shape is
                                  (channels, n_freq, time)).
        Returns:
            torch.Tensor: Time-stretched spectrogram.
        """
        # Ensure that the input spec is complex.
        if not torch.is_complex(spec):
            raise ValueError("Input spectrogram must be a complex tensor for TimeStretch.")
        
        # Randomly choose the time stretching rate.
        rate = random.uniform(*self.rate_range)
        # Compute the transformed spectrogram.
        trans_spec = self.transform(spec, rate)
        trans_spec = trans_spec.abs()


        # Apply time stretching using the transform.
        return trans_spec