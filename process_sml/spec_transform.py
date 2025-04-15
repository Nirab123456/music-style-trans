import torch
import random
from typing import Tuple
import torchaudio.transforms as T
import configarations
import configarations.global_initial_config

class RandomFrequencyMasking_spec:
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

class RandomTimeMasking_spec:
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

        # Apply it
        return transform(spectrogram)

class RandomTimeStretch_spec:
    def __init__(
        self,
        n_freq: int = 201,
        hop_length: int = None,
        rate_range: Tuple[float, float] = (0.8, 1.25)
    ):
        """
        Random time-stretching for complex spectrograms.

        Args:
            n_freq (int): Number of frequency bins (from STFT). Must match the input spectrogram.
            hop_length (int or None): Hop length for STFT. Default is n_fft // 2.
            rate_range (Tuple[float, float]): Range of time-stretching rates to randomly choose from.
        """
        self.n_freq = n_freq
        self.hop_length = 2048,
        self.rate_range = rate_range
        self.transform = T.TimeStretch(n_freq=n_freq, hop_length=hop_length)

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply random time-stretching to a complex spectrogram.

        Args:
            spec (torch.Tensor): Complex-valued spectrogram (e.g., shape [channel, freq, time, complex=2]).

        Returns:
            torch.Tensor: Time-stretched spectrogram.
        """
        # Randomly choose the stretching rate
        rate = random.uniform(*self.rate_range)

        # Apply time-stretching
        return self.transform(spec, rate)
