#!/usr/bin/env python
# coding: utf8
"""
This module implements an audio data preprocessing pipeline using PyTorch.
Data preprocessing includes audio loading, spectrogram computation, cropping,
and data augmentation (random time crop, time stretch, and pitch shift). The
dataset yields a dictionary of spectrograms computed from the input audio files.
"""

import os
import csv
import time
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import cython

# -----------------------------------------------------------------------------
# Default Audio Parameters
# -----------------------------------------------------------------------------

DEFAULT_AUDIO_PARAMS: Dict[str, Any] = {
    "instrument_list": ["vocals", "accompaniment"],
    "mix_name": "mix",
    "sample_rate": 44100,
    "frame_length": 4096,
    "frame_step": 1024,
    "T": 512,           # Target time frames for crop (e.g., 11.88s if using frame_step=1024)
    "F": 1024,          # Target frequency bins
    "n_channels": 2,    # Expected number of channels (stereo)
}

# -----------------------------------------------------------------------------
# Simple Audio Loader
# -----------------------------------------------------------------------------
class SimpleAudioIO:
    def load(
        self,
        path: Union[str, Path],
        sample_rate: Optional[int] = None,
        duration: Optional[float] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Loads an audio file from the given path, normalizes it, optionally crops its duration,
        and resamples it if required.
        """
        path_str: str = str(path)
        # Convert to bytes with a null terminator (if used by a C function, though torchaudio.load accepts str)
        path_bytes: bytes = path_str.encode('utf-8') + b'\x00'
        c_path: cython.p_char = path_bytes  # C-style string (not used further in torchaudio.load)

        # Use path_str since torchaudio.load accepts a string path.
        waveform, sr = torchaudio.load(path_str, normalize=True)

        if duration is not None:
            num_samples: int = int(sr * duration)
            waveform = waveform[:, :num_samples]

        if sample_rate is not None and sr != sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)
            sr = sample_rate

        return waveform, sr

    def save(self, path: Union[str, Path], waveform: torch.Tensor, sample_rate: int) -> None:
        """Saves the given waveform to the specified path."""
        torchaudio.save(str(path), waveform, sample_rate)

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

# -----------------------------------------------------------------------------
# AudioDatasetFolder
# -----------------------------------------------------------------------------

class AudioDatasetFolder(Dataset):
    def __init__(
        self,
        csv_file: str,
        audio_dir: Optional[str] = None,
        components: List[str] = None,
        sample_rate: int = 44100,
        duration: float = 20.0,
        transform: Optional[Union[Callable[[torch.Tensor], torch.Tensor], List[Callable[[torch.Tensor], torch.Tensor]]]] = None,
        is_track_id: bool = True,
    ) -> None:
        """
        Args:
            csv_file: Path to CSV index file.
            audio_dir: Base directory path to prepend to CSV file paths. If None, file paths are assumed absolute.
            components: List of component names to load.
            sample_rate: Sample rate used for audio loading and resampling.
            duration: Duration (in seconds) of audio to load from each file.
            transform: Optional transform(s) to apply on the computed spectrogram.
            is_track_id: If true, use a track identifier from the CSV.
        """
        self.is_track_id = is_track_id
        self.sample_rate = sample_rate
        self.duration = duration
        self.audio_io = SimpleAudioIO()
        
        if components is None:
            raise ValueError("Please provide the list of components in CSV.")
        else:
            self.components = components
            
        if transform is None:
            self.transforms: List[Callable[[torch.Tensor], torch.Tensor]] = []
        elif callable(transform):
            self.transforms = [transform]
        elif isinstance(transform, (list, tuple)):
            if all(callable(t) for t in transform):
                self.transforms = list(transform)
            else:
                raise ValueError("All elements in transform must be callable.")
        else:
            raise TypeError("transform must be either a callable or a list/tuple of callables.")

        self.audio_dir = Path(audio_dir) if audio_dir else None

        self.samples: List[Dict[str, str]] = []
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_entry: Dict[str, str] = {}
                for comp in self.components:
                    if comp in row and row[comp]:
                        sample_entry[comp] = row[comp]
                    else:
                        raise ValueError(f"Component '{comp}' not found in CSV or is empty.")
                if self.is_track_id:
                    keys = list(row.keys())
                    sample_entry['track_id'] = row[keys[1]] if len(keys) > 1 else ""
                self.samples.append(sample_entry)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary where keys are component names (e.g., 'mix', 'drums', etc.)
        and values are the computed spectrogram tensors.
        """
        sample_info: Dict[str, str] = self.samples[idx]
        spectrograms: Dict[str, torch.Tensor] = {}
        
        for comp in self.components:
            file_path_str: str = sample_info[comp]
            file_path = Path(file_path_str)
            if self.audio_dir and not file_path.is_absolute():
                file_path = self.audio_dir / file_path
            
            waveform, sr = self.audio_io.load(file_path, sample_rate=self.sample_rate, duration=self.duration)
            waveform = to_stereo(waveform)
            spec = compute_spectrogram(waveform)
            
            for transform in self.transforms:
                spec = transform(spec)
            spectrograms[comp] = spec
        
        if self.is_track_id:
            spectrograms['track_id'] = sample_info.get('track_id', '')
            
        return spectrograms

# -----------------------------------------------------------------------------
# Optional: DataLoader Getter Function
# -----------------------------------------------------------------------------

def get_dataloader(
    csv_file: str,
    audio_dir: Optional[str],
    components: List[str],
    batch_size: int = 8,
    shuffle: bool = True,
    transforms: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None
) -> DataLoader:
    dataset = AudioDatasetFolder(
        csv_file=csv_file,
        audio_dir=audio_dir,
        components=components,
        sample_rate=DEFAULT_AUDIO_PARAMS["sample_rate"],
        duration=20.0,
        transform=transforms,
        is_track_id=True,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

# -----------------------------------------------------------------------------
# Example Augmentation Pipeline
# -----------------------------------------------------------------------------
def augmentation_pipeline(spec: torch.Tensor) -> torch.Tensor:
    """
    Composite transformation to apply data augmentation:
      1. Random time crop to T frames.
      2. Random time stretch.
      3. Random pitch shift.
    """
    # Random time crop: crop to DEFAULT_AUDIO_PARAMS["T"] time frames.
    spec = random_time_crop(spec, target_time=DEFAULT_AUDIO_PARAMS["T"])
    # Random time stretch.
    spec = random_time_stretch(spec, factor_range=(0.9, 1.1))
    # Random pitch shift.
    spec = random_pitch_shift(spec, shift_range=(-1.0, 1.0))
    return spec

# -----------------------------------------------------------------------------
# Main (Example Usage)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage:
    csv_file = "path/to/your_index.csv"  # Update with your CSV path.
    audio_dir = "path/to/audio_files"    # Update with your audio directory.
    components = [DEFAULT_AUDIO_PARAMS["mix_name"]] + DEFAULT_AUDIO_PARAMS["instrument_list"]

    # Create DataLoader with augmentation pipeline.
    loader = get_dataloader(
        csv_file=csv_file,
        audio_dir=audio_dir,
        components=components,
        batch_size=8,
        shuffle=True,
        transforms=[augmentation_pipeline]
    )

    # Iterate over a single batch.
    for batch in loader:
        input_dict, output_dict = batch, {}  # In this simplified version, the dataset returns one dictionary.
        # Here you can print shapes or further process the batch.
        print("Batch sample keys:", list(input_dict.keys()))
        break
