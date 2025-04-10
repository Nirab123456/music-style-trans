import os
import csv
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T


# -----------------------------------------------------------------------------
# Simple Audio Loader
# -----------------------------------------------------------------------------

class SimpleAudioIO:
    def load(self, path: Union[str, Path], sample_rate: Optional[int] = None, duration: Optional[float] = None) -> Tuple[torch.Tensor, int]:
        path = str(path)
        waveform, sr = torchaudio.load(path, normalize=True)
        if duration:
            num_samples = int(sr * duration)
            waveform = waveform[:, :num_samples]
        if sample_rate and sr != sample_rate:
            resampler = T.Resample(sr, sample_rate)
            waveform = resampler(waveform)
            sr = sample_rate
        return waveform, sr

    def save(self, path: Union[str, Path], waveform: torch.Tensor, sample_rate: int):
        torchaudio.save(str(path), waveform, sample_rate)

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def to_stereo(waveform: torch.Tensor) -> torch.Tensor:
    """If mono, duplicate the channel to get a stereo signal."""
    if waveform.size(0) == 1:
        return waveform.repeat(2, 1)
    return waveform[:2]

def compute_spectrogram(waveform: torch.Tensor, n_fft: int = 2048, hop_length: int = 512) -> torch.Tensor:
    """Computes the magnitude spectrogram."""
    spec = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    return spec.abs()

def time_stretch(spec: torch.Tensor, factor: float = 1.0) -> torch.Tensor:
    _, f, t = spec.shape
    new_t = int(t * factor)
    stretched = F.interpolate(spec.unsqueeze(0), size=(f, new_t), mode='bilinear', align_corners=False).squeeze(0)
    return F.pad(stretched, (0, t - stretched.size(2))) if stretched.size(2) < t else stretched[:, :, :t]

def pitch_shift(spec: torch.Tensor, semitones: float = 0.0) -> torch.Tensor:
    factor = 2 ** (semitones / 12.0)
    _, f, t = spec.shape
    new_f = int(f * factor)
    shifted = F.interpolate(spec.unsqueeze(0), size=(new_f, t), mode='bilinear', align_corners=False).squeeze(0)
    return F.pad(shifted, (0, 0, 0, f - shifted.size(1))) if shifted.size(1) < f else shifted[:, :f, :]

# -----------------------------------------------------------------------------
# AudioDatasetFolder
# -----------------------------------------------------------------------------

class AudioDatasetFolder(Dataset):
    def __init__(
        self,
        csv_file: str,
        
        audio_dir: Optional[str] = None,
        components: list[str] = None,
        sample_rate: int = 44100,
        duration: float = 20.0,
        transform=None,
        is_track_id = True,
    ):
        """
        Args:
            csv_file (str): Path to CSV index file.
            audio_dir (str, optional): Base directory path to prepend to CSV file paths.
                If None, file paths in the CSV are assumed to be absolute.
            components (List[str], required): List of component names to load.
                Default loads all components in the COMPONENT_MAP.
            sample_rate (int): Sample rate used to load and (if necessary) resample audio.
            duration (float): Duration (in seconds) of audio to load from each file.
            transform (callable, optional): Optional transform to apply on the computed spectrogram.
            is_track_id(Boolean) : If true checks for second index of csv for a unique id cordinating to the filenames of all the components.
        """
        self.is_track_id = is_track_id
        self.sample_rate = sample_rate
        self.duration = duration
        self.transform = transform
        self.audio_io = SimpleAudioIO()
        
        # Use all components by default
        if components is None:
            raise ValueError("Please provide the list of components in csv.")
        else:
            self.components = components
            
        # Convert audio_dir to Path if provided
        self.audio_dir = Path(audio_dir) if audio_dir else None
        
        self.samples = []
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Each row will have keys like 'track_id', 'subset', 'mixture', 'drums', etc.
                sample_entry = {}
                for comp in self.components:
                    if comp in row and row[comp]:
                        sample_entry[comp] = row[comp]
                    else:
                        raise ValueError(f"Component '{comp}' not found in CSV or is empty.")
                    
                if self.is_track_id:
                    # Retrieve track_id from the first column
                    first_column_key = list(row.keys())[1]  # Get the first column name
                    sample_entry['track_id'] = row[first_column_key]

                self.samples.append(sample_entry)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary where keys are component names (like 'mixture', 'drums', etc.)
        and values are the computed spectrogram tensors.
        """
        sample_info = self.samples[idx]
        spectrograms = {}
        
        for comp in self.components:
            # Resolve the path for the component
            file_path_str = sample_info[comp]
            file_path = Path(file_path_str)
            
            # If an audio_dir is provided and the file path is relative, combine them.
            if self.audio_dir and not file_path.is_absolute():
                file_path = self.audio_dir / file_path
            
            # Load audio, convert to stereo and compute spectrogram
            waveform, sr = self.audio_io.load(file_path, sample_rate=self.sample_rate, duration=self.duration)
            waveform = to_stereo(waveform)
            spec = compute_spectrogram(waveform)
            
            if self.transform:
                spec = self.transform(spec)
            spectrograms[comp] = spec
        
        if self.is_track_id:
            # Return metadata if needed
            spectrograms['track_id'] = sample_info.get('track_id', '')
            
        return spectrograms

